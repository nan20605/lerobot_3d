"""Jetson-only acquisition. Hardware imports and motor initialization are explicit."""

from __future__ import annotations

import copy
import fcntl
import json
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .models import I2RT_REV


def intrinsics(i):
    return {
        "width": i.width,
        "height": i.height,
        "fx": i.fx,
        "fy": i.fy,
        "ppx": i.ppx,
        "ppy": i.ppy,
        "model": str(i.model),
        "coeffs": list(i.coeffs),
    }


class RealSenseCamera:
    def __init__(self, serial, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = serial
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.stop_event = threading.Event()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        extr = depth.get_extrinsics_to(color)
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        self.info = {
            "serial": serial,
            "name": device.get_info(rs.camera_info.name),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "color": {
                "intrinsics": intrinsics(color.get_intrinsics()),
                "encoding": "rgb8",
            },
            "depth": {
                "intrinsics": intrinsics(depth.get_intrinsics()),
                "encoding": "z16",
                "depth_scale_m_per_unit": device.first_depth_sensor().get_depth_scale(),
            },
            "depth_to_color": {
                "rotation": list(extr.rotation),
                "translation_m": list(extr.translation),
            },
            "depth_registration": "raw_depth_optical_frame",
            "timestamp_note": "host arrival; not exposure synchronization",
        }
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(2000)
                mono, unix = time.monotonic(), time.time_ns()
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                info = copy.deepcopy(self.info)
                info.update(host_monotonic_s=mono, host_timestamp_ns=unix)
                for name, f in (("color", color), ("depth", depth)):
                    info[name].update(
                        camera_timestamp_ms=f.get_timestamp(),
                        frame_number=f.get_frame_number(),
                        timestamp_domain=str(f.get_frame_timestamp_domain()),
                    )
                value = (
                    info,
                    np.asanyarray(color.get_data()).copy(),
                    np.asanyarray(depth.get_data()).copy(),
                )
                with self.lock:
                    self.latest = value
                    self.error = None
            except Exception as e:  # noqa: BLE001 — publish SDK thread failures to the receiver.
                with self.lock:
                    self.error = str(e)

    def read(self):
        with self.lock:
            if self.error or self.latest is None:
                raise RuntimeError(self.error or f"Camera {self.serial} warming up")
            if time.monotonic() - self.latest[0]["host_monotonic_s"] > 0.5:
                raise RuntimeError(f"Camera {self.serial} stale")
            return self.latest

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        self.pipeline.stop()


class LiveProvider:
    def __init__(self, camera_serials, channels=None, ack_motor_initialization=False, joint_source=None):
        self.cameras = {}
        self.robots = {}
        self.lock_files = ExitStack()
        self.feedback_watch = {}
        self.joint_client = None
        if joint_source:
            from urllib.parse import urlparse

            from .transport import Client

            if channels or urlparse(joint_source).hostname != "127.0.0.1":
                raise ValueError("Joint source must be a separate loopback service on the same Jetson")
            self.joint_client = Client(joint_source, timeout=0.2)
        if channels and not ack_motor_initialization:
            raise ValueError(
                "I2RT startup enables motors and may calibrate/move grippers; requires --ack-motor-initialization at a supervised rig"
            )
        try:
            if channels:
                from i2rt.robots import get_robot
                from i2rt.robots.utils import ArmType, GripperType

                root = Path(get_robot.__file__).resolve().parents[2]
                revision = subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip()
                if revision != I2RT_REV:
                    raise ValueError(f"Unreviewed I2RT revision {revision}; expected {I2RT_REV}")
                if len(set(channels.values())) != len(channels):
                    raise ValueError("Two arms cannot own the same CAN interface")
                for side, channel in channels.items():
                    if not channel.replace("_", "").isalnum():
                        raise ValueError("Invalid CAN interface name")
                    # ExitStack retains the lock until provider shutdown, including partial startup failure.
                    f = self.lock_files.enter_context(open(f"/tmp/yamming-{channel}.lock", "a"))  # noqa: SIM115
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.robots[side] = get_robot.get_yam_robot(
                        channel=channel,
                        arm_type=ArmType.YAM,
                        gripper_type=GripperType.LINEAR_4310,
                        zero_gravity_mode=False,
                        enable_auto_recovery=False,
                    )
            for name, serial in camera_serials.items():
                self.cameras[name] = RealSenseCamera(serial)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        meta = {
            "mode": "live",
            "arms": {},
            "cameras": {},
            "joint_timestamp_kind": "SDK poll completion; exposure-aligned interpolation unavailable",
        }
        arrays = {}
        if self.joint_client:
            try:
                state = self.joint_client.state()
                if state["mode"] != "live" or time.monotonic() - state["server_monotonic_s"] > 0.2:
                    raise ValueError("Joint source is not fresh live feedback")
                meta["arms"] = state["arms"]
            except Exception as e:  # noqa: BLE001 — keep camera acquisition available if the joint service fails.
                meta["joint_source_error"] = str(e)
        for side, robot in self.robots.items():
            # Pinned SDK loop death must not masquerade as fresh feedback.
            if not robot._server_thread.is_alive() or not robot.motor_chain.running:
                raise RuntimeError(f"{side} I2RT control loop stopped")
            # The SDK stamps get_state() at read time. Track its feedback object
            # as well, so a frozen CAN receive loop cannot look fresh indefinitely.
            with robot.motor_chain.state_lock:
                feedback = robot.motor_chain.state
            previous, changed_at = self.feedback_watch.get(side, (None, time.monotonic()))
            if feedback is previous and time.monotonic() - changed_at > 0.2:
                raise RuntimeError(f"{side} CAN feedback stopped updating")
            if feedback is not previous:
                self.feedback_watch[side] = (feedback, time.monotonic())
            obs = robot.get_observations()
            q = np.asarray(obs["joint_pos"]).copy()
            v = np.asarray(obs["joint_vel"]).copy()
            g = float(np.asarray(obs["gripper_pos"]).item())
            if q.shape != (6,) or v.shape != (6,) or not np.isfinite(np.r_[q, v, g]).all():
                raise RuntimeError("Invalid SDK observation")
            meta["arms"][side] = {
                "position_rad": q.tolist(),
                "velocity_rad_s": v.tolist(),
                "effort_nm": np.asarray(obs["joint_eff"]).tolist(),
                "gripper_open": g,
                "host_monotonic_s": time.monotonic(),
                "host_timestamp_ns": time.time_ns(),
            }
        for name, camera in self.cameras.items():
            info, color, depth = camera.read()
            meta["cameras"][name] = info
            arrays[f"{name}_rgb"] = color
            arrays[f"{name}_depth_raw"] = depth
        return meta, arrays

    def command(self, side, target):
        self.robots[side].command_joint_pos(target.copy())

    def hold(self):
        for robot in self.robots.values():
            if robot._server_thread.is_alive():
                robot.command_joint_pos(robot.get_joint_pos().copy())

    def close(self):
        for camera in self.cameras.values():
            camera.close()
        for robot in self.robots.values():
            # Pinned SDK does not retain/join its CAN thread in close(). Join it
            # before closing the socket to avoid a concurrent send on a closed FD.
            robot._stop_event.set()
            robot._server_thread.join(timeout=3)
            chain = robot.motor_chain
            chain.running = False
            for thread in threading.enumerate():
                if getattr(thread, "_target", None) == chain._set_torques_and_update_state:
                    thread.join(timeout=3)
            with chain.command_lock:
                chain.close()
            robot.stop_mcap_recording()
        self.lock_files.close()


class SimulationProvider:
    """Real MuJoCo actuator stepping; explicitly marked simulated observations."""

    poll_period_s = 0.05

    def __init__(self, model_path, render_cameras=False):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.rig = json.loads((Path(model_path).parent / "rig.json").read_text())
        self.render_cameras = render_cameras
        self.renderers = {}
        self.targets = {side: np.array([0, 1.5, 1.5, 0, 0, 0, 0.5]) for side in ("left", "right")}
        for side, target in self.targets.items():
            for i in range(1, 9):
                j = self.model.joint(f"{side}_joint{i}")
                self.data.qpos[j.qposadr[0]] = target[i - 1] if i <= 6 else target[6] * j.range[1]
        mujoco.mj_forward(self.model, self.data)
        self._set_controls()

    def _set_controls(self):
        for side, q in self.targets.items():
            for i in range(1, 8):
                self.data.ctrl[self.model.actuator(f"{side}_position{i}").id] = (
                    q[i - 1] if i <= 6 else q[6] * self.model.joint(f"{side}_joint7").range[1]
                )

    def snapshot(self):
        with self.lock:
            self._set_controls()
            for _ in range(25):
                self.mujoco.mj_step(self.model, self.data)
            arms = {}
            for side in self.targets:
                ids = [self.model.joint(f"{side}_joint{i}").qposadr[0] for i in range(1, 7)]
                j = self.model.joint(f"{side}_joint7")
                arms[side] = {
                    "position_rad": self.data.qpos[ids].tolist(),
                    "velocity_rad_s": self.data.qvel[ids].tolist(),
                    "gripper_open": float(np.clip(self.data.qpos[j.qposadr[0]] / j.range[1], 0, 1)),
                    "host_monotonic_s": time.monotonic(),
                    "host_timestamp_ns": time.time_ns(),
                }
            cameras = {}
            arrays = {}
            if self.render_cameras:
                options = self.mujoco.MjvOption()
                options.sitegroup[:] = 0
                for name, info in self.rig["cameras"].items():
                    if "depth_intrinsics" not in info or "intrinsics" not in info:
                        continue
                    side = name.removesuffix("_camera") if name != "top_camera" else "overhead"
                    cameras[side] = {
                        "serial": "SIM:" + str(info["serial"]),
                        "name": "ideal simulated " + str(info["model"]),
                        "host_monotonic_s": time.monotonic(),
                        "host_timestamp_ns": time.time_ns(),
                        "depth_to_color": info["depth_to_color"],
                        "depth_registration": "raw_depth_optical_frame",
                    }
                    for stream, key, kind in (
                        ("rectified", "intrinsics", "color"),
                        ("depth", "depth_intrinsics", "depth"),
                    ):
                        i = info[key]
                        size = (i["height"], i["width"])
                        if size not in self.renderers:
                            self.renderers[size] = self.mujoco.Renderer(
                                self.model, height=size[0], width=size[1]
                            )
                        renderer = self.renderers[size]
                        renderer.disable_depth_rendering()
                        if kind == "depth":
                            renderer.enable_depth_rendering()
                        renderer.update_scene(self.data, camera=name + "_" + stream, scene_option=options)
                        pixels = renderer.render().copy()
                        cameras[side][kind] = {
                            "intrinsics": {
                                **i,
                                "model": "none",
                                "coeffs": [0.0] * 5,
                                "rectified": True,
                            },
                            "camera_timestamp_ms": self.data.time * 1000,
                            "frame_number": int(self.data.time / 0.05),
                            "timestamp_domain": "simulation",
                        }
                        if kind == "depth":
                            scale = info["depth_scale_m_per_unit"]
                            valid = np.isfinite(pixels) & (pixels > 0) & (pixels < 65535 * scale)
                            raw = np.zeros(pixels.shape, np.uint16)
                            raw[valid] = np.rint(pixels[valid] / scale).astype(np.uint16)
                            arrays[side + "_depth_raw"] = raw
                            cameras[side][kind]["depth_scale_m_per_unit"] = scale
                        else:
                            arrays[side + "_rgb"] = pixels
            return {
                "mode": "simulation",
                "arms": arms,
                "cameras": cameras,
                "sim_time_s": self.data.time,
            }, arrays

    def command(self, side, target):
        with self.lock:
            self.targets[side] = target.copy()

    def hold(self):
        with self.lock:
            for side in self.targets:
                q = [self.data.qpos[self.model.joint(f"{side}_joint{i}").qposadr[0]] for i in range(1, 7)]
                j = self.model.joint(f"{side}_joint7")
                self.targets[side] = np.r_[q, np.clip(self.data.qpos[j.qposadr[0]] / j.range[1], 0, 1)]

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
