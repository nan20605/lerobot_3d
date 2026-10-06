"""Client UI reuses Sergio's ViserSceneViewer and YAM mesh-state adapter."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .transport import Client, save_capture


def run_viewer(url, model_dir, port=8080, nominal_preview=False, capture_dir="data/raw"):
    import cv2

    from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
    from lerobot_3d.yam import YamRobotState

    from .calibration import colored_camera_cloud
    from .geometry import apply, quaternion, transform

    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    rig = json.loads((model_dir / "rig.json").read_text())
    viewer = ViserSceneViewer(host="127.0.0.1", port=port, controls=True)
    viewer.server.scene.set_up_direction("+z")
    viewer.server.gui.configure_theme(show_share_button=False)

    @viewer.server.on_client_connect
    def initial_view(client):
        client.camera.position = (1.15, -1.45, 0.95)
        client.camera.look_at = (0.2, -0.305, 0.32)
        client.camera.up_direction = (0, 0, 1)

    client = Client(url)
    # Image transfer/compression must not set the robot's feedback refresh rate.
    packet_lock = threading.Lock()
    packet_cache = {}
    stop_images = threading.Event()

    def receive_images():
        while not stop_images.is_set():
            try:
                metadata, pixels = client.observation()
                with packet_lock:
                    packet_cache.update(meta=metadata, arrays=pixels, error=None)
            except Exception as e:  # noqa: BLE001 — surface background transport/codec failures in the UI.
                with packet_lock:
                    packet_cache["error"] = str(e)
            stop_images.wait(0.03)

    image_thread = threading.Thread(target=receive_images, daemon=True)
    image_thread.start()
    viewer.server.gui.add_markdown(
        "# YAM station\nJoint feedback + RGB + metric depth. **Read-only viewer**."
    )
    viewer.server.gui.add_markdown(
        "**Calibration:** "
        + ("validated" if manifest["calibrated"] else "NOMINAL / incomplete")
        + "\n\n"
        + "\n".join("- " + x for x in manifest["calibration_blockers"])
    )
    adapters = {
        side: YamRobotState(model_dir / "yam_bimanual.urdf", side, i)
        for i, side in enumerate(("left", "right"))
    }
    space = viewer.server.gui.add_dropdown(
        "3D view",
        ("Robot world", "Left camera", "Right camera", "Overhead camera"),
        initial_value="Robot world",
    )
    cloud_note = viewer.server.gui.add_markdown(
        "World fusion requires calibrated camera mounts. Camera views show live local 3D independently."
    )
    viewer.server.scene.add_grid(
        "/metre_reference",
        width=1.4,
        height=1.4,
        cell_size=0.1,
        section_size=0.5,
        position=(0.2, -0.305, 0),
    )
    loaded = set()
    images = {}
    stats = viewer.server.gui.add_markdown("Waiting for observations")
    prev_id = None
    last_received = None
    count = 0
    begin = time.monotonic()
    last_space = None
    last_image_id = None
    nominal = {side: {"position_rad": [0, 1.5, 1.5, 0, 0, 0], "gripper_open": 0.5} for side in adapters}
    top = transform(rig["transforms"]["top_camera"]["T_parent_child"])
    overhead_frame = viewer.server.scene.add_frame(
        "/overhead_nominal_optical",
        position=top[:3, 3],
        wxyz=quaternion(top),
        axes_length=0.08,
    )
    overhead_label = viewer.server.scene.add_label(
        "/overhead_label",
        text="Overhead pose: " + rig["transforms"]["top_camera"]["status"],
        position=top[:3, 3],
    )
    try:
        while not viewer.quit:
            try:
                feedback = client.state()
                with packet_lock:
                    if packet_cache.get("error") or "meta" not in packet_cache:
                        raise RuntimeError(packet_cache.get("error") or "Waiting for first image packet")
                    meta, arrays = packet_cache["meta"], packet_cache["arrays"]
                new_image = meta["observation_id"] != last_image_id
                space_changed = space.value != last_space
                if feedback["observation_id"] == prev_id:
                    if last_received and time.monotonic() - last_received > 0.5:
                        raise RuntimeError("Observation sequence stopped")
                    time.sleep(0.02)
                    continue
                prev_id = feedback["observation_id"]
                last_received = time.monotonic()
                count += 1
                arms = feedback["arms"]
                preview = not arms and nominal_preview
                shown = nominal if preview else arms
                snapshots = []
                for side, state in shown.items():
                    if side not in loaded:
                        viewer.load_static_meshes(adapters[side].get_static_meshes(), adapters[side].index)
                        loaded.add(side)
                    snapshots.append(adapters[side].get_robot_snapshot(state))
                    viewer._robot_frames[adapters[side].index].visible = True
                for side in loaded - set(shown):
                    viewer._robot_frames[adapters[side].index].visible = False
                cloud_parts = []
                color_parts = []
                missing = []
                # Use the joint state carried WITH an image packet for its wrist
                # transform. Newer feedback only drives the visible arm meshes.
                image_arms = meta["arms"]
                world_poses = next(iter(adapters.values())).model.fk(image_arms) if image_arms else {}
                if space.value == "Robot world":
                    for name in meta["cameras"]:
                        frame = "top_camera" if name == "overhead" else name + "_camera"
                        calibration = rig["transforms"].get(frame)
                        if calibration is None or calibration["status"] != "validated":
                            missing.append(name)
                            continue
                        if calibration["parent"] != "world" and name not in image_arms:
                            missing.append(name + " (joint feedback missing)")
                            continue
                        t = (
                            transform(calibration["T_parent_child"])
                            if calibration["parent"] == "world"
                            else world_poses[frame]
                        )
                        points, colors = colored_camera_cloud(meta, arrays, name)
                        cloud_parts.append(apply(t, points))
                        color_parts.append(colors)
                    cloud_note.content = (
                        "World fusion unavailable for: " + ", ".join(missing)
                        if missing
                        else "Showing calibrated world-frame clouds."
                    )
                else:
                    name = space.value.split()[0].lower()
                    if name in meta["cameras"]:
                        points, colors = colored_camera_cloud(meta, arrays, name)
                        # Display optical +Z forward as world +X, optical +Y down as world -Z.
                        display = np.array(
                            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                            float,
                        )
                        cloud_parts.append(apply(display, points))
                        color_parts.append(colors)
                        cloud_note.content = (
                            f"Live {name} RGB-D in its own camera frame. Not aligned to robot world."
                        )
                    else:
                        cloud_note.content = f"No {name} camera stream available."
                scene = np.concatenate(cloud_parts) if cloud_parts else np.empty((0, 3))
                colors = np.concatenate(color_parts) if color_parts else None
                if space.value != last_space and (space.value == "Robot world" or len(scene)):
                    centre = (
                        np.array([0.2, -0.305, 0.32])
                        if space.value == "Robot world"
                        else np.median(scene, axis=0)
                    )
                    offset = (
                        np.array([0.95, -1.15, 0.63])
                        if space.value == "Robot world"
                        else np.array([-0.65, -0.65, 0.45])
                    )
                    for client_handle in viewer.server.get_clients().values():
                        client_handle.camera.look_at = centre
                        client_handle.camera.position = centre + offset
                    last_space = space.value
                for snapshot in snapshots:
                    snapshot.pcd = np.empty((0, 3))
                viewer.update(scene, colors, snapshots, update_scene=new_image or space_changed)
                last_image_id = meta["observation_id"]
                for handle in viewer._robot_handles.values():
                    handle.visible = False
                for side in loaded:
                    viewer._robot_frames[adapters[side].index].visible = (
                        space.value == "Robot world" and side in shown
                    )
                overhead_frame.visible = space.value == "Robot world"
                overhead_label.visible = space.value == "Robot world"
                lines = [
                    f"**Source: {meta['mode']}**",
                    f"State response refresh: {count / (time.monotonic() - begin):.1f} Hz",
                ]
                if preview:
                    lines.append(
                        "**Arms show a NOMINAL example pose: no measured joint states in this capture.**"
                    )
                elif not arms:
                    lines.append("**No arm feedback. Robot poses are unavailable.**")
                    if feedback.get("joint_source_error"):
                        lines.append("Joint service: " + feedback["joint_source_error"])
                        lines.append("Check CAN interfaces and the supervised arm service on the Jetson.")
                for side, state in arms.items():
                    lines.append(f"{side} radians: " + ", ".join(f"{x:.3f}" for x in state["position_rad"]))
                    if meta["mode"] != "replay":
                        lines.append(
                            f"{side} server poll age: {1000 * (feedback['server_monotonic_s'] - state['host_monotonic_s']):.1f} ms"
                        )
                camera_times = []
                for name, info in meta["cameras"].items():
                    for kind in ("rgb", "depth_raw"):
                        if not new_image:
                            continue
                        a = arrays.get(name + "_" + kind)
                        if a is None:
                            continue
                        if kind == "depth_raw":
                            metres = a * info["depth"]["depth_scale_m_per_unit"]
                            value = np.clip(metres / 1.5 * 255, 0, 255).astype(np.uint8)
                            a = cv2.cvtColor(
                                cv2.applyColorMap(value, cv2.COLORMAP_TURBO),
                                cv2.COLOR_BGR2RGB,
                            )
                            a[metres == 0] = 0
                        key = (name, kind)
                        if key not in images:
                            images[key] = viewer.server.gui.add_image(
                                a,
                                label=f"{name} {kind}"
                                + (" (0–1.5 m; black=invalid)" if kind == "depth_raw" else ""),
                            )
                        else:
                            images[key].image = a
                    lines.append(f"{name}: {info['serial']}; frame {info['color'].get('frame_number', '?')}")
                    camera_times.append(info.get("host_timestamp_ns", 0) / 1e6)
                if len(camera_times) > 1:
                    lines.append(
                        f"Camera host-arrival span: {max(camera_times) - min(camera_times):.2f} ms (not exposure sync)"
                    )
                stats.content = "\n\n".join(lines)
                viewer.set_status(
                    meta["mode"].upper()
                    + " — "
                    + ("nominal example arm poses" if preview else "receiving observations")
                )
                if viewer.capture:
                    viewer.capture = False
                    p = Path(capture_dir) / time.strftime("%Y%m%d_%H%M%S")
                    save_capture(client, p)
                    viewer.server.gui.add_markdown(f"Saved `{p.resolve()}`")
                if viewer.save_subgoal:
                    viewer.save_subgoal = False
                    viewer.server.gui.add_markdown(
                        "Subgoal motion is not enabled in this read-only viewer. Use YamRemoteRobot with a supervised server session."
                    )
            except Exception as e:  # noqa: BLE001 — hide stale poses and allow the viewer to reconnect.
                viewer.set_status(f"STALE / DISCONNECTED: {e}")
                for frame in viewer._robot_frames.values():
                    frame.visible = False
                time.sleep(0.5)
            time.sleep(0.03)
    finally:
        stop_images.set()
        image_thread.join(timeout=6)
        viewer.close()
