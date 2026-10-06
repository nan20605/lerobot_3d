"""Single CLI for local/offline work and explicitly selected Jetson acquisition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="task", required=True)
    b = s.add_parser("init-rig", help="Create explicit nominal rig config from pinned vendor station")
    b.add_argument("--vendor", default="third_party/i2rt")
    b.add_argument("--metadata", help="Optional saved RealSense factory metadata.json")
    b.add_argument("--output", default="config/rig.nominal.json")
    b = s.add_parser(
        "build",
        help="Portable vendor URDF/MJCF bundle; refuses false calibrated status",
    )
    b.add_argument("--vendor", default="third_party/i2rt")
    b.add_argument("--rig", default="config/rig.nominal.json")
    b.add_argument("--output", default="sim/generated")
    b.add_argument("--require-calibrated", action="store_true")
    for cmd in ("verify-models", "render", "audit-models", "export-bundle", "check-numerics"):
        a = s.add_parser(cmd)
        a.add_argument("--models", default="sim/generated")
        if cmd == "render":
            a.add_argument("--output", default="sim/generated/renders")
        elif cmd in ("audit-models", "export-bundle", "check-numerics"):
            a.add_argument("--output", required=True)
    b = s.add_parser("serve", help="Loopback only; access remotely with an SSH tunnel")
    mode = b.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", metavar="CAPTURE_DIR")
    mode.add_argument("--sim", action="store_true")
    mode.add_argument("--live", action="store_true")
    b.add_argument("--port", type=int, default=8765)
    b.add_argument("--models", default="sim/generated")
    b.add_argument("--camera", action="append", default=[], metavar="NAME=SERIAL")
    b.add_argument("--arm", action="append", default=[], metavar="SIDE=CAN_INTERFACE")
    b.add_argument(
        "--joint-source",
        help="Read joint states from a same-Jetson loopback service (no motor initialization)",
    )
    b.add_argument(
        "--ack-motor-initialization",
        action="store_true",
        help="SDK enables motors and may move grippers; operator at rig required",
    )
    b.add_argument("--enable-motion", action="store_true")
    b.add_argument("--token-file")
    b.add_argument(
        "--render-cameras",
        action="store_true",
        help="Render ideal camera images in --sim mode",
    )
    b = s.add_parser("view")
    b.add_argument("--url", default="http://127.0.0.1:8765")
    b.add_argument("--models", default="sim/generated")
    b.add_argument("--port", type=int, default=8080)
    b.add_argument(
        "--nominal-preview",
        action="store_true",
        help="Display labeled example arm poses if feedback is absent",
    )
    b = s.add_parser("capture")
    b.add_argument("--url", default="http://127.0.0.1:8765")
    b.add_argument("--output", required=True)
    b.add_argument("--require-joints", action="store_true")
    b.add_argument("--max-skew-ms", type=float, default=50)
    b.add_argument("--stationary-seconds", type=float, default=0.75)
    b = s.add_parser(
        "stream-check",
        help="Record client reception statistics without commanding motion",
    )
    b.add_argument("--url", default="http://127.0.0.1:8765")
    b.add_argument("--seconds", type=float, default=10)
    b.add_argument("--output", required=True)
    b = s.add_parser("board")
    b.add_argument("--output", default="calibration/board")
    b = s.add_parser("rectify")
    b.add_argument("--capture", required=True)
    b.add_argument("--camera", required=True)
    b.add_argument("--output", required=True)
    b = s.add_parser("detect-board")
    b.add_argument("--rgb", required=True)
    b.add_argument("--intrinsics", required=True)
    b.add_argument("--board", default="calibration/board/board.json")
    b.add_argument("--output", required=True)
    for cmd in ("hand-eye", "fixed-camera"):
        b = s.add_parser(cmd)
        b.add_argument("--samples", required=True)
        b.add_argument("--output", required=True)
        b.add_argument("--max-translation-m", type=float, default=0.01)
        b.add_argument("--max-rotation-deg", type=float, default=1)
    b = s.add_parser("board-sample", help="Detect a board in a paired capture and attach measured-state FK")
    b.add_argument("--capture", required=True)
    b.add_argument("--camera", choices=("left", "right", "overhead"), required=True)
    b.add_argument("--models", default="sim/generated")
    b.add_argument("--board", default="calibration/board/board.json")
    b.add_argument("--split", choices=("train", "validation"), required=True)
    b.add_argument("--target-id", required=True, help="One fixed physical placement of the board")
    b.add_argument("--allow-simulation", action="store_true")
    b.add_argument("--output", required=True)
    b = s.add_parser(
        "calibration-set", help="Check sample identity/splits and assemble a calibration dataset"
    )
    b.add_argument("--samples", nargs="+", required=True)
    b.add_argument("--output", required=True)
    b = s.add_parser(
        "station-calibrate",
        help="Two hand-eye fits, relative base pose and overhead fit from one fixed board",
    )
    for name in ("left", "right", "overhead"):
        b.add_argument("--" + name, required=True)
    b.add_argument("--rig", default="config/rig.nominal.json")
    b.add_argument("--output", required=True)
    b.add_argument("--max-translation-m", type=float, default=0.01)
    b.add_argument("--max-rotation-deg", type=float, default=1)
    b = s.add_parser("overlay", help="Static real/URDF overlay and registered depth residuals")
    b.add_argument("--capture", required=True)
    b.add_argument("--camera", choices=("left", "right", "overhead"), required=True)
    b.add_argument("--models", default="sim/generated")
    b.add_argument("--allow-simulation", action="store_true")
    b.add_argument("--output", required=True)
    for cmd in ("icp", "validate-icp"):
        b = s.add_parser(cmd)
        b.add_argument("--capture", required=True)
        b.add_argument("--camera", required=True)
        b.add_argument("--mask", required=True, help="Binary mask in RAW DEPTH image coordinates")
        b.add_argument("--models", default="sim/generated")
        b.add_argument("--arm", choices=["left", "right", "both"], required=True)
        b.add_argument(
            "--transform",
            required=True,
            help="JSON with T_world_camera (e.g. previous candidate)",
        )
        b.add_argument("--output", required=True)
        if cmd == "icp":
            b.add_argument(
                "--manual",
                action="store_true",
                help="Sergio's Viser alignment UI before ICP",
            )
    b = s.add_parser("apply-candidate")
    b.add_argument("--rig", default="config/rig.nominal.json")
    b.add_argument("--candidate", required=True)
    b.add_argument("--camera", required=True)
    b.add_argument("--capture")
    b.add_argument("--models", default="sim/generated")
    b.add_argument("--output", required=True)
    b = s.add_parser("command", help="One bounded supervised command; never used by read-only viewer")
    b.add_argument("--url", default="http://127.0.0.1:8765")
    b.add_argument("--token-file", required=True)
    b.add_argument("--arm", choices=["left", "right"], required=True)
    b.add_argument("--radians", type=float, nargs=6, required=True)
    b.add_argument("--gripper-open", type=float, required=True)
    return p


def parse_pairs(values, allowed):
    result = {}
    for value in values:
        name, identifier = value.split("=", 1)
        if name not in allowed or name in result or not identifier:
            raise ValueError(f"Invalid/duplicate device mapping: {value}")
        result[name] = identifier
    if len(set(result.values())) != len(result):
        raise ValueError("Duplicate device identifier")
    return result


def main(argv=None):
    a = parser().parse_args(argv)
    from .models import YamModel, build
    from .transport import Client, ReplayProvider, save_capture

    if a.task == "init-rig":
        from .models import initial_config

        if Path(a.output).exists():
            raise ValueError("Refusing to overwrite existing rig configuration")
        write(a.output, initial_config(a.vendor, read(a.metadata) if a.metadata else None))
    elif a.task == "build":
        result = build(a.vendor, a.rig, a.output, a.require_calibrated)
        print(
            json.dumps(
                {
                    "output": str(Path(a.output).resolve()),
                    "calibrated": result["calibrated"],
                    "blockers": result["calibration_blockers"],
                },
                indent=2,
            )
        )
    elif a.task == "verify-models":
        from .validation import check_models

        print(json.dumps(check_models(a.models), indent=2))
    elif a.task == "render":
        from .validation import render_model

        print(json.dumps(render_model(a.models, a.output), indent=2))
    elif a.task == "check-numerics":
        from .validation import check_numerics

        result = check_numerics(a.models)
        write(a.output, result)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise ValueError("Offline timestep convergence criterion failed; inspect the saved report")
    elif a.task in ("audit-models", "export-bundle"):
        from .bundle import audit_bundle, export_bundle

        result = audit_bundle(a.models) if a.task == "audit-models" else export_bundle(a.models, a.output)
        if a.task == "audit-models":
            write(a.output, result)
        print(json.dumps(result, indent=2))
    elif a.task == "board-sample":
        from .workflow import board_sample

        write(
            a.output,
            board_sample(
                a.capture,
                a.camera,
                a.models,
                read(a.board),
                a.split,
                a.target_id,
                allow_simulation=a.allow_simulation,
            ),
        )
    elif a.task == "calibration-set":
        from .workflow import collect_samples

        write(a.output, collect_samples([read(path) for path in a.samples]))
    elif a.task == "station-calibrate":
        from .workflow import calibrate_station

        result = calibrate_station(
            read(a.left),
            read(a.right),
            read(a.overhead),
            read(a.rig),
            a.max_translation_m,
            a.max_rotation_deg,
        )
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=False)
        write(output / "rig.candidate.json", result.pop("rig"))
        write(output / "report.json", result)
        print(
            json.dumps(
                {
                    "output": str(output.resolve()),
                    "status": result["status"],
                    "validation_passed": result["validation_passed"],
                    "source_mode": result["source_mode"],
                }
            )
        )
    elif a.task == "overlay":
        from .overlay import overlay_capture

        print(
            json.dumps(
                overlay_capture(a.capture, a.camera, a.models, a.output, allow_simulation=a.allow_simulation),
                indent=2,
            )
        )
    elif a.task == "serve":
        from .acquisition import LiveProvider, SimulationProvider
        from .transport import CommandGate, ObservationServer

        cameras = parse_pairs(a.camera, ("left", "right", "overhead"))
        arms = parse_pairs(a.arm, ("left", "right"))
        if not a.live and (cameras or arms or a.ack_motor_initialization):
            raise ValueError("Hardware device arguments require --live")
        if a.joint_source and (not a.live or arms or a.enable_motion):
            raise ValueError("--joint-source is read-only and mutually exclusive with owned arms/motion")
        if a.render_cameras and not a.sim:
            raise ValueError("--render-cameras requires --sim")
        if a.enable_motion and (a.replay or (a.live and not arms)):
            raise ValueError("Motion requires live arms or simulation")
        token = Path(a.token_file).read_text().strip() if a.token_file else None
        if a.enable_motion and (not token or len(token) < 24):
            raise ValueError("--enable-motion requires a random >=24 character --token-file")
        if a.live and not (cameras or arms):
            raise ValueError("Select explicit camera serials and/or CAN interfaces")
        model = YamModel(Path(a.models) / "yam_bimanual.urdf") if a.enable_motion else None
        provider = (
            ReplayProvider(a.replay)
            if a.replay
            else SimulationProvider(Path(a.models) / "yam_bimanual.xml", a.render_cameras)
            if a.sim
            else LiveProvider(cameras, arms, a.ack_motor_initialization, a.joint_source)
        )
        server = None
        try:
            gate = CommandGate(model.limits, enabled=True) if a.enable_motion else None
            server = ObservationServer(provider, a.port, gate, token).start()
            print(
                f"Observations: http://127.0.0.1:{server.port}/v1/status; motion={bool(gate)}",
                flush=True,
            )
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            if server:
                server.close()
            else:
                provider.close()
    elif a.task == "view":
        from .viewer import run_viewer

        run_viewer(a.url, a.models, a.port, a.nominal_preview)
    elif a.task == "capture":
        meta = save_capture(Client(a.url), a.output, a.require_joints, a.max_skew_ms, a.stationary_seconds)
        print(f"Saved {meta['mode']} capture: {Path(a.output).resolve()}")
    elif a.task == "stream-check":
        client = Client(a.url)
        start = time.monotonic()
        rows = []
        failures = []
        previous = None
        while time.monotonic() - start < a.seconds:
            before = time.monotonic()
            try:
                m, arrays = client.observation()
                if m["observation_id"] != previous:
                    rows.append(
                        {
                            "client_elapsed_s": time.monotonic() - start,
                            "roundtrip_ms": 1000 * (time.monotonic() - before),
                            "mode": m["mode"],
                            "arms": list(m["arms"]),
                            "cameras": list(m["cameras"]),
                            "camera_frame_numbers": {
                                k: v["color"].get("frame_number") for k, v in m["cameras"].items()
                            },
                            "observation_id": m["observation_id"],
                            "arrays": {k: list(v.shape) for k, v in arrays.items()},
                        }
                    )
                    previous = m["observation_id"]
            except Exception as e:  # noqa: BLE001 — the report records every failed receive attempt.
                failures.append(str(e))
            time.sleep(0.03)
        duration = time.monotonic() - start
        result = {
            "duration_s": duration,
            "unique_packets": len(rows),
            "receive_hz": len(rows) / duration,
            "failures": failures,
            "observations": rows,
            "one_way_latency": "unknown; client/server clocks not calibrated",
            "live_joint_and_camera_ingest_verified": bool(rows)
            and all(
                r["mode"] == "live"
                and set(r["arms"]) == {"left", "right"}
                and set(r["cameras"]) == {"left", "right", "overhead"}
                for r in rows
            )
            and not failures,
        }
        write(a.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))
    elif a.task == "board":
        import cv2

        from .calibration import board_spec, make_board

        spec = board_spec()
        p = Path(a.output)
        p.mkdir(parents=True, exist_ok=True)
        image = make_board(spec).generateImage((2380, 1700), marginSize=0, borderBits=1)
        cv2.imwrite(str(p / "charuco.png"), image)
        write(p / "board.json", spec)
        print(f"Print {p / 'charuco.png'} at EXACTLY 238 x 170 mm. Measure square length before calibration.")
    elif a.task == "rectify":
        from PIL import Image

        from .geometry import rectify_rgb

        m, arrays = ReplayProvider(a.capture).snapshot()
        rgb, i = rectify_rgb(arrays[a.camera + "_rgb"], m["cameras"][a.camera]["color"]["intrinsics"])
        p = Path(a.output)
        p.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(p / "rgb_rectified.png")
        write(p / "intrinsics.json", i)
    elif a.task == "detect-board":
        from PIL import Image

        from .calibration import detect_board, file_hash

        result = detect_board(
            np.asarray(Image.open(a.rgb).convert("RGB")),
            read(a.intrinsics),
            read(a.board),
        )
        result["image_sha256"] = file_hash(a.rgb)
        write(a.output, result)
    elif a.task in ("hand-eye", "fixed-camera"):
        from .calibration import file_hash, fixed_camera, hand_eye

        solve = hand_eye if a.task == "hand-eye" else fixed_camera
        result = solve(read(a.samples), a.max_translation_m, a.max_rotation_deg)
        result["samples_sha256"] = file_hash(a.samples)
        write(a.output, result)
        print(json.dumps(result, indent=2))
    elif a.task in ("icp", "validate-icp"):
        from PIL import Image

        from .calibration import camera_cloud, file_hash, icp_to_robot, validate_cloud
        from .geometry import transform

        meta, arrays = ReplayProvider(a.capture).snapshot()
        if not meta["arms"]:
            raise ValueError("Capture has no measured joint states; cannot calibrate a URDF against it")
        selected = ("left", "right") if a.arm == "both" else (a.arm,)
        if any(side not in meta["arms"] for side in selected):
            raise ValueError("Missing target arm feedback")
        mask = np.asarray(Image.open(a.mask).convert("L")) > 0
        points = camera_cloud(meta, arrays, a.camera, mask)
        model = YamModel(Path(a.models) / "yam_bimanual.urdf")
        robot = np.concatenate([model.mesh_points(meta["arms"], side, count=15000) for side in selected])
        initial = transform(read(a.transform)["T_world_camera"])
        if a.task == "icp":
            if a.manual:
                from lerobot_3d.point_clouds.alignment_viewer import AlignmentViewer

                v = AlignmentViewer(host="127.0.0.1")
                v.show("/target", robot, None)
                try:
                    initial = v.align(points, initial, title="YAM masked-depth alignment")
                finally:
                    v.server.stop()
            result = icp_to_robot(points, robot, initial)
        else:
            result = validate_cloud(points, robot, initial)
        result.update(
            camera=a.camera,
            arms=list(selected),
            capture=str(Path(a.capture).resolve()),
            mask_sha256=file_hash(a.mask),
            urdf_sha256=file_hash(Path(a.models) / "yam_bimanual.urdf"),
        )
        write(a.output, result)
        print(json.dumps(result, indent=2))
    elif a.task == "apply-candidate":
        from .calibration import candidate_override

        arms = ReplayProvider(a.capture).snapshot()[0]["arms"] if a.capture else None
        model = YamModel(Path(a.models) / "yam_bimanual.urdf") if arms else None
        write(
            a.output,
            candidate_override(read(a.rig), a.camera, read(a.candidate), arms, model),
        )
    elif a.task == "command":
        client = Client(a.url, Path(a.token_file).read_text().strip())
        meta = client.state()
        print(client.command(a.arm, a.radians, a.gripper_open, meta["observation_id"]))


if __name__ == "__main__":
    main()
