"""Turn paired captures into traceable calibration samples and a candidate station."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import detect_board, file_hash, fixed_camera, hand_eye, pose_errors
from .geometry import rectify_rgb, transform
from .models import YamModel
from .transport import decode


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_paired_capture(directory, camera, *, allow_simulation=False, max_skew_ms=50):
    path = Path(directory) / "observation.npz"
    if not path.is_file():
        raise ValueError("Use yam capture --require-joints to save a paired observation.npz first")
    meta, arrays = decode(path.read_bytes())
    mode = meta["mode"]
    if mode != "live" and not (allow_simulation and mode == "simulation"):
        raise ValueError("Calibration needs live captures; synthetic inputs require --allow-simulation")
    if camera not in meta["cameras"] or set(meta["arms"]) != {"left", "right"}:
        raise ValueError("Capture needs the selected camera and measured feedback from both arms")
    if mode == "live" and meta.get("stationarity", {}).get("sample_count", 0) < 3:
        raise ValueError("Capture lacks a stationarity window; recapture with yam capture --require-joints")
    times = [meta["cameras"][camera]["host_monotonic_s"]]
    for side, state in meta["arms"].items():
        velocity = np.asarray(state["velocity_rad_s"], float)
        if velocity.shape != (6,) or not np.isfinite(velocity).all() or np.max(abs(velocity)) > 0.01:
            raise ValueError(f"{side} must be stationary during calibration capture")
        times.append(state["host_monotonic_s"])
    if not np.isfinite(times).all() or (max(times) - min(times)) * 1000 > max_skew_ms:
        raise ValueError("Joint/camera host-arrival skew exceeds limit")
    return meta, arrays


def board_sample(
    capture, camera, model_dir, board, split, target_id, *, allow_simulation=False, max_reprojection_px=1.5
):
    """Rectify, detect the board and attach base-relative measured-state FK.

    target_id identifies one physically fixed placement of the board, not merely
    a board design. Moving that board requires a new target_id/session.
    """
    if camera not in ("left", "right", "overhead") or split not in ("train", "validation"):
        raise ValueError("Choose a known camera and explicit train/validation split")
    if not target_id.strip():
        raise ValueError("A fixed-board placement identifier is required")
    meta, arrays = read_paired_capture(capture, camera, allow_simulation=allow_simulation)
    if meta["mode"] == "live" and (
        board.get("measurement_status") != "measured" or not board.get("measurement_source")
    ):
        raise ValueError("Measure the printed board; set measurement_status=measured and measurement_source")
    rgb, intrinsics = rectify_rgb(arrays[camera + "_rgb"], meta["cameras"][camera]["color"]["intrinsics"])
    detected = detect_board(rgb, intrinsics, board)
    if detected["reprojection_max_px"] > max_reprojection_px:
        raise ValueError("Board reprojection exceeds the sample quality threshold")
    model_path = Path(model_dir) / "yam_bimanual.urdf"
    poses = YamModel(model_path).fk(meta["arms"])
    result = {
        **detected,
        "schema_version": 1,
        "split": split,
        "camera": camera,
        "target_id": target_id,
        "source_mode": meta["mode"],
        "camera_serial": meta["cameras"][camera]["serial"],
        "camera_intrinsics_sha256": fingerprint(meta["cameras"][camera]["color"]["intrinsics"]),
        "board_sha256": fingerprint(board),
        "urdf_sha256": file_hash(model_path),
        "capture_sha256": file_hash(Path(capture) / "observation.npz"),
        "image_sha256": hashlib.sha256(arrays[camera + "_rgb"].tobytes()).hexdigest(),
        "host_arrival_note": "Static capture; no hardware exposure synchronization or interpolation",
    }
    info = meta["cameras"][camera]
    result["camera_profile"] = {
        "serial": info["serial"],
        "model": info.get("name", "unknown"),
        "intrinsics": info["color"]["intrinsics"],
        "depth_intrinsics": info["depth"]["intrinsics"],
        "depth_scale_m_per_unit": info["depth"]["depth_scale_m_per_unit"],
        "depth_to_color": info["depth_to_color"],
    }
    if camera != "overhead":
        result["T_base_gripper"] = (
            np.linalg.inv(poses[camera + "_base"]) @ poses[camera + "_gripper"]
        ).tolist()
    return result


def collect_samples(samples):
    """Reject duplicate images, mixed boards/models/devices and split leakage."""
    if not samples:
        raise ValueError("No calibration samples")
    if any(s["source_mode"] not in ("live", "simulation") for s in samples):
        raise ValueError("Calibration samples must identify live or simulation input")
    for key in (
        "camera",
        "camera_serial",
        "target_id",
        "board_sha256",
        "urdf_sha256",
        "camera_intrinsics_sha256",
        "source_mode",
    ):
        if len({s[key] for s in samples}) != 1:
            raise ValueError(f"Mixed calibration sample {key}")
    if len({fingerprint(s["camera_profile"]) for s in samples}) != 1:
        raise ValueError("Mixed calibration camera_profile; keep the same stream configuration")
    images = [s["image_sha256"] for s in samples]
    if len(set(images)) != len(images):
        raise ValueError(
            "Duplicate image in calibration set; fitting/validation captures must be independent"
        )
    if any(s["split"] not in ("train", "validation") for s in samples):
        raise ValueError("Invalid calibration split")
    return samples


def calibrate_station(left, right, overhead, rig, max_translation_m=0.01, max_rotation_deg=1):
    """Solve two wrist mounts, the right base and overhead against one fixed board.

    The left base defines world. Hand-eye estimates anchor the stationary board
    in each base, which establishes the relative base pose without a CAD offset.
    This is a candidate fit; independent absolute validation remains necessary.
    """
    groups = {
        name: collect_samples(samples)
        for name, samples in (("left", left), ("right", right), ("overhead", overhead))
    }
    for camera, samples in groups.items():
        if samples[0]["camera"] != camera:
            raise ValueError(f"Wrong sample set for {camera}")
    for key in ("target_id", "board_sha256", "urdf_sha256", "source_mode"):
        if len({samples[0][key] for samples in groups.values()}) != 1:
            raise ValueError(f"All cameras must share {key}")
    cfg = copy.deepcopy(rig)
    left_base = cfg["transforms"]["left_base"]
    if left_base["status"] not in ("frame_definition", "validated"):
        raise ValueError("Establish the world/left-base frame first")
    fits = {name: hand_eye(groups[name], max_translation_m, max_rotation_deg) for name in ("left", "right")}
    world_left = transform(left_base["T_parent_child"])
    world_target = world_left @ transform(fits["left"]["T_base_target"])
    world_right = world_target @ np.linalg.inv(transform(fits["right"]["T_base_target"]))
    anchored_overhead = [{**sample, "T_world_target": world_target.tolist()} for sample in overhead]
    fits["overhead"] = fixed_camera(anchored_overhead, max_translation_m, max_rotation_deg)
    transforms = {
        "left_camera": fits["left"]["T_gripper_camera"],
        "right_camera": fits["right"]["T_gripper_camera"],
        "right_base": world_right.tolist(),
        "top_camera": fits["overhead"]["T_world_camera"],
    }
    source_mode = left[0]["source_mode"]
    for frame, value in transforms.items():
        cfg["transforms"][frame].update(
            T_parent_child=value,
            status="candidate",
            source={
                "method": "fixed_board_station_hand_eye",
                "target_id": left[0]["target_id"],
                "sample_set_sha256": fingerprint(groups),
                "source_mode": source_mode,
            },
        )
    cfg["calibration_source_mode"] = source_mode
    for camera, samples in groups.items():
        frame = "top_camera" if camera == "overhead" else camera + "_camera"
        cfg["cameras"][frame] = {
            **samples[0]["camera_profile"],
            "intrinsics_status": "factory" if source_mode == "live" else "synthetic",
            "source": "paired_capture:" + samples[0]["capture_sha256"],
        }
    # Cross-camera held-out board consistency. Validation never updates any fit.
    validation = {}
    for camera, samples in groups.items():
        estimates = []
        for sample in samples:
            if sample["split"] != "validation":
                continue
            if camera == "overhead":
                world_camera = transform(fits[camera]["T_world_camera"])
            else:
                world_base = world_left if camera == "left" else world_right
                world_camera = (
                    world_base
                    @ transform(sample["T_base_gripper"])
                    @ transform(fits[camera]["T_gripper_camera"])
                )
            estimates.append(world_camera @ transform(sample["T_camera_target"]))
        validation[camera] = pose_errors(world_target, estimates)
    passed = all(
        max(m["translation_m"]) <= max_translation_m and max(m["rotation_deg"]) <= max_rotation_deg
        for m in validation.values()
    )
    return {
        "status": "candidate",
        "source_mode": source_mode,
        "target_id": left[0]["target_id"],
        "rig": cfg,
        "fits": fits,
        "T_world_target": world_target.tolist(),
        "held_out_board_consistency": validation,
        "validation_passed": bool(passed),
        "sample_set_sha256": fingerprint(groups),
        "thresholds": {"translation_m": max_translation_m, "rotation_deg": max_rotation_deg},
        "limitations": [
            "Board placement must remain fixed for the entire session.",
            "Consistency is not independent absolute accuracy or an uncertainty estimate.",
            "Confirm joint zeros, scale, installed hardware and held-out real overlays before acceptance.",
        ],
    }
