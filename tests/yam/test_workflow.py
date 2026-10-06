import copy
import json

import cv2
import numpy as np
import pytest

from lerobot_3d.yam.runtime.calibration import board_spec, make_board
from lerobot_3d.yam.runtime.geometry import pose
from lerobot_3d.yam.runtime.models import YamModel
from lerobot_3d.yam.runtime.transport import encode
from lerobot_3d.yam.runtime.workflow import board_sample, calibrate_station, collect_samples, fingerprint


@pytest.fixture
def station_samples(rig):
    rng = np.random.default_rng(87)
    target = pose([0.4, -0.2, 0.3], rpy=[0.1, 0.2, 0.3])
    bases = {"left": np.eye(4), "right": pose([0.04, -0.64, 0.02], rpy=[0.02, 0.03, 0.08])}
    wrists = {
        "left": pose([0.02, -0.07, 0.09], rpy=[0.2, 0.1, 0.3]),
        "right": pose([0.03, -0.06, 0.1], rpy=[0.1, -0.2, 0.15]),
    }
    overhead = pose([0.1, -0.3, 0.9], rpy=[3.0, 0.1, 0.2])
    result = {}
    for camera in ("left", "right", "overhead"):
        profile = copy.deepcopy(rig["cameras"]["left_camera"])
        profile["serial"] = "SYNTHETIC_" + camera
        result[camera] = []
        for i in range(16):
            gripper = pose(rng.uniform(-0.2, 0.2, 3), rpy=rng.uniform(-0.9, 0.9, 3))
            world_camera = overhead if camera == "overhead" else bases[camera] @ gripper @ wrists[camera]
            sample = {
                "camera": camera,
                "camera_serial": profile["serial"],
                "camera_profile": profile,
                "camera_intrinsics_sha256": fingerprint(profile["intrinsics"]),
                "image_sha256": f"synthetic-{camera}-{i}",
                "capture_sha256": f"capture-{camera}-{i}",
                "board_sha256": "synthetic-board",
                "urdf_sha256": "synthetic-model",
                "source_mode": "simulation",
                "target_id": "fixed-placement-001",
                "split": "train" if i < 12 else "validation",
                "T_camera_target": (np.linalg.inv(world_camera) @ target).tolist(),
            }
            if camera != "overhead":
                sample["T_base_gripper"] = gripper.tolist()
            result[camera].append(sample)
    return result, bases, wrists, overhead


def test_station_solves_base_and_three_cameras_without_cad_alignment(rig, station_samples):
    samples, bases, wrists, overhead = station_samples
    result = calibrate_station(samples["left"], samples["right"], samples["overhead"], rig)
    assert result["validation_passed"] and result["status"] == "candidate"
    fitted = result["rig"]["transforms"]
    np.testing.assert_allclose(fitted["right_base"]["T_parent_child"], bases["right"], atol=1e-10)
    np.testing.assert_allclose(fitted["top_camera"]["T_parent_child"], overhead, atol=1e-10)
    for side in ("left", "right"):
        np.testing.assert_allclose(fitted[side + "_camera"]["T_parent_child"], wrists[side], atol=1e-10)
    samples["overhead"][-1]["T_camera_target"][0][3] += 0.08
    failed = calibrate_station(samples["left"], samples["right"], samples["overhead"], rig)
    assert not failed["validation_passed"]
    for frame in fitted:
        np.testing.assert_array_equal(
            failed["rig"]["transforms"][frame]["T_parent_child"], fitted[frame]["T_parent_child"]
        )


def test_calibration_rejects_duplicate_images_and_moved_board(rig, station_samples):
    samples, *_ = station_samples
    duplicated = samples["left"] + [copy.deepcopy(samples["left"][0])]
    duplicated[-1]["split"] = "validation"
    with pytest.raises(ValueError, match="Duplicate image"):
        collect_samples(duplicated)
    for sample in samples["right"]:
        sample["target_id"] = "board-moved"
    with pytest.raises(ValueError, match="target_id"):
        calibrate_station(samples["left"], samples["right"], samples["overhead"], rig)


def test_board_sample_from_capture_runs_detection_and_fk(tmp_path, bundle):
    spec = board_spec()
    board = make_board(spec).generateImage((700, 500), marginSize=0, borderBits=1)
    canvas = np.full((800, 1000), 255, np.uint8)
    canvas[150:650, 150:850] = board
    intrinsics = {
        "width": 1000,
        "height": 800,
        "fx": 1000,
        "fy": 1000,
        "ppx": 500,
        "ppy": 400,
        "model": "none",
        "coeffs": [0] * 5,
    }
    arms = {
        side: {
            "position_rad": [0, 1, 1, 0, 0, 0],
            "gripper_open": 0.5,
            "velocity_rad_s": [0] * 6,
            "host_monotonic_s": 1.0,
        }
        for side in ("left", "right")
    }
    meta = {
        "schema_version": 1,
        "mode": "simulation",
        "arms": arms,
        "cameras": {
            "left": {
                "serial": "SYNTHETIC_left",
                "color": {"intrinsics": intrinsics},
                "host_monotonic_s": 1.0,
                "depth": {"intrinsics": intrinsics, "depth_scale_m_per_unit": 0.0001},
                "depth_to_color": {
                    "rotation": np.eye(3).ravel(order="F").tolist(),
                    "translation_m": [0, 0, 0],
                },
            }
        },
    }
    arrays = {
        "left_rgb": cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB),
        "left_depth_raw": np.full((800, 1000), 3400, np.uint16),
    }
    (tmp_path / "observation.npz").write_bytes(encode(meta, arrays))
    with pytest.raises(ValueError, match="allow-simulation"):
        board_sample(tmp_path, "left", bundle, spec, "train", "fixed-board")
    sample = board_sample(tmp_path, "left", bundle, spec, "train", "fixed-board", allow_simulation=True)
    assert np.array(sample["T_camera_target"])[2, 3] == pytest.approx(0.34, abs=0.002)
    poses = YamModel(bundle / "yam_bimanual.urdf").fk(arms)
    np.testing.assert_allclose(
        sample["T_base_gripper"], np.linalg.inv(poses["left_base"]) @ poses["left_gripper"]
    )
    json.dumps(sample, allow_nan=False)
    meta.update(mode="live", stationarity={"sample_count": 3})
    (tmp_path / "observation.npz").write_bytes(encode(meta, arrays))
    with pytest.raises(ValueError, match="Measure the printed board"):
        board_sample(tmp_path, "left", bundle, spec, "train", "fixed-board")


def test_calibration_rejects_changed_depth_profile(station_samples):
    samples, *_ = station_samples
    changed = copy.deepcopy(samples["left"])
    changed[-1]["camera_profile"] = copy.deepcopy(changed[-1]["camera_profile"])
    changed[-1]["camera_profile"]["depth_scale_m_per_unit"] = 0.001
    with pytest.raises(ValueError, match="camera_profile"):
        collect_samples(changed)
