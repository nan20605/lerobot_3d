import cv2
import numpy as np
import pytest

from lerobot_3d.yam.runtime.calibration import (
    board_spec,
    detect_board,
    fixed_camera,
    hand_eye,
    make_board,
)
from lerobot_3d.yam.runtime.geometry import pose


def test_handeye_recovers_known_transform_and_heldout_catches_bad_pose():
    rng = np.random.default_rng(42)
    x = pose([0.04, -0.07, 0.09], rpy=[0.3, -0.2, 0.1])
    target = pose([0.5, 0.1, 0.3], rpy=[0.1, 0.3, 0.4])
    samples = []
    for i in range(16):
        g = pose(rng.uniform(-0.2, 0.2, 3), rpy=rng.uniform(-0.8, 0.8, 3))
        samples.append(
            {
                "T_base_gripper": g.tolist(),
                "T_camera_target": (np.linalg.inv(g @ x) @ target).tolist(),
                "split": "train" if i < 12 else "validation",
            }
        )
    fit = hand_eye(samples)
    np.testing.assert_allclose(fit["T_gripper_camera"], x, atol=1e-9)
    assert fit["validation_passed"] and fit["status"] == "candidate"
    samples[-1]["T_camera_target"][0][3] += 0.1
    bad = hand_eye(samples)
    np.testing.assert_allclose(bad["T_gripper_camera"], x, atol=1e-9)
    assert not bad["validation_passed"]


def test_degenerate_handeye_is_rejected():
    samples = [
        {
            "split": "train" if i < 10 else "validation",
            "T_base_gripper": pose([i * 0.01, 0, 0]).tolist(),
            "T_camera_target": np.eye(4).tolist(),
        }
        for i in range(13)
    ]
    with pytest.raises(ValueError, match="Degenerate"):
        hand_eye(samples)


def test_fixed_camera_anchoring_and_validation():
    camera = pose([0.1, 0.2, 0.9], rpy=[3.0, 0.1, 0.2])
    samples = []
    for i in range(8):
        target = pose([i * 0.02, 0, 0.1], rpy=[0, 0.1 * i, 0])
        samples.append(
            {
                "T_world_target": target.tolist(),
                "T_camera_target": (np.linalg.inv(camera) @ target).tolist(),
                "split": "train" if i < 4 else "validation",
            }
        )
    fit = fixed_camera(samples)
    np.testing.assert_allclose(fit["T_world_camera"], camera, atol=1e-10)
    assert fit["validation_passed"]


def test_charuco_detection_recovers_metric_pose():
    spec = board_spec()
    board = make_board(spec)
    image = board.generateImage((700, 500), marginSize=0, borderBits=1)
    canvas = np.full((800, 1000), 255, np.uint8)
    canvas[150:650, 150:850] = image
    i = {
        "width": 1000,
        "height": 800,
        "fx": 1000,
        "fy": 1000,
        "ppx": 500,
        "ppy": 400,
        "model": "none",
        "coeffs": [0] * 5,
    }
    result = detect_board(cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB), i, spec)
    # 100 pixels per .034 m square => fx * square / pixels = .34 m.
    assert np.array(result["T_camera_target"])[2, 3] == pytest.approx(0.34, abs=0.002)
    assert result["reprojection_rmse_px"] < 0.5
