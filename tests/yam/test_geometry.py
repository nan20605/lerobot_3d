import numpy as np
import pytest

from lerobot_3d.yam.runtime.geometry import (
    depth_points,
    pose,
    rectify_rgb,
    require_opencv_intrinsics,
    rs_depth_to_color,
    transform,
)


def test_transform_direction_and_column_major():
    t = pose([0.01, 0.02, 0.03], rpy=[0.1, 0.2, 0.3])
    converted = rs_depth_to_color(
        {"rotation": t[:3, :3].ravel(order="F").tolist(), "translation_m": t[:3, 3]}
    )
    np.testing.assert_allclose(t, converted)
    with pytest.raises(ValueError):
        transform(np.diag([-1, 1, 1, 1]))


def test_raw_depth_uses_its_own_intrinsics_and_actual_d405_scale():
    i = {
        "width": 3,
        "height": 2,
        "fx": 100,
        "fy": 100,
        "ppx": 1,
        "ppy": 0,
        "model": "brown_conrady",
        "coeffs": [0] * 5,
    }
    d = np.array([[0, 10000, 10000], [10000, 0, 0]], np.uint16)
    points = depth_points(d, i, 0.0001)
    np.testing.assert_allclose(points, [[0, 0, 1], [0.01, 0, 1], [-0.01, 0.01, 1]])
    with pytest.raises(ValueError):
        depth_points(d, i, 0.0001, mask=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        depth_points(d, i, 0)


def test_inverse_brown_is_not_opencv_distortion(rig):
    i = rig["cameras"]["left_camera"]["intrinsics"]
    with pytest.raises(ValueError):
        require_opencv_intrinsics(i)


def test_inverse_brown_rectification_matches_realsense_projection(rig):
    # Encode source pixel coordinates as float channels; inspect remapped values.
    i = rig["cameras"]["left_camera"]["intrinsics"]
    y, x = np.mgrid[: i["height"], : i["width"]]
    source = np.stack([x, y, x * 0], axis=-1).astype(np.float32)
    result, intr = rectify_rgb(source, i)
    for u, v in [(25, 30), (500, 420), (320, 240)]:
        # Scalar transcription of librealsense rs2_project_point_to_pixel.
        a = (u - i["ppx"]) / i["fx"]
        b = (v - i["ppy"]) / i["fy"]
        r = a * a + b * b
        k1, k2, p1, p2, k3 = i["coeffs"]
        f = 1 + k1 * r + k2 * r * r + k3 * r * r * r
        a *= f
        b *= f
        expected = [
            (a + 2 * p1 * a * b + p2 * (r + 2 * a * a)) * i["fx"] + i["ppx"],
            (b + 2 * p2 * a * b + p1 * (r + 2 * b * b)) * i["fy"] + i["ppy"],
        ]
        np.testing.assert_allclose(result[v, u, :2], expected, atol=0.017)
    assert intr["model"] == "none"
