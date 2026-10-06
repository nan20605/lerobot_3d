"""Metric transforms: T_A_B maps B points into A; quaternions are wxyz."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Transform must be finite 4x4")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=2e-6):
        raise ValueError("Rotation is not orthonormal")
    if not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=2e-6):
        raise ValueError("Rotation must be right handed")
    return t


def pose(xyz=(0, 0, 0), *, wxyz=None, rpy=None):
    t = np.eye(4)
    t[:3, 3] = xyz
    if wxyz is not None:
        q = np.asarray(wxyz, dtype=float)
        if q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
            raise ValueError("Expected unit wxyz quaternion")
        t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    elif rpy is not None:
        t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return transform(t)


def quaternion(t):
    return Rotation.from_matrix(transform(t)[:3, :3]).as_quat()[[3, 0, 1, 2]]


def apply(t, points):
    t = transform(t)
    # Explicit contraction avoids BLAS floating-point status warnings after an
    # OpenGL render on macOS, for this small three-column operation.
    return np.einsum("...j,ij->...i", np.asarray(points), t[:3, :3]) + t[:3, 3]


def rs_depth_to_color(metadata):
    """librealsense rs2_extrinsics.rotation is column-major, translation metres."""
    t = np.eye(4)
    t[:3, :3] = np.array(metadata["rotation"]).reshape(3, 3, order="F")
    t[:3, 3] = metadata["translation_m"]
    return transform(t)


def camera_matrix(intrinsics):
    i = intrinsics
    k = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]], float)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    return k


def depth_points(depth, intrinsics, scale, mask=None, stride=1, max_depth=2.0):
    """Deproject RAW depth using DEPTH intrinsics (never the RGB intrinsics)."""
    import cv2

    d = np.asarray(depth)
    if d.shape != (intrinsics["height"], intrinsics["width"]) or stride < 1:
        raise ValueError("Depth shape/stride mismatch")
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Invalid metres per depth unit")
    yy, xx = np.mgrid[: d.shape[0] : stride, : d.shape[1] : stride]
    z = d[::stride, ::stride] * scale
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    if mask is not None:
        if np.shape(mask) != d.shape:
            raise ValueError("Mask must be in the raw DEPTH image coordinates")
        valid &= np.asarray(mask, bool)[::stride, ::stride]
    pixels = np.column_stack((xx[valid], yy[valid])).astype(float)
    k = camera_matrix(intrinsics)
    coeff = np.asarray(intrinsics.get("coeffs", [0] * 5), float)
    model = intrinsics.get("model", "none").split(".")[-1]
    if not np.any(coeff):
        rays = (pixels - k[:2, 2]) / np.diag(k)[:2]
    elif model == "brown_conrady":
        rays = cv2.undistortPoints(pixels[:, None, :], k, coeff)[:, 0]
    else:
        raise ValueError(f"Depth distortion {model} requires librealsense deprojection")
    return np.column_stack((rays * z[valid, None], z[valid]))


def require_opencv_intrinsics(intrinsics):
    """Never interpret inverse Brown coefficients as forward OpenCV distortion."""
    model = intrinsics.get("model", "none").split(".")[-1]
    coeffs = np.asarray(intrinsics.get("coeffs", [0] * 5), float)
    if model not in ("none", "brown_conrady") and np.any(coeffs):
        raise ValueError("Rectify RGB with yam rectify first; this is not OpenCV forward distortion")
    return camera_matrix(intrinsics), coeffs


def rectify_rgb(rgb, intrinsics):
    """Remap ideal pixels to native pixels using librealsense projection semantics.

    rs2_project_point_to_pixel handles inverse_brown_conrady using radial scaling
    BEFORE tangential terms, unlike OpenCV's Brown polynomial. Match that code,
    not an assumption based on the enum's name. K and resolution stay unchanged.
    See librealsense src/rs.cpp, rs2_project_point_to_pixel.
    """
    import cv2

    k = camera_matrix(intrinsics)
    coeffs = np.asarray(intrinsics.get("coeffs", [0] * 5), float)
    model = intrinsics.get("model", "none").split(".")[-1]
    if np.shape(rgb)[:2] != (intrinsics["height"], intrinsics["width"]):
        raise ValueError("RGB shape differs from intrinsics")
    if not np.any(coeffs):
        result = np.array(rgb, copy=True)
    elif model == "brown_conrady":
        result = cv2.undistort(rgb, k, coeffs)
    elif model == "inverse_brown_conrady":
        yy, xx = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
        u, v = (xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1]
        r2 = u * u + v * v
        k1, k2, p1, p2, k3 = coeffs
        f = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        u = u * f
        v = v * f
        xd = u + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
        yd = v + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)
        result = cv2.remap(
            rgb,
            (xd * k[0, 0] + k[0, 2]).astype("float32"),
            (yd * k[1, 1] + k[1, 2]).astype("float32"),
            cv2.INTER_LINEAR,
        )
    else:
        raise ValueError(f"Unsupported RGB distortion: {model}")
    return result, {
        **intrinsics,
        "model": "none",
        "coeffs": [0.0] * 5,
        "rectified": True,
    }


def project_color(points, intrinsics):
    """Project color-optical XYZ to native RGB pixels, matching librealsense."""
    points = np.asarray(points, float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    z = np.where(valid, points[:, 2], 1)
    x, y = points[:, 0] / z, points[:, 1] / z
    model = intrinsics.get("model", "none").split(".")[-1]
    coefficients = np.asarray(intrinsics.get("coeffs", [0] * 5), float)
    if np.any(coefficients):
        if model not in ("brown_conrady", "inverse_brown_conrady"):
            raise ValueError(f"Unsupported RGB projection {model}")
        k1, k2, p1, p2, k3 = coefficients
        r2 = x * x + y * y
        f = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        if model == "inverse_brown_conrady":
            x = x * f
            y = y * f
            f = 1
        xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        x, y = xd, yd
    pixels = np.column_stack(
        (
            x * intrinsics["fx"] + intrinsics["ppx"],
            y * intrinsics["fy"] + intrinsics["ppy"],
        )
    )
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < intrinsics["width"])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < intrinsics["height"])
    )
    return pixels, valid
