"""Camera-to-URDF alignment, ChArUco poses and held-out hand-eye checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import (
    apply,
    depth_points,
    project_color,
    require_opencv_intrinsics,
    rs_depth_to_color,
    transform,
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_errors(reference, transforms):
    deltas = [np.linalg.inv(reference) @ transform(t) for t in transforms]
    return {
        "translation_m": [float(np.linalg.norm(t[:3, 3])) for t in deltas],
        "rotation_deg": [float(np.rad2deg(Rotation.from_matrix(t[:3, :3]).magnitude())) for t in deltas],
    }


def mean_pose(values):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(np.array([t[:3, :3] for t in values])).mean().as_matrix()
    result[:3, 3] = np.mean([t[:3, 3] for t in values], axis=0)
    return result


def hand_eye(samples, max_translation_m=0.01, max_rotation_deg=1.0):
    """Solve fixed target / wrist camera: B_G * G_C * C_target = B_target.

    Each sample needs split=train|validation, T_base_gripper, T_camera_target.
    Validation poses never influence the fitted hand-eye or reference target.
    """
    import cv2

    groups = {key: [x for x in samples if x["split"] == key] for key in ("train", "validation")}
    if len(groups["train"]) < 8 or len(groups["validation"]) < 3:
        raise ValueError("Need >=8 training and >=3 held-out poses")
    if any(x.get("split") not in groups for x in samples):
        raise ValueError("Every sample needs an explicit train/validation split")
    train = groups["train"]
    g = [transform(x["T_base_gripper"]) for x in train]
    c = [transform(x["T_camera_target"]) for x in train]
    rotations = np.array([Rotation.from_matrix(g[0][:3, :3].T @ t[:3, :3]).as_rotvec() for t in g[1:]])
    singular = np.linalg.svd(rotations, compute_uv=False)
    if singular[1] < 0.25:
        raise ValueError("Degenerate hand-eye poses: rotate about at least two substantially different axes")
    if np.linalg.norm(np.ptp(np.array([t[:3, 3] for t in g]), axis=0)) < 0.05:
        raise ValueError("Insufficient translation diversity (<5 cm)")
    r, t = cv2.calibrateHandEye(
        [x[:3, :3] for x in g],
        [x[:3, 3] for x in g],
        [x[:3, :3] for x in c],
        [x[:3, 3] for x in c],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    x = np.eye(4)
    x[:3, :3] = r
    x[:3, 3] = t.ravel()
    transform(x)
    target = mean_pose([a @ x @ b for a, b in zip(g, c)])
    metrics = {}
    for split, group in groups.items():
        estimates = [transform(s["T_base_gripper"]) @ x @ transform(s["T_camera_target"]) for s in group]
        metrics[split] = pose_errors(target, estimates)
    v = metrics["validation"]
    passed = max(v["translation_m"]) <= max_translation_m and max(v["rotation_deg"]) <= max_rotation_deg
    return {
        "status": "candidate",
        "method": "opencv_PARK_hand_eye",
        "T_gripper_camera": x.tolist(),
        "T_base_target": target.tolist(),
        "metrics": metrics,
        "rotation_excitation_singular_values": singular.tolist(),
        "validation_passed": passed,
        "thresholds": {
            "max_translation_m": max_translation_m,
            "max_rotation_deg": max_rotation_deg,
        },
        "limitation": "Consistency is not absolute accuracy; inspect image reprojection, joint mapping, board scale, timestamp alignment and mounting rigidity before accepting.",
    }


def board_spec(squares_x=7, squares_y=5, square_m=0.034, marker_m=0.025):
    return {
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_m": square_m,
        "marker_m": marker_m,
        "dictionary": "DICT_5X5_100",
        "units": "metres",
        "print_instruction": "Print at measured physical size, no fit-to-page. Measure squares with calipers.",
    }


def make_board(spec):
    import cv2

    if not 0 < spec["marker_m"] < spec["square_m"]:
        raise ValueError("Marker must be smaller than square")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    return cv2.aruco.CharucoBoard(
        (spec["squares_x"], spec["squares_y"]),
        spec["square_m"],
        spec["marker_m"],
        dictionary,
    )


def detect_board(rgb, intrinsics, spec):
    import cv2

    k, dist = require_opencv_intrinsics(intrinsics)
    board = make_board(spec)
    params = cv2.aruco.CharucoParameters()
    params.cameraMatrix = k
    params.distCoeffs = dist
    detector = cv2.aruco.CharucoDetector(board, params)
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or len(ids) < 8:
        raise ValueError("Need at least eight visible ChArUco intersections")
    objects = board.getChessboardCorners()[ids.ravel()].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(objects, corners, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or float(tvec[2, 0]) <= 0:
        raise ValueError("Invalid target pose")
    projected, _ = cv2.projectPoints(objects, rvec, tvec, k, dist)
    error = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    t[:3, 3] = tvec.ravel()
    return {
        "T_camera_target": t.tolist(),
        "corner_ids": ids.ravel().tolist(),
        "corners_px": corners.reshape(-1, 2).tolist(),
        "reprojection_rmse_px": float(np.sqrt(np.mean(error**2))),
        "reprojection_max_px": float(max(error)),
        "board": spec,
    }


def fixed_camera(candidates, max_translation_m=0.01, max_rotation_deg=1.0):
    """Each observation independently anchors a target in the chosen world.

    T_world_target must come from a measured fixture or calibrated robot mount;
    do not call an arbitrary table marker the arm's world without measuring it.
    """
    groups = {key: [x for x in candidates if x["split"] == key] for key in ("train", "validation")}
    if len(groups["train"]) < 3 or len(groups["validation"]) < 3:
        raise ValueError("Need >=3 training and >=3 held-out anchored target observations")
    estimates = {
        s: [transform(x["T_world_target"]) @ np.linalg.inv(transform(x["T_camera_target"])) for x in group]
        for s, group in groups.items()
    }
    result = mean_pose(estimates["train"])
    metrics = {key: pose_errors(result, ts) for key, ts in estimates.items()}
    v = metrics["validation"]
    return {
        "status": "candidate",
        "method": "anchored_target_fixed_camera",
        "T_world_camera": result.tolist(),
        "metrics": metrics,
        "validation_passed": max(v["translation_m"]) <= max_translation_m
        and max(v["rotation_deg"]) <= max_rotation_deg,
        "thresholds": {
            "max_translation_m": max_translation_m,
            "max_rotation_deg": max_rotation_deg,
        },
    }


def camera_cloud(metadata, arrays, camera, mask=None, stride=2):
    info = metadata["cameras"][camera]
    points = depth_points(
        arrays[camera + "_depth_raw"],
        info["depth"]["intrinsics"],
        info["depth"]["depth_scale_m_per_unit"],
        mask,
        stride,
    )
    return apply(rs_depth_to_color(info["depth_to_color"]), points)


def colored_camera_cloud(metadata, arrays, camera, stride=3):
    points = camera_cloud(metadata, arrays, camera, stride=stride)
    pixels, valid = project_color(points, metadata["cameras"][camera]["color"]["intrinsics"])
    pixels = np.floor(pixels[valid]).astype(int)
    colors = arrays[camera + "_rgb"][pixels[:, 1], pixels[:, 0]]
    return points[valid], colors


def icp_to_robot(camera_points, robot_points, initial):
    import open3d as o3d

    if len(camera_points) < 100 or len(robot_points) < 100:
        raise ValueError("At least 100 masked depth and robot points required")
    result = transform(initial).copy()
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(camera_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(robot_points))
    levels = []
    for voxel, distance in ((0.012, 0.05), (0.006, 0.025), (0.003, 0.012)):
        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=40))
        fit = o3d.pipelines.registration.registration_icp(
            s,
            t,
            distance,
            result,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        result = fit.transformation
        levels.append({"voxel_m": voxel, "fitness": fit.fitness, "inlier_rmse_m": fit.inlier_rmse})
    if levels[-1]["fitness"] < 0.25:
        raise ValueError("ICP overlap too low; inspect target mask and initial alignment")
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, 0.012, result
    )
    return {
        "status": "candidate",
        "method": "masked_depth_to_YAM_URDF_ICP",
        "T_world_camera": transform(result).tolist(),
        "levels": levels,
        "information_eigenvalues": np.linalg.eigvalsh(information).tolist(),
        "validation_passed": False,
        "limitation": "Single-pose ICP is ambiguous. Validate on separate joint poses and RGB/depth overlays. Geometry/zero offsets can bias this fit.",
    }


def validate_cloud(camera_points, robot_points, t_world_camera, threshold_m=0.01):
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(robot_points).query(apply(t_world_camera, camera_points))
    return {
        "count": len(distances),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(max(distances)),
        "fraction_within_threshold": float(np.mean(distances < threshold_m)),
        "threshold_m": threshold_m,
        "note": "One-sided masked depth-to-mesh distance; report occlusion and segmentation failures separately",
    }


def candidate_override(rig, camera_name, candidate, arms=None, model=None):
    """Copy a candidate into rig config without promoting it to validated."""
    import copy

    cfg = copy.deepcopy(rig)
    if camera_name not in cfg["transforms"]:
        raise ValueError("Unknown camera frame")
    entry = cfg["transforms"][camera_name]
    if "T_gripper_camera" in candidate:
        if entry["parent"] not in ("left_gripper", "right_gripper"):
            raise ValueError("Hand-eye calibration only applies to a wrist mount")
        t = transform(candidate["T_gripper_camera"])
    else:
        t = transform(candidate["T_world_camera"])
        if entry["parent"] != "world":
            if model is None or arms is None:
                raise ValueError("Wrist ICP conversion needs measured arm state and URDF FK")
            t = np.linalg.inv(model.fk(arms)[entry["parent"]]) @ t
    entry.update(T_parent_child=t.tolist(), status="candidate", source=candidate)
    return cfg
