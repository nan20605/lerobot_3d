"""Reproject a static paired capture through the candidate twin for inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import camera_cloud, file_hash
from .geometry import project_color, rectify_rgb
from .models import YamModel
from .workflow import read_paired_capture


def overlay_capture(capture, camera, model_dir, output, *, allow_simulation=False):
    import cv2
    import mujoco
    from PIL import Image, ImageDraw

    meta, arrays = read_paired_capture(capture, camera, allow_simulation=allow_simulation)
    root, output = Path(model_dir), Path(output)
    rig = json.loads((root / "rig.json").read_text())
    frame = "top_camera" if camera == "overhead" else camera + "_camera"
    current = meta["cameras"][camera]["color"]["intrinsics"]
    expected = rig["cameras"][frame].get("intrinsics")
    if expected is None or any(
        not np.isclose(current[k], expected[k]) for k in ("width", "height", "fx", "fy", "ppx", "ppy")
    ):
        raise ValueError("Build a model using the captured camera intrinsics before making overlays")
    expected_serial = rig["cameras"][frame].get("serial")
    accepted_serials = {expected_serial}
    if meta["mode"] == "simulation":
        accepted_serials.add("SIM:" + str(expected_serial))
    if meta["cameras"][camera]["serial"] not in accepted_serials:
        raise ValueError("Camera serial differs from the model; verify the physical mapping")
    rgb, rectified = rectify_rgb(arrays[camera + "_rgb"], current)
    model = mujoco.MjModel.from_xml_path(str((root / "yam_bimanual.xml").resolve()))
    data = mujoco.MjData(model)
    kin = YamModel(root / "yam_bimanual.urdf")
    for name, value in kin.joint_config(meta["arms"]).items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    height, width = rgb.shape[:2]
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=frame + "_rectified", scene_option=options)
        rendered = renderer.render().copy()
        renderer.enable_depth_rendering()
        rendered_depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    geom_ids = segmentation[:, :, 0]
    is_geom = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    robot_geoms = [
        g
        for g in range(model.ngeom)
        if model.body(int(model.geom_bodyid[g])).name.startswith(("left_", "right_"))
    ]
    predicted_mask = is_geom & np.isin(geom_ids, robot_geoms)
    # Raw depth and RGB are not registered. Reproject through the factory rigid
    # transform and z-buffer into the rectified color camera before comparison.
    points = camera_cloud(meta, arrays, camera, stride=1)
    pixels, valid = project_color(points, rectified)
    pixels, z = np.rint(pixels[valid]).astype(int), points[valid, 2]
    inside = (pixels[:, 0] < width) & (pixels[:, 1] < height)
    pixels, z = pixels[inside], z[inside]
    measured = np.full((height, width), np.inf)
    np.minimum.at(measured, (pixels[:, 1], pixels[:, 0]), z)
    measured[~np.isfinite(measured)] = np.nan
    comparable = predicted_mask & np.isfinite(measured) & (rendered_depth > 0)
    residual = np.full((height, width), np.nan)
    residual[comparable] = measured[comparable] - rendered_depth[comparable]
    values = residual[comparable]
    overlay = rgb.copy()
    overlay[predicted_mask] = (0.55 * rgb[predicted_mask] + 0.45 * rendered[predicted_mask]).astype(np.uint8)
    contours, _ = cv2.findContours(
        predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 180, 0), 1)
    heat = np.full_like(rgb, 35)
    scaled = np.clip(np.nan_to_num(residual) / 0.05, -1, 1)
    heat[comparable] = np.column_stack(
        (
            255 * np.maximum(scaled[comparable], 0),
            255 * (1 - abs(scaled[comparable])),
            255 * np.maximum(-scaled[comparable], 0),
        )
    ).astype(np.uint8)
    output.mkdir(parents=True, exist_ok=False)
    for name, image in (
        ("real_rectified", rgb),
        ("predicted", rendered),
        ("overlay", overlay),
        ("depth_residual", heat),
    ):
        Image.fromarray(image).save(output / f"{name}.png")
    np.save(output / "depth_residual_m.npy", residual)
    np.save(output / "predicted_depth_m.npy", rendered_depth)
    panel = Image.new("RGB", (3 * width, height + 48), "white")
    draw = ImageDraw.Draw(panel)
    for index, (title, image) in enumerate(
        (
            ("Rectified capture", rgb),
            ("Twin overlay (orange outline)", overlay),
            ("Depth: red +5 cm / blue -5 cm", heat),
        )
    ):
        panel.paste(Image.fromarray(image), (index * width, 48))
        draw.text((index * width + 8, 8), title, fill="black")
        draw.text(
            (index * width + 8, 25),
            f"{meta['mode']} input; mount: {rig['transforms'][frame]['status']}",
            fill="black",
        )
    panel.save(output / "comparison.png")
    report = {
        "source_mode": meta["mode"],
        "camera": camera,
        "mount_status": rig["transforms"][frame]["status"],
        "capture_sha256": file_hash(Path(capture) / "observation.npz"),
        "urdf_sha256": file_hash(root / "yam_bimanual.urdf"),
        "predicted_robot_pixels": int(predicted_mask.sum()),
        "compared_depth_pixels": len(values),
        "depth_residual_m": {
            "median_signed": float(np.median(values)),
            "median_absolute": float(np.median(abs(values))),
            "p95_absolute": float(np.quantile(abs(values), 0.95)),
        }
        if len(values)
        else None,
        "validation_passed": None,
        "limitations": "Occluders, segmentation and depth holes affect residuals. Inspect held-out captures; this diagnostic never accepts calibration automatically.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
