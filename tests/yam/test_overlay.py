import copy
import json

import mujoco
import numpy as np
import pytest

from lerobot_3d.yam.runtime.geometry import pose
from lerobot_3d.yam.runtime.models import build
from lerobot_3d.yam.runtime.overlay import overlay_capture
from lerobot_3d.yam.runtime.transport import encode
from lerobot_3d.yam.runtime.validation import render_model


def test_real_renderer_overlay_has_correct_registered_depth(tmp_path, rig, vendor_path):
    # Independent synthetic overhead sensor, looking down at the nominal rig.
    intrinsics = {
        "width": 320,
        "height": 240,
        "fx": 260,
        "fy": 260,
        "ppx": 159.5,
        "ppy": 119.5,
        "model": "none",
        "coeffs": [0] * 5,
    }
    extrinsic = {"rotation": np.eye(3).ravel(order="F").tolist(), "translation_m": [0, 0, 0]}
    rig["transforms"]["top_camera"]["T_parent_child"] = pose([0, -0.305, 1.1], wxyz=[0, 1, 0, 0]).tolist()
    rig["cameras"]["top_camera"] = {
        "serial": "SYNTHETIC_OVERHEAD",
        "model": "synthetic",
        "intrinsics_status": "synthetic",
        "intrinsics": intrinsics,
        "depth_intrinsics": intrinsics,
        "depth_scale_m_per_unit": 0.0001,
        "depth_to_color": extrinsic,
    }
    config = tmp_path / "rig.json"
    config.write_text(json.dumps(rig))
    directory = tmp_path / "models"
    build(vendor_path, config, directory)
    from PIL import Image

    preview = render_model(directory, tmp_path / "preview", camera="top_camera_rectified")
    assert Image.open(preview[0]).size == (320, 240)
    m = mujoco.MjModel.from_xml_path(str(directory / "yam_bimanual.xml"))
    d = mujoco.MjData(m)
    arms = {}
    for side in ("left", "right"):
        q = [0, 1.5, 1.5, 0, 0, 0]
        for i, value in enumerate(q + [0.02, 0.02], 1):
            d.qpos[m.joint(f"{side}_joint{i}").qposadr[0]] = value
        arms[side] = {
            "position_rad": q,
            "velocity_rad_s": [0] * 6,
            "gripper_open": 0.02 / 0.04695,
            "host_monotonic_s": 1,
        }
    mujoco.mj_forward(m, d)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    with mujoco.Renderer(m, height=240, width=320) as renderer:
        renderer.update_scene(d, camera="top_camera_rectified", scene_option=options)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    raw = np.zeros(depth.shape, np.uint16)
    valid = (depth > 0) & (depth < 2)
    raw[valid] = np.rint(depth[valid] / 0.0001).astype(np.uint16)
    metadata = {
        "schema_version": 1,
        "mode": "simulation",
        "arms": arms,
        "cameras": {
            "overhead": {
                "serial": "SIM:SYNTHETIC_OVERHEAD",
                "host_monotonic_s": 1,
                "color": {"intrinsics": intrinsics},
                "depth": {"intrinsics": intrinsics, "depth_scale_m_per_unit": 0.0001},
                "depth_to_color": extrinsic,
            }
        },
    }
    capture = tmp_path / "capture"
    capture.mkdir()
    packet = encode(metadata, {"overhead_rgb": rgb, "overhead_depth_raw": raw})
    (capture / "observation.npz").write_bytes(packet)
    report = overlay_capture(capture, "overhead", directory, tmp_path / "overlay", allow_simulation=True)
    assert report["predicted_robot_pixels"] > 1000
    assert report["compared_depth_pixels"] > 1000
    assert report["depth_residual_m"]["p95_absolute"] < 0.000051
    assert report["validation_passed"] is None
    assert (tmp_path / "overlay/comparison.png").exists()
    wrong = copy.deepcopy(metadata)
    wrong["cameras"]["overhead"]["serial"] = "WRONG_CAMERA"
    (capture / "observation.npz").write_bytes(encode(wrong, {"overhead_rgb": rgb, "overhead_depth_raw": raw}))
    with pytest.raises(ValueError, match="serial"):
        overlay_capture(capture, "overhead", directory, tmp_path / "bad", allow_simulation=True)
