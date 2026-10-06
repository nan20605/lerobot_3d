"""Offline checks do not constitute validation of the physical digital twin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def check_models(directory, poses=30, seed=7):
    import mujoco
    from scipy.spatial.transform import Rotation

    from .models import YamModel

    directory = Path(directory)
    model = YamModel(directory / "yam_bimanual.urdf")
    mj = mujoco.MjModel.from_xml_path(str((directory / "yam_bimanual.xml").resolve()))
    data = mujoco.MjData(mj)
    rng = np.random.default_rng(seed)
    max_t = max_r = 0.0
    for _ in range(poses):
        arms = {
            side: {
                "position_rad": [rng.uniform(*model.limits[f"{side}_joint{i}"]) for i in range(1, 7)],
                "gripper_open": rng.random(),
            }
            for side in ("left", "right")
        }
        for name, value in model.joint_config(arms).items():
            data.qpos[mj.joint(name).qposadr[0]] = value
        mujoco.mj_forward(mj, data)
        for name, t in model.fk(arms).items():
            if name.startswith("tcp_"):
                site = mj.site(name).id
                position = data.site_xpos[site]
                rotation = data.site_xmat[site].reshape(3, 3)
            elif name == "world":
                continue
            else:
                try:
                    body = mj.body(name).id
                except KeyError:
                    continue
                position = data.xpos[body]
                rotation = data.xmat[body].reshape(3, 3)
            max_t = max(max_t, float(np.linalg.norm(t[:3, 3] - position)))
            max_r = max(max_r, float(Rotation.from_matrix(t[:3, :3].T @ rotation).magnitude()))
    report = {
        "scope": "OFFLINE representation equivalence only",
        "poses": poses,
        "seed": seed,
        "max_link_translation_error_m": max_t,
        "max_link_rotation_error_rad": max_r,
        "passed": max_t < 1e-7 and max_r < 1e-7,
        "mujoco_version": mujoco.__version__,
    }
    (directory / "fk_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(report)
    return report


def render_model(directory, output, camera=None):
    import mujoco
    from PIL import Image

    directory = Path(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str((directory / "yam_bimanual.xml").resolve()))
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        for i, v in enumerate([0, 1.5, 1.5, 0, 0, 0, 0.023475, 0.023475], 1):
            d.qpos[m.joint(f"{side}_joint{i}").qposadr[0]] = v
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    c.lookat[:] = [0.15, -0.305, 0.4]
    c.distance = 2.0
    c.azimuth = 135
    c.elevation = -20
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    paths = []
    with mujoco.Renderer(m, height=480, width=640) as renderer:
        for name, view in [("station_nominal", c)] + [(m.camera(i).name, i) for i in range(m.ncam)]:
            if camera and name != camera:
                continue
            renderer.disable_depth_rendering()
            renderer.update_scene(d, camera=view, scene_option=options)
            path = output / f"{name}.png"
            Image.fromarray(renderer.render()).save(path)
            paths.append(str(path.resolve()))
            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera=view, scene_option=options)
            np.save(output / f"{name}_depth_m.npy", renderer.render())
    (output / "README.txt").write_text(
        "NOMINAL vendor geometry at synthetic joint positions. Not a render matched to a physical capture. Depth is ideal optical-axis distance in metres. No calibrated overhead camera is present yet.\n"
    )
    return paths
