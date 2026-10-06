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
    for name, view in [("station_nominal", c)] + [(m.camera(i).name, i) for i in range(m.ncam)]:
        if camera and name != camera:
            continue
        width, height = (640, 480) if name == "station_nominal" else m.cam_resolution[view]
        with mujoco.Renderer(m, height=int(height), width=int(width)) as renderer:
            renderer.disable_depth_rendering()
            renderer.update_scene(d, camera=view, scene_option=options)
            path = output / f"{name}.png"
            Image.fromarray(renderer.render()).save(path)
            paths.append(str(path.resolve()))
            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera=view, scene_option=options)
            np.save(output / f"{name}_depth_m.npy", renderer.render())
    (output / "README.txt").write_text(
        "Synthetic example joint positions, not a render matched to a physical capture. Depth is ideal optical-axis distance in metres. See rig.json and manifest.json for calibration status. Camera views use their configured resolution.\n"
    )
    return paths


def check_numerics(directory, duration=2.0):
    """Compare the same smooth synthetic trajectory at three integration steps.

    This tests numerical stability/convergence only. It does not identify real
    gains, latency, friction, camera mass, payload or contact parameters.
    """
    import mujoco

    runs = []
    traces = []
    for step in (0.002, 0.001, 0.0005):
        model = mujoco.MjModel.from_xml_path(str((Path(directory) / "yam_bimanual.xml").resolve()))
        model.opt.timestep = step
        data = mujoco.MjData(model)
        q0 = np.array([0, 1.5, 1.5, 0, 0, 0, 0.023475])
        addresses = []
        actuators = []
        for side in ("left", "right"):
            for i in range(1, 9):
                data.qpos[model.joint(f"{side}_joint{i}").qposadr[0]] = q0[min(i - 1, 6)]
            addresses.extend(int(model.joint(f"{side}_joint{i}").qposadr[0]) for i in range(1, 7))
            actuators.append([model.actuator(f"{side}_position{i}").id for i in range(1, 8)])
        mujoco.mj_forward(model, data)
        rows = []
        penetration = 0.0
        for index in range(round(duration / step)):
            target = q0.copy()
            target[:6] += 0.015 * np.sin(2 * np.pi * data.time / duration)
            for ids in actuators:
                data.ctrl[ids] = target
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
                raise ValueError("MuJoCo integration failed during the offline convergence check")
            if data.ncon:
                penetration = max(penetration, float(max(0, -np.min(data.contact.dist))))
            if (index + 1) % round(0.01 / step) == 0:
                rows.append(data.qpos[addresses].copy())
        traces.append(np.array(rows))
        runs.append(
            {
                "timestep_s": step,
                "max_contact_penetration_m": penetration,
                "warnings": data.warning.number.tolist(),
                "final_arm_radians": rows[-1].tolist(),
            }
        )
    comparisons = []
    for index in (0, 1):
        error = np.abs(traces[index] - traces[-1])
        comparisons.append(
            {
                "timestep_s": runs[index]["timestep_s"],
                "reference_timestep_s": runs[-1]["timestep_s"],
                "max_joint_difference_rad": float(error.max()),
                "rms_joint_difference_rad": float(np.sqrt(np.mean(error**2))),
            }
        )
    return {
        "scope": "synthetic timestep convergence; NOT hardware fidelity",
        "duration_s": duration,
        "sample_period_s": 0.01,
        "runs": runs,
        "comparisons": comparisons,
        "passed": bool(comparisons[0]["max_joint_difference_rad"] < 0.01),
        "criterion": "less than 0.01 rad versus 0.5 ms reference on this one synthetic trajectory",
        "physical_validation": False,
    }
