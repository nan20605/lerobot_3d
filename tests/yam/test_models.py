import json

import mujoco
import numpy as np
import pytest

from lerobot_3d.yam.runtime.geometry import pose
from lerobot_3d.yam.runtime.models import YamModel, build
from lerobot_3d.yam.runtime.validation import check_models


def test_urdf_mujoco_fk_multiple_poses(bundle):
    assert check_models(bundle, poses=30)["passed"]


def test_overhead_rooted_exports_and_grippers(bundle):
    full = YamModel(bundle / "yam_bimanual.urdf")
    arms = {
        side: {"position_rad": [0.1, 1, 1.6, 0.2, -0.1, 0.3], "gripper_open": 0.7}
        for side in ("left", "right")
    }
    world = full.fk(arms)
    for side, state in arms.items():
        aligned = YamModel(bundle / f"yam_{side}_overhead.urdf").fk({side: state})
        for name, t in aligned.items():
            np.testing.assert_allclose(t, np.linalg.inv(world["top_camera"]) @ world[name], atol=1e-12)
        q = full.joint_config({side: state})
        assert q[f"{side}_joint7"] == pytest.approx(0.7 * 0.04695)
        assert q[f"{side}_joint8"] == q[f"{side}_joint7"]
    with pytest.raises(ValueError):
        full.joint_config({"left": {"position_rad": [0] * 6, "gripper_open": 1.1}})


def test_rotated_measured_overrides_apply_in_both_formats(tmp_path, rig, vendor_path):

    rig["transforms"]["right_base"]["T_parent_child"] = pose(
        [0.04, -0.64, 0.02], rpy=[0.02, -0.03, 0.1]
    ).tolist()
    rig["transforms"]["left_camera"]["T_parent_child"] = pose(
        [0.02, -0.05, 0.08], rpy=[2.5, 0.1, 0.2]
    ).tolist()
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(rig))
    build(vendor_path, path, tmp_path / "models")
    assert check_models(tmp_path / "models", poses=5)["passed"]


def test_no_false_calibrated_export(tmp_path, rig, vendor_path):

    p = tmp_path / "rig.json"
    p.write_text(json.dumps(rig))
    with pytest.raises(ValueError, match="Calibrated export refused"):
        build(vendor_path, p, tmp_path / "out", require_calibrated=True)


def test_native_lerobot_snapshot_honors_full_base_pose(bundle):
    from lerobot_3d.yam import YamRobotState

    adapter = YamRobotState(bundle / "yam_bimanual.urdf", "right", 1)
    state = {"position_rad": [0, 1.5, 1.5, 0, 0, 0], "gripper_open": 0.5}
    snap = adapter.get_robot_snapshot(state)
    base = pose(snap.base_offset, wxyz=snap.base_wxyz)
    for name, (xyz, q) in snap.link_poses.items():
        np.testing.assert_allclose(
            base @ pose(xyz, wxyz=q),
            adapter.model.fk({"right": state})[name],
            atol=1e-10,
        )
    assert snap.index == 1 and len(snap.pcd) > 1000


def test_simulation_stability_under_small_actuated_motion(bundle):
    from lerobot_3d.yam.runtime.acquisition import SimulationProvider

    sim = SimulationProvider(bundle / "yam_bimanual.xml")
    for _ in range(100):
        obs, _ = sim.snapshot()
    before = obs["arms"]["left"]["position_rad"][0]
    command = sim.targets["left"].copy()
    command[0] += 0.02
    sim.command("left", command)
    for _ in range(60):
        obs, _ = sim.snapshot()
    assert np.isfinite(sim.data.qpos).all()
    assert not np.any(sim.data.warning.number)
    assert obs["arms"]["left"]["position_rad"][0] - before == pytest.approx(0.02, abs=0.002)
    assert obs["mode"] == "simulation"


def test_rendered_camera_axes_principal_point_and_pixel_centres():
    # Off-centre point tests both axes, unequal focal lengths and noncentral K.
    m = mujoco.MjModel.from_xml_string("""<mujoco><visual><global offwidth="640" offheight="480"/></visual>
    <worldbody><camera name="c" quat="0 1 0 0" resolution="640 480" sensorsize="1 1"
    focalpixel="400 410" principalpixel="-20.5 -10.5"/>
    <geom type="sphere" pos="0.1 0.05 1" size="0.008" rgba="0 1 0 1"/></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, 480, 640) as r:
        r.update_scene(d, camera="c")
        img = r.render()
    y, x = np.where((img[:, :, 1].astype(float) > 2 * img[:, :, 0]) & (img[:, :, 1] > 20))
    assert x.mean() == pytest.approx(340 + 400 * 0.1, abs=0.5)
    assert y.mean() == pytest.approx(250 + 410 * 0.05, abs=0.5)
