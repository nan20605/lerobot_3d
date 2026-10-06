import json
import zipfile

import numpy as np
import pytest

from lerobot_3d.yam.runtime.bundle import audit_bundle, export_bundle
from lerobot_3d.yam.runtime.models import YamModel, build
from lerobot_3d.yam.runtime.validation import check_models


def test_portable_bundle_survives_relocation_and_detects_corruption(bundle, tmp_path):
    package = tmp_path / "twin.zip"
    report = export_bundle(bundle, package)
    assert not report["calibrated"]
    with zipfile.ZipFile(package) as archive:
        archive.extractall(tmp_path / "different_machine")
    relocated = tmp_path / "different_machine/yam_twin"
    audit = audit_bundle(relocated)
    assert audit["passed"]
    assert audit["descriptions"]["yam_left_base.urdf"]["root_link"] == "left_base"
    assert check_models(relocated, poses=5)["passed"]
    urdf = relocated / "yam_left_base.urdf"
    urdf.write_text(urdf.read_text() + "\n<!-- modified -->")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_bundle(relocated)


def test_arm_only_descriptions_match_world_description(bundle):
    full = YamModel(bundle / "yam_bimanual.urdf")
    state = {"position_rad": [0.2, 1.2, 1.8, 0.1, -0.1, 0.2], "gripper_open": 0.3}
    world = full.fk({"left": state, "right": state})
    for side in ("left", "right"):
        local = YamModel(bundle / f"yam_{side}_base.urdf").fk({side: state})
        for name, actual in local.items():
            np.testing.assert_allclose(actual, np.linalg.inv(world[side + "_base"]) @ world[name], atol=1e-10)


def test_calibrated_export_cannot_omit_a_camera(tmp_path, rig, vendor_path):
    rig["hardware_variant_status"] = "confirmed"
    for entry in rig["transforms"].values():
        entry["status"] = "validated"
    for entry in rig["cameras"].values():
        entry["intrinsics_status"] = "validated"
    rig["cameras"].pop("top_camera")
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(rig))
    with pytest.raises(ValueError, match="missing camera intrinsics"):
        build(vendor_path, path, tmp_path / "models", require_calibrated=True)
