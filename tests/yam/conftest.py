"""Synthetic YAM fixtures. No physical hardware or private rig files required."""

import copy
import json
import os
from pathlib import Path

import pytest

from lerobot_3d.yam.runtime.models import build, initial_config


@pytest.fixture(scope="session")
def vendor_path():
    path = Path(os.environ.get("I2RT_CHECKOUT", Path(__file__).resolve().parents[2] / "third_party/i2rt"))
    if not (path / "i2rt/robot_models").is_dir():
        pytest.skip("Set I2RT_CHECKOUT to the pinned vendor checkout for model tests")
    return path


@pytest.fixture(scope="session")
def nominal_rig(vendor_path):
    value = initial_config(vendor_path)
    for side in ("left", "right"):
        value["cameras"][side + "_camera"] = {
            "serial": "SYNTHETIC_" + side,
            "model": "synthetic pinhole",
            "intrinsics_status": "factory",
            "intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 400.0,
                "fy": 395.0,
                "ppx": 321.0,
                "ppy": 242.0,
                "model": "inverse_brown_conrady",
                "coeffs": [-0.05, 0.05, 0.0005, 0.0003, -0.018],
            },
        }
    return value


@pytest.fixture
def rig(nominal_rig):
    return copy.deepcopy(nominal_rig)


@pytest.fixture(scope="session")
def bundle(tmp_path_factory, vendor_path, nominal_rig):
    p = tmp_path_factory.mktemp("yam_models")
    path = p / "input.json"
    path.write_text(json.dumps(nominal_rig))
    build(vendor_path, path, p / "models")
    return p / "models"
