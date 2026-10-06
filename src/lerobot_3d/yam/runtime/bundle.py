"""Audit and package portable vendor-derived URDF/MJCF assets without hardware."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import pose


def local_asset(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or external bundle asset: {name}")
    return path


def audit_bundle(directory):
    import mujoco
    from urchin import URDF

    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    artifacts = manifest["artifact_sha256"]
    for name, expected in artifacts.items():
        if hashlib.sha256(local_asset(root, name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    m = mujoco.MjModel.from_xml_path(str(root / "yam_bimanual.xml"))
    urdf = ET.parse(root / "yam_bimanual.urdf").getroot()
    rows = []
    for link in urdf.findall("link"):
        name = link.get("name")
        visuals = link.findall("visual")
        if visuals and len(link.findall("collision")) < len(visuals):
            raise ValueError(f"Missing collision geometry: {name}")
        for mesh in link.findall(".//mesh"):
            if mesh.get("filename") not in artifacts:
                raise ValueError(f"Unhashed mesh: {mesh.get('filename')}")
            local_asset(root, mesh.get("filename"))
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = inertial.find("inertia").attrib
        tensor = np.array(
            [
                [float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
                [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
                [float(i["ixz"]), float(i["iyz"]), float(i["izz"])],
            ]
        )
        origin = inertial.find("origin")
        com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rotation = pose(rpy=np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))[:3, :3]
        body = m.body(name)
        mj_rotation = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
        mj_tensor = mj_rotation @ np.diag(body.inertia) @ mj_rotation.T
        mass = float(inertial.find("mass").get("value"))
        error = float(np.max(np.abs(rotation @ tensor @ rotation.T - mj_tensor)))
        if (
            not np.isclose(mass, body.mass[0], atol=1e-10)
            or not np.allclose(com, body.ipos, atol=1e-9)
            or error > 1e-9
        ):
            raise ValueError(f"URDF/MJCF inertial mismatch: {name}")
        eigenvalues = np.linalg.eigvalsh(tensor)
        if mass <= 0 or min(eigenvalues) <= 0 or 2 * max(eigenvalues) > sum(eigenvalues) + 1e-12:
            raise ValueError(f"Nonphysical inertia: {name}")
        rows.append(
            {
                "link": name,
                "mass_kg": mass,
                "inertia_difference_kg_m2": error,
                "status": "CAD placeholder; needs measured override"
                if name.endswith("camera")
                else "vendor baseline",
            }
        )
    for joint in urdf.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        name = joint.get("name")
        mj_joint = m.joint(name)
        if not np.allclose([float(limit.get("lower")), float(limit.get("upper"))], mj_joint.range, atol=1e-8):
            raise ValueError(f"URDF/MJCF joint limit mismatch: {name}")
        number = int(name.rsplit("joint", 1)[1])
        if number != 8:
            actuator = m.actuator(name.replace("joint", "position"))
            if not np.isclose(float(limit.get("effort")), actuator.forcerange[1]):
                raise ValueError(f"URDF/MJCF actuator effort mismatch: {name}")
    descriptions = {}
    for name in artifacts:
        if name.endswith(".urdf"):
            model = URDF.load(str(root / name))
            descriptions[name] = {
                "root_link": model.base_link.name,
                "links": len(model.links),
                "joints": len(model.joints),
            }
    return {
        "scope": "offline asset and representation audit; no physical validation",
        "passed": True,
        "i2rt_revision": manifest["i2rt_revision"],
        "verified_asset_count": len(artifacts),
        "descriptions": descriptions,
        "inertials": rows,
        "collision_model": "per-link convex hulls; concavities and contact parameters remain unvalidated",
        "contact_exclusions": [
            [m.body(int(a)).name, m.body(int(b)).name]
            for a, b in zip(m.exclude_signature >> 16, m.exclude_signature & 0xFFFF)
        ],
        "calibrated": manifest["calibrated"],
        "calibration_blockers": manifest["calibration_blockers"],
    }


def export_bundle(directory, output):
    root, output = Path(directory).resolve(), Path(output)
    report = audit_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Include only enumerated artifacts: never sweep raw captures, credentials or
    # unrelated files into an export. Fixed timestamps make identical builds reproducible.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*manifest["artifact_sha256"], "manifest.json"]):
            info = zipfile.ZipInfo("yam_twin/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, local_asset(root, name).read_bytes())
    return {
        "output": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "calibrated": report["calibrated"],
        "verified_asset_count": report["verified_asset_count"],
    }
