"""Build both representations from one pinned vendor station and one rig config."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .geometry import camera_matrix, pose, quaternion, rs_depth_to_color, transform

I2RT_REV = "120c3c81400171174604e503943f8d1ebc891058"
LEROBOT_REV = "6b8d8582f60d3ce0d3297c320b0893d867cfc310"
STATION = "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405"


def numbers(value):
    return " ".join(format(float(x), ".17g") for x in value)


def xml_pose(node):
    return pose(
        np.fromstring(node.get("pos", "0 0 0"), sep=" "),
        wxyz=np.fromstring(node.get("quat", "1 0 0 0"), sep=" "),
    )


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def initial_config(vendor, capture_metadata=None):
    mj = ET.parse(str(Path(vendor) / STATION) + ".xml").getroot()
    transforms = {}
    for link, parent in (
        ("left_base", "world"),
        ("right_base", "world"),
        ("top_camera", "world"),
        ("left_camera", "left_gripper"),
        ("right_camera", "right_gripper"),
    ):
        transforms[link] = {
            "parent": parent,
            "T_parent_child": xml_pose(mj.find(f".//body[@name='{link}']")).tolist(),
            "status": "frame_definition" if link == "left_base" else "vendor_nominal",
            "source": f"I2RT {I2RT_REV}; NOT measured on this rig",
        }
    cameras = {}
    if capture_metadata:
        for side, meta in capture_metadata["cameras"].items():
            if side not in ("left", "right", "overhead"):
                raise ValueError("Camera metadata keys must be left, right or overhead")
            name = "top_camera" if side == "overhead" else f"{side}_camera"
            cameras[name] = {
                "serial": meta["serial"],
                "model": meta["name"],
                "intrinsics_status": "factory",
                "intrinsics": meta["color"]["intrinsics"],
                "depth_intrinsics": meta["depth"]["intrinsics"],
                "depth_scale_m_per_unit": meta["depth"]["depth_scale_m_per_unit"],
                "depth_to_color": meta["depth_to_color"],
                "source": capture_metadata.get(
                    "capture", capture_metadata.get("observation_id", "provided camera metadata")
                ),
            }
    for name in ("left_camera", "right_camera", "top_camera"):
        cameras.setdefault(name, {"model": None, "serial": None, "intrinsics_status": "unknown"})
    return {
        "schema_version": 1,
        "convention": "T_parent_child maps child into parent; metres, radians, wxyz",
        "hardware_variant_status": "unconfirmed",
        "arm": "yam_v1",
        "gripper": "linear_4310",
        "transforms": transforms,
        "cameras": cameras,
        "inertial_overrides": {},
        "dynamics_status": "unvalidated",
        "contact_status": "unvalidated",
        "environment_status": "unmeasured",
        "environment_boxes": [],
    }


def build(vendor, config_path, output, require_calibrated=False):
    vendor, output = Path(vendor).resolve(), Path(output).resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if revision != I2RT_REV:
        raise ValueError(f"Expected I2RT {I2RT_REV}, got {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty.strip():
        raise ValueError("Vendor checkout has tracked modifications; use explicit rig overrides")
    cfg = json.loads(Path(config_path).read_text())
    if cfg["schema_version"] != 1 or cfg["arm"] != "yam_v1" or cfg["gripper"] != "linear_4310":
        raise ValueError("Only pinned YAM v1 / linear_4310 station is currently supported")
    expected = {
        "left_base": "world",
        "right_base": "world",
        "top_camera": "world",
        "left_camera": "left_gripper",
        "right_camera": "right_gripper",
    }
    for name, parent in expected.items():
        entry = cfg["transforms"][name]
        if entry["parent"] != parent:
            raise ValueError(f"{name} must be relative to {parent}")
        transform(entry["T_parent_child"])
    if set(cfg["transforms"]) != set(expected):
        raise ValueError("Rig must define exactly the two bases and three camera mounts")
    blockers = [
        f"{k}: {v['status']}"
        for k, v in cfg["transforms"].items()
        if v["status"] != "validated" and not (k == "left_base" and v["status"] == "frame_definition")
    ]
    blockers += [
        f"{k}: intrinsics {v['intrinsics_status']}"
        for k, v in cfg["cameras"].items()
        if v["intrinsics_status"] != "validated"
    ]
    for name in ("left_camera", "right_camera", "top_camera"):
        if name not in cfg["cameras"] or "intrinsics" not in cfg["cameras"][name]:
            blockers.append(f"{name}: missing camera intrinsics")
    if cfg["hardware_variant_status"] != "confirmed":
        blockers.append("installed arm/gripper/mount variants unconfirmed")
    if cfg.get("calibration_source_mode", "live") != "live":
        blockers.append("calibration data are synthetic, not physical measurements")
    if require_calibrated and blockers:
        raise ValueError("Calibrated export refused: " + "; ".join(blockers))
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    station = vendor / STATION
    urdf = ET.parse(str(station) + ".urdf").getroot()
    mj = ET.parse(str(station) + ".xml").getroot()
    provenance = {}
    collision_provenance = {}

    def mesh_path(filename):
        source = (station.parent / filename).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        name = f"{digest[:12]}_{source.name}"
        shutil.copyfile(source, output / "meshes" / name)
        provenance[name] = {"sha256": digest, "source": str(source.relative_to(vendor))}
        return f"meshes/{name}"

    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", mesh_path(mesh.attrib["filename"]))
    for mesh in mj.findall("asset/mesh"):
        mesh.set("file", mesh_path(mesh.attrib["file"]))
    # Upstream URDFs have visual meshes but no collision elements. Export the
    # same per-link convex hull baseline used by MuJoCo's mesh collision engine.
    # A hull is not a measured contact model or a convex decomposition.
    import trimesh

    (output / "collisions").mkdir(exist_ok=True)
    hull_paths = {}
    for link in urdf.findall("link"):
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            source_mesh = geometry.find("mesh")
            if source_mesh is None:
                continue
            original = source_mesh.get("filename")
            if original not in hull_paths:
                hull = trimesh.load_mesh(output / original).convex_hull
                payload = hull.export(file_type="stl")
                digest = hashlib.sha256(payload).hexdigest()
                filename = f"collisions/{digest[:12]}_{Path(original).name}"
                (output / filename).write_bytes(payload)
                hull_paths[original] = filename
                collision_provenance[filename] = {
                    "sha256": digest,
                    "visual_mesh": original,
                    "method": "trimesh convex_hull; MuJoCo baseline, not contact calibration",
                }
            collision = ET.SubElement(link, "collision")
            if visual.find("origin") is not None:
                collision.append(copy.deepcopy(visual.find("origin")))
            collision.append(copy.deepcopy(geometry))
            collision.find("geometry/mesh").set("filename", hull_paths[original])
    mj.find("compiler").set("meshdir", ".")
    # Flat world attachment makes independent measured base/overhead poses explicit.
    ET.SubElement(urdf, "link", name="world")
    for name, entry in cfg["transforms"].items():
        t = transform(entry["T_parent_child"])
        joint = next(
            (j for j in urdf.findall("joint") if j.find("child").get("link") == name),
            None,
        )
        if joint is None:
            joint = ET.SubElement(urdf, "joint", name=f"world_to_{name}", type="fixed")
            ET.SubElement(joint, "parent")
            ET.SubElement(joint, "child", link=name)
        joint.find("parent").set("link", entry["parent"])
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", numbers(t[:3, 3]))
        origin.set("rpy", numbers(Rotation.from_matrix(t[:3, :3]).as_euler("xyz")))
        body = mj.find(f".//body[@name='{name}']")
        parent = next(p for p in mj.iter() if body in list(p))
        new_parent = (
            mj.find("worldbody")
            if entry["parent"] == "world"
            else mj.find(f".//body[@name='{entry['parent']}']")
        )
        parent.remove(body)
        new_parent.append(body)
        body.set("pos", numbers(t[:3, 3]))
        body.set("quat", numbers(quaternion(t)))
    # Add matching tool frames to URDF, using vendor MJCF sites verbatim.
    for side in ("left", "right"):
        site = mj.find(f".//site[@name='tcp_{side}']")
        t = xml_pose(site)
        ET.SubElement(urdf, "link", name=f"tcp_{side}")
        j = ET.SubElement(urdf, "joint", name=f"{side}_tool_frame", type="fixed")
        ET.SubElement(j, "parent", link=f"{side}_gripper")
        ET.SubElement(j, "child", link=f"tcp_{side}")
        ET.SubElement(
            j,
            "origin",
            xyz=numbers(t[:3, 3]),
            rpy=numbers(Rotation.from_matrix(t[:3, :3]).as_euler("xyz")),
        )
        # Both fingers are represented but only one independent gripper coordinate.
        ET.SubElement(
            urdf.find(f"joint[@name='{side}_joint8']"),
            "mimic",
            joint=f"{side}_joint7",
            multiplier="1",
            offset="0",
        )
    # Optional measured inertia overrides apply identically to both model formats.
    for link, item in cfg.get("inertial_overrides", {}).items():
        mass, com, inertia = (
            float(item["mass_kg"]),
            np.array(item["com_m"]),
            np.array(item["inertia_kg_m2"]),
        )
        if inertia.shape != (3, 3) or not np.isfinite(inertia).all() or not np.isfinite(mass):
            raise ValueError(f"Invalid physical inertia for {link}")
        eig, axes = np.linalg.eigh(inertia)
        if (
            mass <= 0
            or com.shape != (3,)
            or not np.isfinite(com).all()
            or np.min(eig) <= 0
            or 2 * max(eig) > sum(eig) + 1e-12
            or not np.allclose(inertia, inertia.T)
        ):
            raise ValueError(f"Invalid physical inertia for {link}")
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        u = urdf.find(f"link[@name='{link}']/inertial")
        u.find("mass").set("value", str(mass))
        u.find("origin").set("xyz", numbers(com))
        u.find("origin").set("rpy", "0 0 0")
        for key, val in zip(
            ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
            inertia[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        ):
            u.find("inertia").set(key, str(val))
        m = mj.find(f".//body[@name='{link}']/inertial")
        m.set("mass", str(mass))
        m.set("pos", numbers(com))
        m.set("diaginertia", numbers(eig))
        m.set("quat", numbers(Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]))
    # Provisional simulation actuators. Arm gains from vendor config; gripper gains
    # expressed in metres (NOT copied from its motor-angle gains).
    gains = yaml.safe_load((vendor / "i2rt/robots/config/yam_v1.yml").read_text())
    ET.SubElement(
        mj,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        iterations="100",
        tolerance="1e-10",
    )
    act = ET.SubElement(mj, "actuator")
    contact = ET.SubElement(mj, "contact")
    for side in ("left", "right"):
        # MuJoCo's automatic parent filter does not cover a child of the welded
        # world. Explicitly exclude only the base/joint1 bearing overlap.
        ET.SubElement(contact, "exclude", body1=f"{side}_base", body2=f"{side}_link1")
        for i in range(1, 8):
            joint = mj.find(f".//joint[@name='{side}_joint{i}']")
            kp, kd, effort = (gains["kp"][i - 1], gains["kd"][i - 1], 10) if i < 7 else (400, 10, 20)
            ET.SubElement(
                act,
                "position",
                name=f"{side}_position{i}",
                joint=f"{side}_joint{i}",
                kp=str(kp),
                kv=str(kd),
                ctrlrange=joint.get("range"),
                forcerange=f"{-effort} {effort}",
            )
            # Vendor URDF effort=1 placeholders disagree with its MJCF arm cap.
            # Preserve that cap for arms and label the provisional gripper force.
            urdf.find(f"joint[@name='{side}_joint{i}']/limit").set("effort", str(effort))
        urdf.find(f"joint[@name='{side}_joint8']/limit").set("effort", "0")
    visual = ET.SubElement(mj, "visual")
    framebuffer = ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "map", znear="0.002", zfar="10")
    ET.SubElement(
        mj.find("worldbody"),
        "light",
        pos="0 -0.3 2",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
    )
    rendered = []
    for name, camera in cfg["cameras"].items():
        if "intrinsics" not in camera:
            continue  # Unknown overhead camera: no invented sensor model.
        streams = [("rectified", camera["intrinsics"], np.eye(4))]
        if "depth_intrinsics" in camera:
            streams.append(
                (
                    "depth",
                    camera["depth_intrinsics"],
                    rs_depth_to_color(camera["depth_to_color"]),
                )
            )
        for stream, i, optical in streams:
            camera_matrix(i)
            if any(not isinstance(i[key], int) or i[key] <= 0 for key in ("width", "height")):
                raise ValueError("Camera dimensions must be positive integers")
            framebuffer.set("offwidth", str(max(int(framebuffer.get("offwidth")), i["width"])))
            framebuffer.set("offheight", str(max(int(framebuffer.get("offheight")), i["height"])))
            renderer_pose = optical @ pose(wxyz=[0, 1, 0, 0])
            ET.SubElement(
                mj.find(f".//body[@name='{name}']"),
                "camera",
                name=name + "_" + stream,
                pos=numbers(renderer_pose[:3, 3]),
                quat=numbers(quaternion(renderer_pose)),
                resolution=f"{i['width']} {i['height']}",
                sensorsize="1 1",
                focalpixel=f"{i['fx']} {i['fy']}",
                principalpixel=f"{(i['width'] - 1) / 2 - i['ppx']} {(i['height'] - 1) / 2 - i['ppy']}",
            )
            rendered.append(name + "_" + stream)
    for box in cfg.get("environment_boxes", []):
        if box.get("status") != "measured":
            raise ValueError("Environment boxes require measured dimensions and poses")
        t = transform(box["T_world_box"])
        size = np.array(box["size_m"], float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid box dimensions")
        ET.SubElement(
            mj.find("worldbody"),
            "geom",
            name=box["name"],
            type="box",
            size=numbers(size / 2),
            pos=numbers(t[:3, 3]),
            quat=numbers(quaternion(t)),
            rgba="0.4 0.4 0.4 1",
        )
        link = ET.SubElement(urdf, "link", name=box["name"])
        for kind in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(link, kind), "geometry")
            ET.SubElement(geometry, "box", size=numbers(size))
        j = ET.SubElement(urdf, "joint", name="world_to_" + box["name"], type="fixed")
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=box["name"])
        ET.SubElement(
            j,
            "origin",
            xyz=numbers(t[:3, 3]),
            rpy=numbers(Rotation.from_matrix(t[:3, :3]).as_euler("xyz")),
        )
    write_xml(urdf, output / "yam_bimanual.urdf")
    write_xml(mj, output / "yam_bimanual.xml")
    for side, other in (("left", "right"), ("right", "left")):
        single = copy.deepcopy(urdf)
        remove = {
            x.get("name")
            for x in single.findall("link")
            if x.get("name").startswith(other + "_") or x.get("name") == f"tcp_{other}"
        }
        for node in list(single):
            if (
                node.tag == "link"
                and node.get("name") in remove
                or node.tag == "joint"
                and (node.find("child").get("link") in remove or node.find("parent").get("link") in remove)
            ):
                single.remove(node)
        single.set("name", f"yam_{side}")
        write_xml(single, output / f"yam_{side}.urdf")
        # Arm-only descriptions are useful to consumers that supply their own
        # world/base transforms. Keep every descendant of the chosen arm base.
        arm_only = copy.deepcopy(single)
        descendants = {f"{side}_base"}
        while True:
            expanded = descendants | {
                j.find("child").get("link")
                for j in arm_only.findall("joint")
                if j.find("parent").get("link") in descendants
            }
            if expanded == descendants:
                break
            descendants = expanded
        for node in list(arm_only):
            if (
                node.tag == "link"
                and node.get("name") not in descendants
                or node.tag == "joint"
                and (
                    node.find("parent").get("link") not in descendants
                    or node.find("child").get("link") not in descendants
                )
            ):
                arm_only.remove(node)
        write_xml(arm_only, output / f"yam_{side}_base.urdf")
        # Also deliver a description rooted directly in the overhead optical frame.
        overhead = copy.deepcopy(single)
        j = next(j for j in overhead.findall("joint") if j.find("child").get("link") == "top_camera")
        j.find("parent").set("link", "top_camera")
        j.find("child").set("link", "world")
        t = np.linalg.inv(transform(cfg["transforms"]["top_camera"]["T_parent_child"]))
        j.set("name", "overhead_to_world")
        j.find("origin").set("xyz", numbers(t[:3, 3]))
        j.find("origin").set("rpy", numbers(Rotation.from_matrix(t[:3, :3]).as_euler("xyz")))
        write_xml(overhead, output / f"yam_{side}_overhead.urdf")
    shutil.copyfile(config_path, output / "rig.json")
    shutil.copyfile(vendor / "LICENSE", output / "I2RT_LICENSE")
    manifest = {
        "schema_version": 1,
        "i2rt_revision": revision,
        "calibrated": not blockers,
        "calibration_blockers": blockers,
        "meshes": provenance,
        "collision_meshes": collision_provenance,
        "source_files": {
            suffix: {
                "url": f"https://github.com/i2rt-robotics/i2rt/blob/{revision}/{STATION}.{suffix}",
                "sha256": hashlib.sha256(Path(str(station) + "." + suffix).read_bytes()).hexdigest(),
            }
            for suffix in ("urdf", "xml")
        },
        "joint_limits": {
            j.get("name"): {
                "type": j.get("type"),
                **{key: float(value) for key, value in j.find("limit").attrib.items()},
                "effort_source": "passive mimic"
                if j.get("name").endswith("joint8")
                else "provisional linear force"
                if j.get("name").endswith("joint7")
                else "vendor MJCF actuatorfrcrange",
                "velocity_status": "vendor URDF placeholder; not validated hardware speed",
            }
            for j in urdf.findall("joint")
            if j.find("limit") is not None
        },
        "rendered_cameras": rendered,
        "geometry": "vendor station; actual mounting dimensions unverified",
        "dynamics": "UNVALIDATED: camera unit-density CAD inertias unless overridden; provisional force limits and gripper gains; no motor latency/friction fit",
        "contacts": "UNVALIDATED: per-link convex hulls in both formats, no measured friction or table unless configured; preserves vendor finger exclusions and excludes each base/link1 bearing pair in addition to MuJoCo defaults",
        "rendering": "Ideal rectified pinhole RGB/depth; no measured lighting, noise or rolling shutter",
        "camera_convention": "optical +X right +Y down +Z forward; renderer Rx(pi); principalpixel=((w-1)/2-cx, (h-1)/2-cy), accounting for integer pixel centres",
    }
    portable = [
        output / "yam_bimanual.urdf",
        *(
            output / f"yam_{side}{suffix}.urdf"
            for side in ("left", "right")
            for suffix in ("", "_base", "_overhead")
        ),
        output / "yam_bimanual.xml",
        output / "rig.json",
        output / "I2RT_LICENSE",
        *(output / "meshes" / name for name in provenance),
        *(output / name for name in collision_provenance),
    ]
    manifest["artifact_sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(portable)
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class YamModel:
    """Joint angles in radians; normalized gripper 0 closed / 1 open."""

    def __init__(self, urdf):
        from urchin import URDF

        self.urdf = URDF.load(str(urdf))
        self.limits = {
            j.name: (j.limit.lower, j.limit.upper) for j in self.urdf.joints if j.joint_type != "fixed"
        }

    def joint_config(self, arms):
        result = {}
        for side, state in arms.items():
            if side not in ("left", "right"):
                raise ValueError("Unknown arm")
            q = np.asarray(state["position_rad"], float)
            g = float(state["gripper_open"])
            if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(g) or not 0 <= g <= 1:
                raise ValueError("Expected six finite radians and gripper_open in [0,1]")
            for index, value in enumerate(q, 1):
                key = f"{side}_joint{index}"
                if key in self.limits:
                    lo, hi = self.limits[key]
                    if not lo - 1e-5 <= value <= hi + 1e-5:
                        raise ValueError(f"Measured {key} outside URDF range; verify hardware mapping")
                    result[key] = float(value)
            for index in (7, 8):
                key = f"{side}_joint{index}"
                if key in self.limits:
                    lo, hi = self.limits[key]
                    result[key] = lo + g * (hi - lo)
        return result

    def fk(self, arms):
        return self.urdf.link_fk(cfg=self.joint_config(arms), use_names=True)

    def mesh_points(self, arms, side=None, count=12000, seed=0):
        import trimesh

        cfg = self.joint_config(arms)
        links = [x for x in self.urdf.links if side is None or x.name.startswith(side + "_")]
        meshes = []
        for mesh, t in self.urdf.visual_trimesh_fk(cfg=cfg, links=links).items():
            m = mesh.copy()
            m.apply_transform(t)
            meshes.append(m)
        if not meshes:
            raise ValueError("No visual meshes selected")
        return trimesh.sample.sample_surface(trimesh.util.concatenate(meshes), count, seed=seed)[0]
