"""YAM adapter for the existing LeRobot 3D mesh viewer and calibration workflow.

Hardware runs on the Jetson;
this module never constructs an I2RT motor interface, homes an arm, or assumes
SO101 encoder ticks. All poses are in metres/radians with explicit base rotation.
"""

from __future__ import annotations

import numpy as np

from lerobot_3d.common.types import RobotSnapshot

from .runtime.geometry import apply, quaternion
from .runtime.models import YamModel
from .runtime.transport import Client


class YamRobotState:
    def __init__(self, urdf_path, side: str, index: int = 0):
        import trimesh

        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.model = YamModel(urdf_path)
        self.side, self.index = side, index
        self.meshes = []
        self.points = []
        for link in self.model.urdf.links:
            if not link.name.startswith(side + "_"):
                continue
            for visual_index, visual in enumerate(link.visuals):
                mesh_geometry = visual.geometry.mesh
                if mesh_geometry is None:
                    continue
                for mesh_index, mesh in enumerate(mesh_geometry.meshes):
                    scaled = mesh.copy()
                    if mesh_geometry.scale is not None:
                        scaled.apply_scale(mesh_geometry.scale)
                    scaled.apply_transform(visual.origin)
                    self.meshes.append(
                        (
                            link.name,
                            f"visual_{visual_index}_{mesh_index}",
                            np.asarray(scaled.vertices),
                            np.asarray(scaled.faces),
                        )
                    )
                    sampled = trimesh.sample.sample_surface(scaled, 300, seed=visual_index)[0]
                    self.points.append((link.name, sampled))

    def get_static_meshes(self):
        return self.meshes

    def get_robot_snapshot(self, state):
        joints = self.model.joint_config({self.side: state})
        poses = self.model.fk({self.side: state})
        base = poses[f"{self.side}_base"]
        inv_base = np.linalg.inv(base)
        local = {name: inv_base @ t for name, t in poses.items() if name.startswith(self.side + "_")}
        per_link = {}
        for name, points in self.points:
            per_link.setdefault(name, []).append(apply(local[name], points))
        per_link = {name: np.concatenate(parts) for name, parts in per_link.items()}
        return RobotSnapshot(
            index=self.index,
            joint_positions=joints,
            joint_radians={k: v for k, v in joints.items() if int(k.rsplit("joint", 1)[1]) <= 6},
            pcd=np.concatenate(list(per_link.values())),
            link_pcds={},
            link_poses={name: (t[:3, 3], quaternion(t)) for name, t in local.items()},
            base_offset=base[:3, 3],
            base_wxyz=tuple(quaternion(base)),
        )


class YamRemoteRobot:
    def __init__(self, side, url="http://127.0.0.1:8765", token=None):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.client = Client(url, token)
        self.observation_id = None
        self.mode = None

    def get_observation(self):
        metadata = self.client.state()
        self.observation_id = metadata["observation_id"]
        self.mode = metadata["mode"]
        if self.side not in metadata["arms"]:
            raise ValueError("No joint feedback for requested arm")
        return metadata["arms"][self.side]

    def send_action(self, action):
        if self.observation_id is None or self.mode not in ("live", "simulation"):
            raise ValueError("Read fresh live/sim feedback before each command")
        observation_id = self.observation_id
        self.observation_id = None
        return self.client.command(self.side, action["position_rad"], action["gripper_open"], observation_id)
