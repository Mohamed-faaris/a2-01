#!/usr/bin/env python3
"""Kinematic model and forward kinematics engine for DOBOT CR5A robot.

Implements forward kinematics matching:
1. /home/mfk01/Downloads/Robotics_Resources/cr5/urdf/cr5a.urdf
2. CR5A user guide V1.8 Appendix A & cuRobo kinematic chain.
3. End-effector pose recorded in joint_angles.yaml across all 18 poses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger("hand-eye-0.kinematics")

DEFAULT_CR5_RESOURCE_DIR = Path("/home/mfk01/Downloads/Robotics_Resources/cr5")


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """Euler XYZ (fixed roll, pitch, yaw) to 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", [r, p, y]).as_matrix()


def make_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Creates a 4x4 homogeneous transformation matrix from xyz and rpy."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def joint_z_rotation(angle_rad: float) -> np.ndarray:
    """Creates a 4x4 transform for rotation around local Z axis."""
    T = np.eye(4, dtype=np.float64)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


@dataclass
class CR5LinkTransforms:
    """Collection of forward kinematics frames in base_link coordinates."""
    base_link: np.ndarray        # 4x4
    link1: np.ndarray            # 4x4
    link2: np.ndarray            # 4x4
    link3: np.ndarray            # 4x4
    link4: np.ndarray            # 4x4
    link5: np.ndarray            # 4x4
    link6: np.ndarray            # 4x4
    flange: np.ndarray           # 4x4
    tcp: np.ndarray              # 4x4


@dataclass
class CR5RodSegment:
    """A cylindrical rod segment representing a robot link."""
    name: str
    p_start: np.ndarray          # 3D start point [x, y, z] in metres
    p_end: np.ndarray            # 3D end point [x, y, z] in metres
    radius: float                # Cylinder radius in metres
    color: tuple[int, int, int]  # RGB color 0-255


def compute_cr5_fk(q_rad: Sequence[float]) -> CR5LinkTransforms:
    """Computes full forward kinematics for DOBOT CR5A given 6 joint angles in radians.

    Kinematics definition from cr5a.urdf:
      - joint1: origin [0, 0, 0.147], rpy [0, 0, 0], axis Z
      - joint2: origin [0, 0, 0], rpy [pi/2, pi/2, 0], axis Z
      - joint3: origin [-0.427, -0.000349, 0], rpy [0, 0, 0], axis Z
      - joint4: origin [-0.357, 0.000349, 0.141], rpy [0, 0, -pi/2], axis Z
      - joint5: origin [0, -0.116, 0], rpy [pi/2, 0, 0], axis Z
      - joint6: origin [0, 0.105, 0], rpy [-pi/2, 0, 0], axis Z
      - flange: origin [0, 0, 0], rpy [0, 0, 0]
      - tcp: origin [0, -0.117, 0.180], rpy [0, 0, -pi/2]
    """
    T_base = np.eye(4, dtype=np.float64)

    # Joint 1
    T_b_j1_fixed = make_transform([0.0, 0.0, 0.147], [0.0, 0.0, 0.0])
    T_base_link1 = T_base @ T_b_j1_fixed @ joint_z_rotation(q_rad[0])

    # Joint 2
    T_l1_j2_fixed = make_transform([0.0, 0.0, 0.0], [np.pi / 2, np.pi / 2, 0.0])
    T_base_link2 = T_base_link1 @ T_l1_j2_fixed @ joint_z_rotation(q_rad[1])

    # Joint 3
    T_l2_j3_fixed = make_transform([-0.427, -0.000349, 0.0], [0.0, 0.0, 0.0])
    T_base_link3 = T_base_link2 @ T_l2_j3_fixed @ joint_z_rotation(q_rad[2])

    # Joint 4
    T_l3_j4_fixed = make_transform([-0.357, 0.000349, 0.141], [0.0, 0.0, -np.pi / 2])
    T_base_link4 = T_base_link3 @ T_l3_j4_fixed @ joint_z_rotation(q_rad[3])

    # Joint 5
    T_l4_j5_fixed = make_transform([0.0, -0.116, 0.0], [np.pi / 2, 0.0, 0.0])
    T_base_link5 = T_base_link4 @ T_l4_j5_fixed @ joint_z_rotation(q_rad[4])

    # Joint 6
    T_l5_j6_fixed = make_transform([0.0, 0.105, 0.0], [-np.pi / 2, 0.0, 0.0])
    T_base_link6 = T_base_link5 @ T_l5_j6_fixed @ joint_z_rotation(q_rad[5])

    # Flange
    T_base_flange = T_base_link6.copy()

    # TCP (PGEA-100-40 grasp centre)
    T_l6_tcp_fixed = make_transform([0.0, -0.117, 0.180], [0.0, 0.0, -np.pi / 2])
    T_base_tcp = T_base_link6 @ T_l6_tcp_fixed

    return CR5LinkTransforms(
        base_link=T_base,
        link1=T_base_link1,
        link2=T_base_link2,
        link3=T_base_link3,
        link4=T_base_link4,
        link5=T_base_link5,
        link6=T_base_link6,
        flange=T_base_flange,
        tcp=T_base_tcp,
    )


def compute_cr5_rods(transforms: CR5LinkTransforms) -> list[CR5RodSegment]:
    """Extracts 3D rod segments connecting key joint pivots for skeleton rendering."""
    p_base = transforms.base_link[:3, 3]       # [0, 0, 0]
    p_j1 = transforms.link1[:3, 3]            # [0, 0, 0.147]
    p_j2 = transforms.link2[:3, 3]            # [0, 0, 0.147]
    p_j3 = transforms.link3[:3, 3]            # Elbow pivot
    p_j4 = transforms.link4[:3, 3]            # Forearm pivot
    p_j5 = transforms.link5[:3, 3]            # Wrist 1 pivot
    p_j6 = transforms.link6[:3, 3]            # Wrist 2 pivot
    p_tcp = transforms.tcp[:3, 3]             # Tool Center Point

    rods = [
        # Base column / pedestal rod
        CR5RodSegment(
            name="rod_pedestal",
            p_start=p_base,
            p_end=p_j1,
            radius=0.045,
            color=(45, 55, 72),  # Slate dark metallic
        ),
        # Upper arm rod (Link 2 -> Link 3)
        CR5RodSegment(
            name="rod_upper_arm",
            p_start=p_j2,
            p_end=p_j3,
            radius=0.038,
            color=(0, 210, 180),  # Bright cyan / robotics teal
        ),
        # Forearm rod (Link 3 -> Link 4)
        CR5RodSegment(
            name="rod_forearm",
            p_start=p_j3,
            p_end=p_j4,
            radius=0.032,
            color=(0, 160, 240),  # Electric blue
        ),
        # Wrist 1 rod (Link 4 -> Link 5)
        CR5RodSegment(
            name="rod_wrist_1",
            p_start=p_j4,
            p_end=p_j5,
            radius=0.026,
            color=(130, 90, 230),  # Violet
        ),
        # Wrist 2 rod (Link 5 -> Link 6)
        CR5RodSegment(
            name="rod_wrist_2",
            p_start=p_j5,
            p_end=p_j6,
            radius=0.024,
            color=(230, 90, 150),  # Magenta
        ),
        # Gripper rod (Link 6 -> TCP)
        CR5RodSegment(
            name="rod_gripper",
            p_start=p_j6,
            p_end=p_tcp,
            radius=0.018,
            color=(255, 170, 0),  # Industrial Amber / Gold
        ),
    ]
    return rods
