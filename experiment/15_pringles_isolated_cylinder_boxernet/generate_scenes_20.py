#!/usr/bin/env python3
"""Scene generator for Experiment 15 (20 Scenes):
Isolated Pringles Models on Tabletop with AprilTag Origin and RealSense Depth.

20 Distinct Benchmark Scenarios:
  001-004: Single Cans (Standing & Rolling in different axes & angles)
  005-008: 3 Cans (Mixed Flavors, Upright + Rolling)
  009-012: 5 Cans (All 5 flavors, clustered, spaced, rolling)
  013-016: 7-8 Cans (Varied standoff, angled cameras, multi-roll)
  017-020: 10 Cans Extreme Stress Tests (Dense tabletop layout, grazing angles, severe rolls)
"""

import os
import sys
import json
import math
import subprocess
import argparse
from pathlib import Path
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 925.09
CX = 640.0
CY = 360.0

CAN_RADIUS_M = 0.0318
CAN_HEIGHT_M = 0.2695
TAG_SIZE_M = 0.088
TAG_ID = 1

ASSETS_DIR = Path("/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated")
EXP_ROOT = Path("/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/15_pringles_isolated_cylinder_boxernet")

FLAVOR_FILES = {
    "Original": str(ASSETS_DIR / "pringles_original.glb"),
    "BBQ": str(ASSETS_DIR / "pringles_bbq.glb"),
    "Extra_Hot": str(ASSETS_DIR / "pringles_extra_hot.glb"),
    "French_Onion": str(ASSETS_DIR / "pringles_french_onion.glb"),
    "Paprika": str(ASSETS_DIR / "pringles_paprika.glb"),
}

SCENE_CONFIGS = [
    {
        "id": "001",
        "desc": "Single upright can (Original Red)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.85, 0.45],
        "cam_target": [0.0, 0.15, 0.10],
        "cans": [
            {"name": "can_orig", "flavor": "Original", "state": "standing", "pos": [0.08, 0.22, CAN_HEIGHT_M/2.0], "rot_z_deg": 15.0}
        ]
    },
    {
        "id": "002",
        "desc": "Single rolling can (BBQ Purple, rolling along X axis)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.85, 0.45],
        "cam_target": [0.0, 0.18, 0.05],
        "cans": [
            {"name": "can_bbq", "flavor": "BBQ", "state": "rolling", "pos": [0.05, 0.20, CAN_RADIUS_M], "roll_axis_deg": 0.0}
        ]
    },
    {
        "id": "003",
        "desc": "Single rolling can (French Onion Green, 45 deg roll angle)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.05, -0.88, 0.48],
        "cam_target": [0.0, 0.20, 0.05],
        "cans": [
            {"name": "can_fo", "flavor": "French_Onion", "state": "rolling", "pos": [-0.06, 0.24, CAN_RADIUS_M], "roll_axis_deg": 45.0}
        ]
    },
    {
        "id": "004",
        "desc": "Single upright can (Paprika, severe side standoff)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.25, -0.82, 0.42],
        "cam_target": [0.0, 0.20, 0.12],
        "cans": [
            {"name": "can_paprika", "flavor": "Paprika", "state": "standing", "pos": [-0.10, 0.26, CAN_HEIGHT_M/2.0], "rot_z_deg": -35.0}
        ]
    },
    {
        "id": "005",
        "desc": "3 cans (2 standing, 1 rolling - Original, Paprika, Extra Hot)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.92, 0.50],
        "cam_target": [0.0, 0.25, 0.12],
        "cans": [
            {"name": "can_orig", "flavor": "Original", "state": "standing", "pos": [-0.18, 0.28, CAN_HEIGHT_M/2.0], "rot_z_deg": -10.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "standing", "pos": [0.18, 0.26, CAN_HEIGHT_M/2.0], "rot_z_deg": 40.0},
            {"name": "can_eh", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.02, 0.18, CAN_RADIUS_M], "roll_axis_deg": 15.0}
        ]
    },
    {
        "id": "006",
        "desc": "3 cans (2 standing, 1 rolling perpendicular - BBQ, French Onion, Original)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.10, -0.90, 0.52],
        "cam_target": [0.0, 0.24, 0.10],
        "cans": [
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [-0.15, 0.22, CAN_HEIGHT_M/2.0], "rot_z_deg": 60.0},
            {"name": "can_fo", "flavor": "French_Onion", "state": "standing", "pos": [0.15, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": -30.0},
            {"name": "can_orig", "flavor": "Original", "state": "rolling", "pos": [-0.02, 0.35, CAN_RADIUS_M], "roll_axis_deg": 90.0}
        ]
    },
    {
        "id": "007",
        "desc": "3 cans (1 standing, 2 rolling cross-angles - Extra Hot, Paprika, BBQ)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.12, -0.95, 0.48],
        "cam_target": [0.0, 0.22, 0.08],
        "cans": [
            {"name": "can_eh", "flavor": "Extra_Hot", "state": "standing", "pos": [0.0, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": 20.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "rolling", "pos": [-0.16, 0.18, CAN_RADIUS_M], "roll_axis_deg": -35.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "rolling", "pos": [0.16, 0.20, CAN_RADIUS_M], "roll_axis_deg": 50.0}
        ]
    },
    {
        "id": "008",
        "desc": "4 cans (3 standing, 1 rolling diagonal - All distinct)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.05, -1.00, 0.55],
        "cam_target": [0.0, 0.26, 0.12],
        "cans": [
            {"name": "can_orig", "flavor": "Original", "state": "standing", "pos": [-0.22, 0.26, CAN_HEIGHT_M/2.0], "rot_z_deg": 0.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [-0.05, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": -45.0},
            {"name": "can_fo", "flavor": "French_Onion", "state": "standing", "pos": [0.20, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": 35.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "rolling", "pos": [0.05, 0.18, CAN_RADIUS_M], "roll_axis_deg": -15.0}
        ]
    },
    {
        "id": "009",
        "desc": "5 cans all 5 flavors (3 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -1.05, 0.60],
        "cam_target": [0.0, 0.30, 0.12],
        "cans": [
            {"name": "can_orig", "flavor": "Original", "state": "standing", "pos": [-0.22, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": 0.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [0.0, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": 45.0},
            {"name": "can_fo", "flavor": "French_Onion", "state": "standing", "pos": [0.22, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": -45.0},
            {"name": "can_eh", "flavor": "Extra_Hot", "state": "rolling", "pos": [-0.12, 0.18, CAN_RADIUS_M], "roll_axis_deg": -25.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "rolling", "pos": [0.14, 0.20, CAN_RADIUS_M], "roll_axis_deg": 30.0}
        ]
    },
    {
        "id": "010",
        "desc": "5 cans arc cluster around AprilTag (4 standing, 1 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.10, -1.08, 0.58],
        "cam_target": [0.0, 0.30, 0.12],
        "cans": [
            {"name": "can_orig", "flavor": "Original", "state": "standing", "pos": [-0.25, 0.28, CAN_HEIGHT_M/2.0], "rot_z_deg": 10.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [-0.10, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": -20.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "standing", "pos": [0.10, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": 75.0},
            {"name": "can_fo", "flavor": "French_Onion", "state": "standing", "pos": [0.25, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": -15.0},
            {"name": "can_eh", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.0, 0.18, CAN_RADIUS_M], "roll_axis_deg": 90.0}
        ]
    },
    {
        "id": "011",
        "desc": "6 cans cluster (4 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.12, -1.10, 0.62],
        "cam_target": [0.0, 0.32, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.25, 0.28, CAN_HEIGHT_M/2.0], "rot_z_deg": 10.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.08, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": -20.0},
            {"name": "can_3", "flavor": "Paprika", "state": "standing", "pos": [0.10, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": 75.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.26, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": -15.0},
            {"name": "can_5", "flavor": "Extra_Hot", "state": "rolling", "pos": [-0.15, 0.18, CAN_RADIUS_M], "roll_axis_deg": 10.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [0.15, 0.19, CAN_RADIUS_M], "roll_axis_deg": -60.0}
        ]
    },
    {
        "id": "012",
        "desc": "6 cans staggered (3 standing, 3 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.08, -1.12, 0.64],
        "cam_target": [0.0, 0.34, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.20, 0.36, CAN_HEIGHT_M/2.0], "rot_z_deg": -25.0},
            {"name": "can_2", "flavor": "Original", "state": "standing", "pos": [0.02, 0.46, CAN_HEIGHT_M/2.0], "rot_z_deg": 15.0},
            {"name": "can_3", "flavor": "Paprika", "state": "standing", "pos": [0.22, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": 80.0},
            {"name": "can_4", "flavor": "BBQ", "state": "rolling", "pos": [-0.22, 0.18, CAN_RADIUS_M], "roll_axis_deg": 30.0},
            {"name": "can_5", "flavor": "French_Onion", "state": "rolling", "pos": [0.0, 0.22, CAN_RADIUS_M], "roll_axis_deg": -45.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [0.22, 0.18, CAN_RADIUS_M], "roll_axis_deg": 15.0}
        ]
    },
    {
        "id": "013",
        "desc": "7 cans spread across table (5 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.10, -1.15, 0.65],
        "cam_target": [0.0, 0.35, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.30, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": 0.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.16, 0.45, CAN_HEIGHT_M/2.0], "rot_z_deg": 35.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.0, 0.48, CAN_HEIGHT_M/2.0], "rot_z_deg": -50.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": 20.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": -40.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [-0.18, 0.18, CAN_RADIUS_M], "roll_axis_deg": 40.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [0.16, 0.18, CAN_RADIUS_M], "roll_axis_deg": -30.0}
        ]
    },
    {
        "id": "014",
        "desc": "7 cans dense cluster (4 standing, 3 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.15, -1.18, 0.66],
        "cam_target": [0.0, 0.36, 0.10],
        "cans": [
            {"name": "can_1", "flavor": "French_Onion", "state": "standing", "pos": [-0.26, 0.34, CAN_HEIGHT_M/2.0], "rot_z_deg": 10.0},
            {"name": "can_2", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.08, 0.46, CAN_HEIGHT_M/2.0], "rot_z_deg": -25.0},
            {"name": "can_3", "flavor": "Paprika", "state": "standing", "pos": [0.12, 0.44, CAN_HEIGHT_M/2.0], "rot_z_deg": 60.0},
            {"name": "can_4", "flavor": "Original", "state": "standing", "pos": [0.28, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": -45.0},
            {"name": "can_5", "flavor": "BBQ", "state": "rolling", "pos": [-0.20, 0.18, CAN_RADIUS_M], "roll_axis_deg": 0.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [0.0, 0.24, CAN_RADIUS_M], "roll_axis_deg": 75.0},
            {"name": "can_7", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.20, 0.18, CAN_RADIUS_M], "roll_axis_deg": -45.0}
        ]
    },
    {
        "id": "015",
        "desc": "8 cans (4 standing, 4 rolling across various directions)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -1.20, 0.70],
        "cam_target": [0.0, 0.38, 0.10],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.28, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": 10.0},
            {"name": "can_2", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.10, 0.52, CAN_HEIGHT_M/2.0], "rot_z_deg": -45.0},
            {"name": "can_3", "flavor": "French_Onion", "state": "standing", "pos": [0.10, 0.52, CAN_HEIGHT_M/2.0], "rot_z_deg": 30.0},
            {"name": "can_4", "flavor": "Paprika", "state": "standing", "pos": [0.28, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": -15.0},
            {"name": "can_5", "flavor": "BBQ", "state": "rolling", "pos": [-0.26, 0.20, CAN_RADIUS_M], "roll_axis_deg": 20.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [-0.08, 0.28, CAN_RADIUS_M], "roll_axis_deg": 80.0},
            {"name": "can_7", "flavor": "French_Onion", "state": "rolling", "pos": [0.12, 0.26, CAN_RADIUS_M], "roll_axis_deg": -70.0},
            {"name": "can_8", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.26, 0.18, CAN_RADIUS_M], "roll_axis_deg": -15.0}
        ]
    },
    {
        "id": "016",
        "desc": "8 cans angled camera viewpoint (5 standing, 3 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.25, -1.18, 0.65],
        "cam_target": [0.05, 0.35, 0.10],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.24, 0.32, CAN_HEIGHT_M/2.0], "rot_z_deg": 30.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.12, 0.46, CAN_HEIGHT_M/2.0], "rot_z_deg": -15.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.06, 0.48, CAN_HEIGHT_M/2.0], "rot_z_deg": 50.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.22, 0.44, CAN_HEIGHT_M/2.0], "rot_z_deg": -30.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.30, CAN_HEIGHT_M/2.0], "rot_z_deg": 10.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [-0.16, 0.18, CAN_RADIUS_M], "roll_axis_deg": -40.0},
            {"name": "can_7", "flavor": "Paprika", "state": "rolling", "pos": [0.02, 0.22, CAN_RADIUS_M], "roll_axis_deg": 65.0},
            {"name": "can_8", "flavor": "BBQ", "state": "rolling", "pos": [0.20, 0.18, CAN_RADIUS_M], "roll_axis_deg": 10.0}
        ]
    },
    {
        "id": "017",
        "desc": "9 cans large array (6 standing, 3 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.18, -1.22, 0.72],
        "cam_target": [0.0, 0.38, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.30, 0.36, CAN_HEIGHT_M/2.0], "rot_z_deg": 20.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.14, 0.50, CAN_HEIGHT_M/2.0], "rot_z_deg": -35.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.02, 0.54, CAN_HEIGHT_M/2.0], "rot_z_deg": 70.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.50, CAN_HEIGHT_M/2.0], "rot_z_deg": -20.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": 45.0},
            {"name": "can_6", "flavor": "Original", "state": "standing", "pos": [0.0, 0.36, CAN_HEIGHT_M/2.0], "rot_z_deg": -60.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.22, 0.20, CAN_RADIUS_M], "roll_axis_deg": 25.0},
            {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.02, 0.22, CAN_RADIUS_M], "roll_axis_deg": -30.0},
            {"name": "can_9", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.20, 0.18, CAN_RADIUS_M], "roll_axis_deg": 45.0}
        ]
    },
    {
        "id": "018",
        "desc": "10 cans dense tabletop layout (6 standing, 4 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.05, -1.25, 0.75],
        "cam_target": [0.0, 0.40, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.34, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": 25.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.20, 0.54, CAN_HEIGHT_M/2.0], "rot_z_deg": -10.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.02, 0.58, CAN_HEIGHT_M/2.0], "rot_z_deg": 65.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.54, CAN_HEIGHT_M/2.0], "rot_z_deg": -35.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.34, 0.42, CAN_HEIGHT_M/2.0], "rot_z_deg": 40.0},
            {"name": "can_6", "flavor": "Original", "state": "standing", "pos": [0.0, 0.36, CAN_HEIGHT_M/2.0], "rot_z_deg": -80.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.28, 0.22, CAN_RADIUS_M], "roll_axis_deg": 15.0},
            {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.12, 0.16, CAN_RADIUS_M], "roll_axis_deg": -45.0},
            {"name": "can_9", "flavor": "Paprika", "state": "rolling", "pos": [0.12, 0.16, CAN_RADIUS_M], "roll_axis_deg": 45.0},
            {"name": "can_10", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.28, 0.22, CAN_RADIUS_M], "roll_axis_deg": -20.0}
        ]
    },
    {
        "id": "019",
        "desc": "10 cans extreme grazing angle (camera low at 0.35m elevation)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -1.15, 0.35],
        "cam_target": [0.0, 0.38, 0.08],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.32, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": 15.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.18, 0.50, CAN_HEIGHT_M/2.0], "rot_z_deg": -20.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.0, 0.55, CAN_HEIGHT_M/2.0], "rot_z_deg": 45.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.50, CAN_HEIGHT_M/2.0], "rot_z_deg": -15.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": 30.0},
            {"name": "can_6", "flavor": "Original", "state": "standing", "pos": [0.0, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": -45.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.25, 0.22, CAN_RADIUS_M], "roll_axis_deg": 10.0},
            {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.08, 0.18, CAN_RADIUS_M], "roll_axis_deg": -35.0},
            {"name": "can_9", "flavor": "Paprika", "state": "rolling", "pos": [0.08, 0.18, CAN_RADIUS_M], "roll_axis_deg": 35.0},
            {"name": "can_10", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.25, 0.22, CAN_RADIUS_M], "roll_axis_deg": -15.0}
        ]
    },
    {
        "id": "020",
        "desc": "10 cans severe perspective roll (camera optical roll 15 deg)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.15, -1.20, 0.70],
        "cam_target": [0.0, 0.38, 0.10],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.30, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": -10.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.15, 0.52, CAN_HEIGHT_M/2.0], "rot_z_deg": 25.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.02, 0.56, CAN_HEIGHT_M/2.0], "rot_z_deg": -40.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.52, CAN_HEIGHT_M/2.0], "rot_z_deg": 15.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.40, CAN_HEIGHT_M/2.0], "rot_z_deg": -30.0},
            {"name": "can_6", "flavor": "Original", "state": "standing", "pos": [0.02, 0.38, CAN_HEIGHT_M/2.0], "rot_z_deg": 60.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.24, 0.20, CAN_RADIUS_M], "roll_axis_deg": -20.0},
            {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.06, 0.22, CAN_RADIUS_M], "roll_axis_deg": 50.0},
            {"name": "can_9", "flavor": "Paprika", "state": "rolling", "pos": [0.10, 0.20, CAN_RADIUS_M], "roll_axis_deg": -60.0},
            {"name": "can_10", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.26, 0.20, CAN_RADIUS_M], "roll_axis_deg": 15.0}
        ]
    }
]

def make_blender_scene_script(exp_id: str, cfg: dict, art_dir: Path, inp_dir: Path) -> str:
    escaped_art = str(art_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_inp = str(inp_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    marker_png = str((EXP_ROOT / "shared" / "marker_1.png").resolve()).replace("\\", "\\\\").replace("'", "\\'")
    cfg_json = json.dumps(cfg)
    flavor_files_json = json.dumps(FLAVOR_FILES)

    return f'''import bpy
import math
import os
import json
import mathutils
import numpy as np

art_dir = r'{escaped_art}'
inp_dir = r'{escaped_inp}'
marker_png = r'{marker_png}'
cfg = json.loads(r\'\'\'{cfg_json}\'\'\')
flavor_files = json.loads(r\'\'\'{flavor_files_json}\'\'\')

os.makedirs(art_dir, exist_ok=True)
os.makedirs(inp_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {{item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = {WIDTH}
scene.render.resolution_y = {HEIGHT}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

# Ambient world lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Directional lights
sun1 = bpy.data.lights.new("Sun1", 'SUN')
sun1.energy = 3.5
sun1_obj = bpy.data.objects.new("Sun1", sun1)
scene.collection.objects.link(sun1_obj)
sun1_obj.location = (1.5, -2.0, 3.0)
sun1_obj.rotation_euler = (math.radians(45), math.radians(20), math.radians(35))

sun2 = bpy.data.lights.new("Sun2", 'SUN')
sun2.energy = 1.5
sun2_obj = bpy.data.objects.new("Sun2", sun2)
scene.collection.objects.link(sun2_obj)
sun2_obj.location = (-1.5, -1.0, 2.5)
sun2_obj.rotation_euler = (math.radians(40), math.radians(-25), math.radians(-30))

# Table surface
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
table = bpy.context.active_object
table.name = "TableSurface"
table_mat = bpy.data.materials.new("TableMat")
table_mat.use_nodes = True
table_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.28, 0.30, 1.0)
table_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.45
table.data.materials.append(table_mat)

# AprilTag Marker (tag36h11 id=1, 0.088m size)
tag_size = {TAG_SIZE_M}
hm = tag_size / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

m_img = bpy.data.images.load(marker_png, check_existing=True)
m_img.colorspace_settings.name = "Non-Color"
mat_m = bpy.data.materials.new("MarkerMat")
mat_m.use_nodes = True
m_nodes = mat_m.node_tree.nodes
m_nodes.clear()
tex = m_nodes.new("ShaderNodeTexImage")
tex.image = m_img
tex.interpolation = "Closest"
emission = m_nodes.new("ShaderNodeEmission")
out_node = m_nodes.new("ShaderNodeOutputMaterial")
mat_m.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
mat_m.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
m_pos = cfg["marker_pos"]
marker_obj.location = tuple(m_pos)
marker_obj.data.materials.append(mat_m)

# Import isolated standalone models for each can
can_gt_metadata = []

for c_info in cfg["cans"]:
    flv = c_info["flavor"]
    model_file = flavor_files.get(flv, flavor_files["Original"])
    
    # Import isolated GLB
    bpy.ops.import_scene.gltf(filepath=model_file)
    # The newly imported object is the active/selected one
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    can_obj = imported[0]
    can_obj.name = c_info["name"]

    state = c_info["state"]
    pos = c_info["pos"]

    if state == "standing":
        rz = math.radians(c_info.get("rot_z_deg", 0.0))
        can_obj.location = (pos[0], pos[1], 0.0)
        can_obj.rotation_euler = (0, 0, rz)
        center_world = [pos[0], pos[1], {CAN_HEIGHT_M} / 2.0]
        axis_world = [0.0, 0.0, 1.0]
    elif state == "rolling":
        roll_angle_rad = math.radians(c_info.get("roll_axis_deg", 0.0))
        can_obj.rotation_mode = 'XYZ'
        can_obj.rotation_euler = (0, math.radians(90), roll_angle_rad)
        dx = -({CAN_HEIGHT_M} / 2.0) * math.cos(roll_angle_rad)
        dy = -({CAN_HEIGHT_M} / 2.0) * math.sin(roll_angle_rad)
        can_obj.location = (pos[0] + dx, pos[1] + dy, {CAN_RADIUS_M})
        center_world = [pos[0], pos[1], {CAN_RADIUS_M}]
        axis_world = [float(math.cos(roll_angle_rad)), float(math.sin(roll_angle_rad)), 0.0]

    bpy.context.view_layer.update()
    
    can_gt_metadata.append({{
        "name": c_info["name"],
        "flavor": flv,
        "model_file": model_file,
        "state": state,
        "center_world_m": center_world,
        "axis_world": axis_world,
        "radius_m": {CAN_RADIUS_M},
        "height_m": {CAN_HEIGHT_M},
        "matrix_world": [list(row) for row in can_obj.matrix_world]
    }})

# Setup Camera
cam_pos = cfg["cam_pos"]
cam_target = cfg["cam_target"]
cam_data = bpy.data.cameras.new("Camera_0")
cam_data.lens = {FOCAL_PX} * 36.0 / {WIDTH}
cam_data.sensor_width = 36.0
cam_data.sensor_fit = "HORIZONTAL"
cam_data.clip_start = 0.01
cam_data.clip_end = 50.0

cam_obj = bpy.data.objects.new("Camera_0", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = tuple(cam_pos)
dir_vec = mathutils.Vector(cam_target) - mathutils.Vector(cam_pos)
cam_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
scene.camera = cam_obj

# Compositor for Metric Depth Pass (32-bit float EXR)
scene.view_layers['ViewLayer'].use_pass_z = True
tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
scene.compositing_node_group = tree
rl = tree.nodes.new("CompositorNodeRLayers")

fo = tree.nodes.new("CompositorNodeOutputFile")
fo.directory = inp_dir
fo.file_name = "depth_pass"
fo.file_output_items.clear()
item = fo.file_output_items.new('FLOAT', 'Z')
item.format.file_format = 'OPEN_EXR'
item.format.color_depth = '32'
tree.links.new(rl.outputs['Depth'], fo.inputs['Z'])

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

# Render RGB & Depth
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

# Export ground truth cameras & objects
M_cam = np.array(cam_obj.matrix_world)
R_b = M_cam[:3, :3]
t_b = M_cam[:3, 3]
R_cv2b = np.diag([1.0, -1.0, -1.0])
R_w2c = R_cv2b @ R_b.T
t_w2c = -R_w2c @ t_b

tag_corners_world = [
    [m_pos[0] - hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] + hm, m_pos[2]],
    [m_pos[0] - hm, m_pos[1] + hm, m_pos[2]]
]

gt_doc = {{
    "experiment_id": "{exp_id}",
    "description": cfg["desc"],
    "intrinsics": {{
        "width": {WIDTH},
        "height": {HEIGHT},
        "fx": {FOCAL_PX},
        "fy": {FOCAL_PX},
        "cx": {CX},
        "cy": {CY},
        "camera_matrix": [
            [{FOCAL_PX}, 0.0, {CX}],
            [0.0, {FOCAL_PX}, {CY}],
            [0.0, 0.0, 1.0]
        ]
    }},
    "camera": {{
        "position_world": list(cam_obj.location),
        "rotation_euler": [float(v) for v in cam_obj.rotation_euler],
        "matrix_world": [list(row) for row in cam_obj.matrix_world],
        "R_w2c": [list(row) for row in R_w2c],
        "t_w2c": list(t_w2c)
    }},
    "apriltag": {{
        "tag_id": {TAG_ID},
        "tag_size_m": {TAG_SIZE_M},
        "center_world": m_pos,
        "corners_world": tag_corners_world
    }},
    "cans": can_gt_metadata
}}

with open(os.path.join(inp_dir, "ground_truth.json"), "w") as f:
    json.dump(gt_doc, f, indent=2)

print(f"Scene {exp_id} build and render complete!")
'''

def build_scene(exp_id: str, cfg: dict):
    exp_dir = EXP_ROOT / exp_id
    art_dir = exp_dir / "artifacts"
    inp_dir = exp_dir / "inputs"
    art_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)

    script_path = art_dir / "build_scene.py"
    script_content = make_blender_scene_script(exp_id, cfg, art_dir, inp_dir)
    script_path.write_text(script_content, encoding="utf-8")

    cmd = ["/snap/bin/blender", "-b", "--python", str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error rendering scene {exp_id}:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)

    # Post-process depth pass: extract NPY and 16-bit PNG
    depth_exr = inp_dir / "depth_pass.exr"
    if depth_exr.exists():
        import pyexr
        import cv2
        f = pyexr.open(str(depth_exr))
        z = f.get('Z.V')[:, :, 0]
        z[z > 1e6] = 0.0
        np.save(str(inp_dir / "depth_map.npy"), z.astype(np.float32))
        
        # Save RealSense 16-bit uint (millimeters)
        z_mm = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(inp_dir / "depth_realsense_16bit.png"), z_mm)

        # Depth colormap visualization
        valid = z > 0
        vis = np.zeros_like(z, dtype=np.uint8)
        if valid.any():
            min_d, max_d = z[valid].min(), z[valid].max()
            vis[valid] = ((z[valid] - min_d) / (max_d - min_d + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(inp_dir / "depth_vis.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
        print(f"[{exp_id}] Rendered & extracted metric depth map ({WIDTH}x{HEIGHT}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="all")
    args = parser.parse_args()

    scenes_to_run = SCENE_CONFIGS if args.scene == "all" else [c for c in SCENE_CONFIGS if c["id"] == args.scene]
    for cfg in scenes_to_run:
        print(f"Building scene {cfg['id']}: {cfg['desc']}...")
        build_scene(cfg["id"], cfg)
