#!/usr/bin/env python3
"""Configuration and calibration loader for Hand-Eye-0 Demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default dataset path
DEFAULT_DATASET_DIR = REPO_ROOT / "tmp/eye-hand/008"

# Tag dimensions (edge length in metres) matching markers.yaml & apriltag_tags.yaml
TAG_SIZES_M = {
    0: 0.1240,
    1: 0.1240,
    4: 0.0885,
    5: 0.0885,
    20: 0.0410,
    21: 0.0410,
    40: 0.0660,
    41: 0.0660,
    42: 0.0660,
    43: 0.0660,
    44: 0.0660,
    45: 0.0660,
    50: 0.0700,
    51: 0.0700,
    52: 0.0700,
    53: 0.0700,
    54: 0.0700,
    55: 0.0700,
}
DEFAULT_TAG_SIZE_M = 0.0885

# Tag Roles & Workpiece Box Groups
BASE_LINKED_TAGS = frozenset({0, 1, 4, 5, 41})
BOX_TAG_GROUPS: dict[str, frozenset[int]] = {
    "box": frozenset({42, 43, 44, 45}),
    "box2": frozenset({50, 51, 52, 53}),
}
OBJECT_LINKED_TAGS = frozenset().union(*BOX_TAG_GROUPS.values())
STATIC_TAGS = BASE_LINKED_TAGS | OBJECT_LINKED_TAGS
TOOL_TCP_TAGS = frozenset({20, 21, 40})

PROJECTS_ROOT = REPO_ROOT.parent.parent  # /home/mfk01/Projects/econ-systems

# Default calibration paths
DEFAULT_D415_CALIB = PROJECTS_ROOT / "tools/calibrator/output/rl720/camera_calibration.json"
DEFAULT_USB_CALIB = PROJECTS_ROOT / "tools/calibrator/output/see3cam/camera_calibration.json"


def load_camera_intrinsics(path_or_name: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Loads camera intrinsic matrix K (3x3), dist coeffs D (5,), and image_size (w, h)."""
    p = Path(path_or_name)
    if not p.is_file():
        # Search candidate calibration directories
        candidates = [
            PROJECTS_ROOT / "tools/calibrator/output" / path_or_name / "camera_calibration.json",
            PROJECTS_ROOT / "tools/calibrator/output" / path_or_name / "camera_calibration.npz",
            REPO_ROOT / "calibrations" / path_or_name / "camera_calibration.json",
            REPO_ROOT / "calibrations" / path_or_name / "camera_calibration.npz",
            Path(path_or_name),
        ]
        for c in candidates:
            if c.is_file():
                p = c
                break

    if not p.is_file():
        raise FileNotFoundError(f"Could not find calibration file: {path_or_name}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        c_info = data.get("calibration_info", {})
        w = int(c_info.get("image_width", 1280))
        h = int(c_info.get("image_height", 720))
        K = np.array(data["camera_matrix"]["matrix_3x3"], dtype=np.float64)
        D = np.array(data.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64)
        return K, D, (w, h)

    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=True)
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        D = np.asarray(data["dist_coeffs"], dtype=np.float64)
        size = tuple(int(v) for v in data["image_size"])
        return K, D, size

    if p.suffix.lower() in (".yaml", ".yml"):
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        w = int(data.get("image_width", 1280))
        h = int(data.get("image_height", 720))
        k_data = data.get("camera_matrix", {}).get("data", [])
        K = np.array(k_data, dtype=np.float64).reshape((3, 3)) if len(k_data) == 9 else np.eye(3)
        d_data = data.get("distortion_coefficients", {}).get("data", [])
        D = np.array(d_data, dtype=np.float64)
        return K, D, (w, h)

    raise ValueError(f"Unsupported calibration format: {p}")
