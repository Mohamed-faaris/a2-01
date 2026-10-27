#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a JSONL pose trajectory for workpiece box corners from scene.json.

This script reads a hand-eye / multi-camera calibrated `scene.json`
(such as `outputs/dobot-vision-024/scene.json`), computes the 3D poses of the
workpiece box corners / platform tags in robot `base_link` frame, and formats
them into a JSON Lines (`.jsonl`) file compatible with `dobot_move_pose`:

    {"name": "box_corner_01", "pose": [x, y, z, rx, ry, rz]}

Units:
    - x, y, z in millimetres (mm)
    - rx, ry, rz in degrees (deg, intrinsic XYZ / Dobot convention)

Supports Multiple Workpiece Boxes (e.g. 2 boxes = 8 corners total):
    - Box 1 (`box`): Tags [42, 43, 44, 45] -> 4 corners
    - Box 2 (`box2`): Tags [50, 51, 52, 53] -> 4 corners

Supported Generation Modes:
    - `box-corners` (default): Computes the 4 outer oriented bounding corners
      per box platform (8 corners total for 2 boxes).
    - `tag-centers`: Visits the individual tag centers in perimeter order.
    - `tag-corners`: Visits physical corners of each tag.

Default Output:
    output-dobots/pose-above-box/001.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_SCENE_FILE = (
    "/home/mfk01/Projects/econ-systems/demos/a2-01/outputs/"
    "dobot-vision-024/scene.json"
)
DEFAULT_OUT_FILE = "output-dobots/pose-above-box/001.jsonl"


def resolve_output_path(out_arg: str, overwrite: bool = False) -> Path:
    """Resolve output path. If file exists and overwrite is False, auto-increments count (001 -> 002 -> 003)."""
    p = Path(out_arg)

    if p.is_dir() or out_arg.endswith(("/", "\\")):
        target_dir = p
        base_prefix = ""
        digits = 3
        ext = ".jsonl"
        req_file = None
    else:
        target_dir = p.parent
        name = p.name
        m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", name)
        if m:
            base_prefix = m.group(1)
            digits = max(3, len(m.group(2)))
            ext = m.group(3)
        else:
            base_prefix = p.stem + "_"
            digits = 3
            ext = p.suffix or ".jsonl"
        req_file = p

    target_dir.mkdir(parents=True, exist_ok=True)

    if overwrite and req_file is not None:
        return req_file

    if req_file is not None and not req_file.exists():
        return req_file

    # File exists and overwrite is False -> find next available sequence number
    existing_nums = []
    pattern = (
        rf"^{re.escape(base_prefix)}(\d+){re.escape(ext)}$"
        if base_prefix
        else rf"^(\d+){re.escape(ext)}$"
    )
    for item in target_dir.iterdir():
        if item.is_file():
            mat = re.match(pattern, item.name)
            if mat:
                try:
                    existing_nums.append(int(mat.group(1)))
                except ValueError:
                    pass

    next_num = (max(existing_nums) + 1) if existing_nums else 1
    new_filename = f"{base_prefix}{next_num:0{digits}d}{ext}"
    return target_dir / new_filename


def rotation_matrix_to_euler_xyz_deg(R_mat: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to [rx, ry, rz] in degrees (XYZ Euler)."""
    rot = Rotation.from_matrix(R_mat)
    angles = rot.as_euler("xyz", degrees=True)
    return [round(float(v), 3) for v in angles]


def compute_tool_down_orientation(plane_normal: np.ndarray,
                                  primary_axis: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation where tool Z points opposite to table normal."""
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    # Tool Z points downwards into the table (-normal)
    tool_z = -n if n[2] > 0 else n

    # Project primary axis onto the plane perpendicular to tool_z
    p = np.asarray(primary_axis, dtype=np.float64)
    p = p - np.dot(p, tool_z) * tool_z
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-6:
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        p = p - np.dot(p, tool_z) * tool_z
        p_norm = np.linalg.norm(p)
    tool_x = p / p_norm

    # Tool Y completes the right-handed frame
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)

    R_tool = np.column_stack([tool_x, tool_y, tool_z])
    return R_tool


def extract_box_bounding_corners(
    landmarks: Dict[str, Any],
    plane_normal: np.ndarray,
    plane_center_m: np.ndarray,
    box_tags: List[int],
    margin_mm: float = 0.0,
) -> Tuple[List[List[float]], np.ndarray]:
    """Calculate the 4 oriented bounding corners of the workpiece platform (in mm)."""
    all_corners_mm = []
    for tid in box_tags:
        tag_str = str(tid)
        if tag_str in landmarks and "corners_base_m" in landmarks[tag_str]:
            for c in landmarks[tag_str]["corners_base_m"]:
                all_corners_mm.append([v * 1000.0 for v in c])
        elif tag_str in landmarks and "center_base_m" in landmarks[tag_str]:
            all_corners_mm.append([v * 1000.0 for v in landmarks[tag_str]["center_base_m"]])

    if len(all_corners_mm) < 3:
        raise ValueError(
            f"Not enough landmark points found for tags {box_tags} to compute box corners"
        )

    pts = np.array(all_corners_mm, dtype=np.float64)
    c_center = plane_center_m * 1000.0  # mm
    norm = plane_normal / np.linalg.norm(plane_normal)

    # Project onto fitted plane basis
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = u - np.dot(u, norm) * norm
    if np.linalg.norm(u) < 1e-5:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = u - np.dot(u, norm) * norm
    u /= np.linalg.norm(u)
    v = np.cross(norm, u)
    v /= np.linalg.norm(v)

    diffs = pts - c_center
    coords_2d = np.column_stack([np.dot(diffs, u), np.dot(diffs, v)]).astype(np.float32)

    # 2D Minimum Area Bounding Box
    rect = cv2.minAreaRect(coords_2d)
    (rect_center, (width, height), angle) = rect
    if margin_mm > 0:
        width += 2.0 * margin_mm
        height += 2.0 * margin_mm
        rect = (rect_center, (width, height), angle)

    box_2d = cv2.boxPoints(rect)

    # Sort 2D points counter-clockwise starting from bottom-left
    center_2d = np.mean(box_2d, axis=0)
    angles_2d = np.arctan2(box_2d[:, 1] - center_2d[1], box_2d[:, 0] - center_2d[0])
    sorted_idx = np.argsort(angles_2d)
    box_2d_sorted = box_2d[sorted_idx]

    # Reconstruct 3D corners in base_link (mm)
    corners_3d = []
    for pt in box_2d_sorted:
        p3 = c_center + pt[0] * u + pt[1] * v
        corners_3d.append([round(float(x), 3) for x in p3])

    # Primary axis along the longest edge
    edge_vec = np.array(corners_3d[1]) - np.array(corners_3d[0])
    primary_axis = edge_vec / np.linalg.norm(edge_vec)

    return corners_3d, primary_axis


def discover_boxes(scene_data: Dict[str, Any], requested_box: str = "all") -> Dict[str, Dict[str, Any]]:
    """Discover workpiece boxes from scene.json (workpiece_planes or workpiece_plane)."""
    boxes = {}
    landmarks = scene_data.get("static_landmarks", {})

    if "workpiece_planes" in scene_data and scene_data["workpiece_planes"]:
        for name, p in scene_data["workpiece_planes"].items():
            if requested_box != "all" and requested_box != name:
                continue
            boxes[name] = {
                "name": name,
                "normal": np.array(p["normal"], dtype=np.float64),
                "center_m": np.array(p["center_m"], dtype=np.float64),
                "tags": p.get("member_tags") or p.get("tag_ids") or [],
            }
    elif "workpiece_plane" in scene_data and scene_data["workpiece_plane"]:
        p = scene_data["workpiece_plane"]
        boxes["box"] = {
            "name": "box",
            "normal": np.array(p["normal"], dtype=np.float64),
            "center_m": np.array(p["center_m"], dtype=np.float64),
            "tags": p.get("member_tags") or [42, 43, 44, 45],
        }

    # If boxes still empty, fallback by inspecting static_landmarks for known groups
    if not boxes:
        candidate_groups = {
            "box": [42, 43, 44, 45],
            "box2": [50, 51, 52, 53],
        }
        for g_name, tag_list in candidate_groups.items():
            if requested_box != "all" and requested_box != g_name:
                continue
            pts = [
                landmarks[str(tid)]["center_base_m"]
                for tid in tag_list
                if str(tid) in landmarks and "center_base_m" in landmarks[str(tid)]
            ]
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.float64)
                c_m = np.mean(pts_arr, axis=0)
                _, _, vv = np.linalg.svd(pts_arr - c_m)
                norm = vv[-1]
                if norm[2] < 0:
                    norm = -norm
                boxes[g_name] = {
                    "name": g_name,
                    "normal": norm,
                    "center_m": c_m,
                    "tags": [t for t in tag_list if str(t) in landmarks],
                }

    return boxes


def build_box_poses(
    scene_data: Dict[str, Any],
    mode: str = "box-corners",
    box_select: str = "all",
    orientation_mode: str = "normal",
    fixed_rpy: Optional[List[float]] = None,
    approach_lift_mm: float = 0.0,
    z_offset_mm: float = 0.0,
    margin_mm: float = 0.0,
    close_loop: bool = False,
) -> List[Dict[str, Any]]:
    """Build list of pose records formatted for JSONL output."""
    landmarks = scene_data.get("static_landmarks", {})
    boxes = discover_boxes(scene_data, requested_box=box_select)

    if not boxes:
        raise ValueError(f"No workpiece boxes found in scene.json for selection '{box_select}'")

    pose_records: List[Dict[str, Any]] = []

    for b_name, b_info in boxes.items():
        plane_normal = b_info["normal"]
        plane_center_m = b_info["center_m"]
        box_tags = b_info["tags"]

        primary_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        raw_waypoints: List[Tuple[str, List[float], List[float]]] = []

        if mode == "box-corners":
            corners_3d, primary_axis = extract_box_bounding_corners(
                landmarks, plane_normal, plane_center_m, box_tags, margin_mm=margin_mm
            )
            R_normal = compute_tool_down_orientation(plane_normal, primary_axis)
            rpy_normal = rotation_matrix_to_euler_xyz_deg(R_normal)

            for idx, corner in enumerate(corners_3d, 1):
                lbl = f"{b_name}_corner_{idx:02d}"
                raw_waypoints.append((lbl, corner, rpy_normal))

        elif mode == "tag-centers":
            for tid in box_tags:
                tid_str = str(tid)
                if tid_str not in landmarks or "center_base_m" not in landmarks[tid_str]:
                    continue
                c_mm = [round(v * 1000.0, 3) for v in landmarks[tid_str]["center_base_m"]]
                T = np.array(landmarks[tid_str].get("T_base_tag", np.eye(4)))
                if orientation_mode == "tag":
                    rpy = rotation_matrix_to_euler_xyz_deg(T[:3, :3])
                else:
                    R_normal = compute_tool_down_orientation(plane_normal, primary_axis)
                    rpy = rotation_matrix_to_euler_xyz_deg(R_normal)
                raw_waypoints.append((f"{b_name}_tag_{tid}_center", c_mm, rpy))

        elif mode == "tag-corners":
            for tid in box_tags:
                tid_str = str(tid)
                if tid_str not in landmarks or "corners_base_m" not in landmarks[tid_str]:
                    continue
                T = np.array(landmarks[tid_str].get("T_base_tag", np.eye(4)))
                rpy = (
                    rotation_matrix_to_euler_xyz_deg(T[:3, :3])
                    if orientation_mode == "tag"
                    else rotation_matrix_to_euler_xyz_deg(
                        compute_tool_down_orientation(plane_normal, primary_axis)
                    )
                )
                for c_idx, corner in enumerate(landmarks[tid_str]["corners_base_m"], 1):
                    c_mm = [round(v * 1000.0, 3) for v in corner]
                    raw_waypoints.append((f"{b_name}_tag_{tid}_corner_{c_idx:02d}", c_mm, rpy))

        if close_loop and len(raw_waypoints) > 1:
            first_lbl, first_pos, first_rpy = raw_waypoints[0]
            raw_waypoints.append((f"{first_lbl}_return", list(first_pos), list(first_rpy)))

        for lbl, pos, rpy in raw_waypoints:
            final_rpy = fixed_rpy if orientation_mode == "fixed" and fixed_rpy else rpy
            target_pos = [pos[0], pos[1], round(pos[2] + z_offset_mm, 3)]

            if approach_lift_mm > 0:
                approach_pos = [pos[0], pos[1], round(target_pos[2] + approach_lift_mm, 3)]
                pose_records.append({
                    "name": f"{lbl}_approach",
                    "pose": approach_pos + final_rpy,
                })

            pose_records.append({
                "name": lbl,
                "pose": target_pos + final_rpy,
            })

            if approach_lift_mm > 0:
                retract_pos = [pos[0], pos[1], round(target_pos[2] + approach_lift_mm, 3)]
                pose_records.append({
                    "name": f"{lbl}_retract",
                    "pose": retract_pos + final_rpy,
                })

    return pose_records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE_FILE,
        help="path to calibrated scene.json (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_FILE,
        help="output .jsonl file path (default: %(default)s)",
    )
    parser.add_argument(
        "--box",
        default="all",
        help="which box to process: 'all' (both boxes, 8 corners), 'box' (box 1), or 'box2' (box 2) (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["box-corners", "tag-centers", "tag-corners"],
        default="box-corners",
        help="pose generation mode (default: %(default)s)",
    )
    parser.add_argument(
        "--orientation",
        choices=["normal", "tag", "fixed"],
        default="normal",
        help="end-effector tool orientation mode (default: %(default)s)",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=[-180.0, 0.0, 90.0],
        metavar=("RX", "RY", "RZ"),
        help="fixed RPY in degrees when --orientation fixed is used",
    )
    parser.add_argument(
        "--approach-lift-mm",
        type=float,
        default=0.0,
        help="height in mm above each corner for approach/retract waypoints (default: 0 = corner only)",
    )
    parser.add_argument(
        "--z-offset-mm",
        type=float,
        default=0.0,
        help="additional height in mm to offset target points (default: 0)",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=0.0,
        help="margin in mm added outwards to the bounding box corners",
    )
    parser.add_argument(
        "--close-loop",
        action="store_true",
        help="re-visit the first corner at the end to complete the perimeter",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing target file instead of auto-incrementing count (001 -> 002 -> 003)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print JSON Lines to stdout instead of writing to disk",
    )

    args = parser.parse_args(argv)

    scene_path = Path(args.scene)
    if not scene_path.is_file():
        print(f"Error: scene file not found: {scene_path}", file=sys.stderr)
        return 2

    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except Exception as exc:
        print(f"Error: failed to parse {scene_path}: {exc}", file=sys.stderr)
        return 2

    try:
        records = build_box_poses(
            scene_data=scene_data,
            mode=args.mode,
            box_select=args.box,
            orientation_mode=args.orientation,
            fixed_rpy=args.rpy,
            approach_lift_mm=args.approach_lift_mm,
            z_offset_mm=args.z_offset_mm,
            margin_mm=args.margin_mm,
            close_loop=args.close_loop,
        )
    except Exception as exc:
        print(f"Error: failed to build poses: {exc}", file=sys.stderr)
        return 1

    lines = [json.dumps(r) for r in records]
    jsonl_text = "\n".join(lines) + "\n"

    out_path = resolve_output_path(args.out, overwrite=args.overwrite)

    print("=" * 64)
    print("WORKPIECE BOX CORNERS -> DOBOT JSONL GENERATOR")
    print("=" * 64)
    print(f"Input Scene        : {args.scene}")
    print(f"Output JSONL       : {out_path}")
    print(f"Boxes Processed    : {args.box}")
    print(f"Generation Mode    : {args.mode}")
    print(f"Tool Orientation   : {args.orientation}")
    print(f"Approach Lift (mm) : {args.approach_lift_mm}")
    print(f"Z Offset (mm)      : {args.z_offset_mm}")
    print(f"Total Waypoints    : {len(records)}")
    print("=" * 64)

    for i, r in enumerate(records[:10], 1):
        print(f"  [{i:02d}] {r['name']:<20}: pos={r['pose'][:3]} rpy={r['pose'][3:]}")
    if len(records) > 10:
        print(f"  ... and {len(records) - 10} more.")

    if args.dry_run:
        print("\n--- JSONL Output (dry-run) ---")
        sys.stdout.write(jsonl_text)
        return 0

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(jsonl_text)
    except OSError as exc:
        print(f"Error: failed to write {out_path}: {exc}", file=sys.stderr)
        return 2

    print(f"\nSuccessfully wrote {len(records)} poses to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
