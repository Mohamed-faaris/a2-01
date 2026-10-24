#!/usr/bin/env python3
"""Main CLI for Hand-Eye-0 Demo: Multi-Pose Eye-to-Hand Calibration & Mapping."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# Add demo directory to sys.path
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_D415_CALIB,
    DEFAULT_USB_CALIB,
    load_camera_intrinsics,
)
from detector import load_all_poses
from evaluator import generate_markdown_report, run_cross_validation
from solver import solve_handeye_pipeline
from viz import save_annotated_gallery

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("hand-eye-0")


def result_to_serializable(res) -> dict:
    t_d = res.T_base_d415[:3, 3].tolist()
    q_d = Rotation.from_matrix(res.T_base_d415[:3, :3]).as_quat().tolist()
    data = {
        "calibrated": True,
        "cameras": {
            "d415": {
                "T_base_camera": res.T_base_d415.tolist(),
                "position_m": t_d,
                "quaternion_xyzw": q_d,
            }
        },
        "tool_mounts_in_tcp": {
            str(tid): {
                "T_tcp_tag": M.tolist(),
                "position_mm": (M[:3, 3] * 1000.0).tolist(),
                "quaternion_xyzw": Rotation.from_matrix(M[:3, :3]).as_quat().tolist(),
            }
            for tid, M in res.tool_mounts.items()
        },
        "static_tags_in_base_link": {
            str(tid): {
                "T_base_tag": T.tolist(),
                "position_mm": (T[:3, 3] * 1000.0).tolist(),
                "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
            }
            for tid, T in res.static_tag_poses.items()
        },
        "metrics": {
            "training_pos_rms_mm": res.pos_rms_mm,
            "training_pos_max_mm": res.pos_max_mm,
            "training_rot_rms_deg": res.rot_rms_deg,
            "training_rot_max_deg": res.rot_max_deg,
            "num_tool_sightings": res.num_tool_sightings,
        }
    }

    if res.T_base_usb is not None:
        t_u = res.T_base_usb[:3, 3].tolist()
        q_u = Rotation.from_matrix(res.T_base_usb[:3, :3]).as_quat().tolist()
        data["cameras"]["usb"] = {
            "T_base_camera": res.T_base_usb.tolist(),
            "position_m": t_u,
            "quaternion_xyzw": q_u,
        }

    if res.box_plane_normal is not None:
        data["workpiece_platform"] = {
            "normal": res.box_plane_normal.tolist(),
            "center_mm": (res.box_plane_center * 1000.0).tolist(),
            "coplanarity_rmse_mm": res.box_coplanarity_rmse_mm,
        }

    return data


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye-0 Multi-Pose Calibration & World Mapping")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_DIR),
                        help="Path to pose dataset directory (default: tmp/eye-hand/008)")
    parser.add_argument("--output-dir", type=str, default=str(DEMO_DIR / "outputs"),
                        help="Directory to save output YAML, JSON, report, and annotated images")
    parser.add_argument("--calib-d415", type=str, default=str(DEFAULT_D415_CALIB),
                        help="Path or name of D415 camera calibration")
    parser.add_argument("--calib-usb", type=str, default=str(DEFAULT_USB_CALIB),
                        help="Path or name of USB camera calibration")
    parser.add_argument("--k-folds", type=int, default=5,
                        help="Folds for held-out cross-validation (default: 5)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip generating annotated gallery images")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🤖 STARTING HAND-EYE-0 MULTI-POSE CALIBRATION & WORLD MAPPING")
    print("=" * 80)

    # 1. Load Camera Intrinsics
    log.info(f"Loading D415 calibration: {args.calib_d415}")
    K_d, D_d, size_d = load_camera_intrinsics(args.calib_d415)
    log.info(f"Loading USB calibration:  {args.calib_usb}")
    K_u, D_u, size_u = load_camera_intrinsics(args.calib_usb)

    # 2. Detect Tags across all 18 Poses
    print("\n🔍 Detecting AprilTags across all poses...")
    observations = load_all_poses(args.dataset, K_d, D_d, K_u, D_u)
    if not observations:
        log.error("No valid pose observations found in dataset.")
        sys.exit(1)

    # 3. Solve Hand-Eye Calibration
    print("\n⚙️  Optimizing Hand-Eye Calibration & Mountings...")
    result = solve_handeye_pipeline(observations)

    # 4. Cross-Validation
    print("\n🧪 Running Held-Out Cross-Validation...")
    cv_stats = run_cross_validation(observations, k_folds=args.k_folds)

    # 5. Export JSON and YAML
    dict_payload = result_to_serializable(result)
    dict_payload["cross_validation"] = cv_stats

    json_path = out_root / "handeye_calibration.json"
    yaml_path = out_root / "handeye_calibration.yaml"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict_payload, f, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict_payload, f, sort_keys=False)

    # 6. Generate Markdown Report
    report_path = out_root / "summary_report.md"
    generate_markdown_report(result, cv_stats, report_path)

    # 7. Generate Annotated Gallery Images
    if not args.no_images:
        print("\n🎨 Generating annotated verification images for all poses...")
        save_annotated_gallery(observations, result, K_d, D_d, K_u, D_u, out_root / "annotated")

    # 8. Print Results Summary
    t_d = result.T_base_d415[:3, 3]
    print("\n" + "=" * 80)
    print("🎉 CALIBRATION COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Training Position RMS: {result.pos_rms_mm:.2f} mm  (Max: {result.pos_max_mm:.2f} mm)")
    print(f"  Training Rotation RMS: {result.rot_rms_deg:.2f}°  (Max: {result.rot_max_deg:.2f}°)")
    if cv_stats.get("status") == "success":
        print(f"  Held-Out Position RMS: {cv_stats['pos_rms_mm']:.2f} mm  (Max: {cv_stats['pos_max_mm']:.2f} mm)")
        print(f"  Held-Out Rotation RMS: {cv_stats['rot_rms_deg']:.2f}°")

    print("\n  📍 Camera Extrinsics in base_link:")
    print(f"     D415 Camera: [{t_d[0]:+7.3f}, {t_d[1]:+7.3f}, {t_d[2]:+7.3f}] m")
    if result.T_base_usb is not None:
        t_u = result.T_base_usb[:3, 3]
        print(f"     USB Camera : [{t_u[0]:+7.3f}, {t_u[1]:+7.3f}, {t_u[2]:+7.3f}] m")

    print("\n  🔧 Tool Tag Mountings in tcp Frame:")
    for tid, M in sorted(result.tool_mounts.items()):
        tm = M[:3, 3] * 1000.0
        print(f"     Tag {tid:2d}: [{tm[0]:+6.1f}, {tm[1]:+6.1f}, {tm[2]:+6.1f}] mm (dist: {np.linalg.norm(tm):.1f} mm)")

    print("\n  📌 Static Tags in base_link Frame:")
    for tid, T in sorted(result.static_tag_poses.items()):
        pm = T[:3, 3] * 1000.0
        tag_type = "Base" if tid in (0, 1, 4, 5, 41) else "Box "
        print(f"     Tag {tid:2d} ({tag_type}): [{pm[0]:+7.1f}, {pm[1]:+7.1f}, {pm[2]:+7.1f}] mm")

    if result.box_plane_normal is not None:
        print(f"\n  📐 Workpiece Box Plane (Tags 42..45):")
        print(f"     Normal: [{result.box_plane_normal[0]:+.3f}, {result.box_plane_normal[1]:+.3f}, {result.box_plane_normal[2]:+.3f}] | Coplanar RMSE: {result.box_coplanarity_rmse_mm:.2f} mm")

    print("\n  📁 Output Files Saved:")
    print(f"     ├── {yaml_path}")
    print(f"     ├── {json_path}")
    print(f"     ├── {report_path}")
    if not args.no_images:
        print(f"     └── {out_root / 'annotated'}/ (d415/ and usb/ verification images)")
    print("=" * 80)


if __name__ == "__main__":
    main()
