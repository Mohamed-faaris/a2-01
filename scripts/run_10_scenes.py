#!/usr/bin/env python3
"""Run 10 two-camera scenes with varying positions and orientations,
evaluating relative camera pose estimation across all 10 experiments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from two_camera_rel_pose import run_experiment


def main():
    rng = np.random.default_rng(42)
    name = "two_camera_rel_pose"
    exp_root = Path("experiment") / name

    results = []

    # 10 configurations with varying camera baselines, depths, and angles
    # Camera 0 & Camera 1 looking at AprilTag at origin (0, 0, 0)
    for i in range(1, 11):
        exp_id = f"{i:03d}"
        target_dir = exp_root / exp_id

        # Cam0 on the left/varying side: x in [-0.35, -0.12], y in [-0.15, 0.15], z in [0.75, 1.25]
        cam0_x = float(rng.uniform(-0.32, -0.12))
        cam0_y = float(rng.uniform(-0.10, 0.10))
        cam0_z = float(rng.uniform(0.80, 1.20))

        # Cam1 on the right/varying side: x in [0.12, 0.35], y in [-0.15, 0.15], z in [0.75, 1.30]
        cam1_x = float(rng.uniform(0.12, 0.35))
        cam1_y = float(rng.uniform(-0.10, 0.10))
        cam1_z = float(rng.uniform(0.80, 1.30))

        cam0_pos = (cam0_x, cam0_y, cam0_z)
        cam1_pos = (cam1_x, cam1_y, cam1_z)

        print(f"\n==========================================")
        print(f"Running Experiment {exp_id} / 010")
        print(f"Cam0: ({cam0_x:.3f}, {cam0_y:.3f}, {cam0_z:.3f})")
        print(f"Cam1: ({cam1_x:.3f}, {cam1_y:.3f}, {cam1_z:.3f})")
        print(f"==========================================")

        report = run_experiment(
            exp_dir=target_dir,
            cam0_blender_pos=cam0_pos,
            cam1_blender_pos=cam1_pos,
        )

        res = {
            "exp_id": exp_id,
            "passed": report["test_passed"],
            "gt_distance_m": round(report["ground_truth"]["rel_cam0_to_cam1_distance_m"], 4),
            "est_distance_m": round(report["estimated"]["rel_cam0_to_cam1_distance_m"], 4),
            "trans_err_cm": round(report["errors"]["translation_error_cm"], 3),
            "dist_err_cm": round(report["errors"]["distance_error_cm"], 3),
            "rot_err_deg": round(report["errors"]["rotation_error_deg"], 3),
        }
        results.append(res)

    # Save summary
    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    csv_path = exp_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY RESULTS ACROSS 10 EXPERIMENTS")
    print("=" * 70)
    print(f"{'Exp ID':<8} {'Status':<8} {'GT Dist (m)':<13} {'Est Dist (m)':<13} {'Trans Err (cm)':<16} {'Rot Err (deg)':<14}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['exp_id']:<8} {status:<8} {r['gt_distance_m']:<13.3f} {r['est_distance_m']:<13.3f} {r['trans_err_cm']:<16.2f} {r['rot_err_deg']:<14.2f}")
    print("-" * 70)

    all_passed = all(r["passed"] for r in results)
    mean_trans = np.mean([r["trans_err_cm"] for r in results])
    mean_rot = np.mean([r["rot_err_deg"] for r in results])
    print(f"Total: {sum(r['passed'] for r in results)}/10 Passed")
    print(f"Mean Translation Error: {mean_trans:.2f} cm")
    print(f"Mean Rotation Error: {mean_rot:.2f} deg")
    print("=" * 70)


if __name__ == "__main__":
    main()
