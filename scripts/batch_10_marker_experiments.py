#!/usr/bin/env python3
"""Run 10 can-relative-to-marker stereo triangulation experiments.
Generates randomized can arrangements, marker positions, and camera views.
Saves inputs, artifacts, outputs, and an aggregated report in reports/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from can_rel_marker_experiment import run_experiment, CAN_RADIUS_M, CAN_HEIGHT_M


def main():
    rng = np.random.default_rng(2026)
    family = "can_rel_to_marker"
    family_dir = Path("experiment") / family
    reports_dir = family_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 10 experiments: 003 to 012
    for exp_num in range(3, 13):
        exp_id = f"{exp_num:03d}"
        exp_dir = family_dir / exp_id
        
        # Vary marker size between 0.06m and 0.10m
        marker_size = round(float(rng.uniform(0.06, 0.10)), 3)
        marker_id = int(rng.choice([0, 1, 2, 3, 4]))

        # Marker position near center/front of scene
        marker_pos = (
            round(float(rng.uniform(-0.05, 0.05)), 3),
            round(float(rng.uniform(-0.08, -0.02)), 3),
            0.001
        )

        # 3 Cans with varied XY locations
        # Red on left, Green in center, Blue on right
        cans = [
            {
                "name": "can_red",
                "color": [0.85, 0.08, 0.08],
                "position": [
                    round(float(rng.uniform(-0.24, -0.12)), 3),
                    round(float(rng.uniform(0.02, 0.12)), 3),
                    0.0625
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M
            },
            {
                "name": "can_green",
                "color": [0.08, 0.85, 0.08],
                "position": [
                    round(float(rng.uniform(-0.06, 0.06)), 3),
                    round(float(rng.uniform(0.08, 0.18)), 3),
                    0.0625
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M
            },
            {
                "name": "can_blue",
                "color": [0.08, 0.20, 0.88],
                "position": [
                    round(float(rng.uniform(0.12, 0.24)), 3),
                    round(float(rng.uniform(0.02, 0.12)), 3),
                    0.0625
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M
            },
        ]

        # Stereo cameras with varied baseline and distance
        cam_dist = float(rng.uniform(0.85, 1.10))
        cam_half_base = float(rng.uniform(0.24, 0.35))
        cam_height = float(rng.uniform(0.28, 0.38))

        cam0_pos = (-cam_half_base, -cam_dist, cam_height)
        cam1_pos = (cam_half_base, -cam_dist, cam_height)

        print(f"\n==========================================")
        print(f"Running Experiment {exp_id} / 012")
        print(f"Marker ID: {marker_id}, Size: {marker_size}m, Pos: {marker_pos}")
        print(f"Cameras baseline: {cam_half_base*2:.2f}m, dist: {cam_dist:.2f}m")
        print(f"==========================================")

        report = run_experiment(
            exp_dir=exp_dir,
            yolo_model_path=Path("/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"),
            marker_size=marker_size,
            marker_id=marker_id,
            marker_pos=marker_pos,
            cans=cans,
            cam0_pos=cam0_pos,
            cam1_pos=cam1_pos,
        )

        row = {
            "exp_id": exp_id,
            "status": report["status"],
            "marker_id": marker_id,
            "marker_size_m": marker_size,
            "cans_triangulated": report["num_cans_triangulated"],
            "mean_corner_err_cm": report["mean_corner_distance_error_cm"],
        }
        for item in report.get("cans_relative_to_marker_corner", []):
            color = item["detected_color"]
            row[f"{color}_dist_err_cm"] = item["corner_distance_error_cm"]
            row[f"{color}_reproj_px"] = item["reprojection_error_px"]

        summary_rows.append(row)

    # Save summary to reports/
    summary_json = reports_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    summary_csv = reports_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print markdown table
    print("\n\n" + "=" * 80)
    print("SUMMARY RESULTS: 10 EXPERIMENTS (003 - 012)")
    print("=" * 80)
    print(f"{'Exp ID':<8} {'Status':<8} {'Marker ID':<10} {'Size (m)':<10} {'Cans':<6} {'Mean Err (cm)':<15}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['exp_id']:<8} {r['status']:<8} {r['marker_id']:<10} {r['marker_size_m']:<10} {r['cans_triangulated']:<6} {r['mean_corner_err_cm']:<15.2f}")
    print("-" * 80)
    overall_mean = np.mean([r["mean_corner_err_cm"] for r in summary_rows if r["mean_corner_err_cm"] is not None])
    pass_count = sum(1 for r in summary_rows if r["status"] == "PASS")
    print(f"Total Passed: {pass_count}/10")
    print(f"Overall Mean Distance to Marker Corner Error: {overall_mean:.2f} cm ({overall_mean*10:.1f} mm)")
    print("=" * 80)


if __name__ == "__main__":
    main()
