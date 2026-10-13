#!/usr/bin/env python3
"""Experiments with cans floating in mid-air viewed by cameras at very different angles!

Camera geometries tested:
  - Wide angular separation (45° to 80° convergence baseline)
  - Highly asymmetric vantage points:
      e.g. Cam0 placed high and steep to the left (pitch ~40°, roll/yaw varied)
      e.g. Cam1 placed low and far to the right or front-right (pitch ~15°, azimuth ~+45°)
  - Cans floating at varied heights (z in [0.15m, 0.45m]) and 3D tilts
  - AprilTag reference marker on the floor
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from multi_method_benchmark_30 import CAN_RADIUS_M, CAN_HEIGHT_M
from can_floating_air_experiment import run_floating_experiment


def main():
    rng = np.random.default_rng(999)
    family_dir = Path("experiment/can_wide_angles")
    reports_dir = family_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO("/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")

    summary_rows = []
    print("\nStarting 10 Wide-Angle Camera Experiments with Floating Cans...")

    for i in range(1, 11):
        exp_id = f"{i:03d}"
        exp_dir = family_dir / exp_id

        marker_size = round(float(rng.uniform(0.075, 0.095)), 3)
        marker_id = int(rng.choice([0, 1, 2, 3, 4]))
        marker_pos = (round(float(rng.uniform(-0.03, 0.03)), 3), round(float(rng.uniform(-0.06, -0.01)), 3), 0.001)

        # 3 Floating Cans in air
        cans = [
            {
                "name": "can_red", "color": [0.85, 0.08, 0.08],
                "position": [round(float(rng.uniform(-0.22, -0.12)), 3),
                             round(float(rng.uniform(0.02, 0.10)), 3),
                             round(float(rng.uniform(0.18, 0.28)), 3)],
                "rotation": [round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.35, 0.35)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
            {
                "name": "can_green", "color": [0.08, 0.85, 0.08],
                "position": [round(float(rng.uniform(-0.05, 0.05)), 3),
                             round(float(rng.uniform(0.08, 0.16)), 3),
                             round(float(rng.uniform(0.28, 0.42)), 3)],
                "rotation": [round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.35, 0.35)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
            {
                "name": "can_blue", "color": [0.08, 0.20, 0.88],
                "position": [round(float(rng.uniform(0.12, 0.22)), 3),
                             round(float(rng.uniform(0.02, 0.10)), 3),
                             round(float(rng.uniform(0.16, 0.30)), 3)],
                "rotation": [round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.25, 0.25)), 3),
                             round(float(rng.uniform(-0.35, 0.35)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
        ]

        # VERY DIFFERENT CAMERA ANGLES & ELEVATIONS:
        # Cam0: Far left (-X), high steep elevation (+Z), closer in Y
        # Azimuth ~ -30° to -45°, Elevation ~ 35° to 50°
        cam0_x = float(rng.uniform(-0.55, -0.40))
        cam0_y = float(rng.uniform(-0.85, -0.65))
        cam0_z = float(rng.uniform(0.50, 0.70))  # High steep angle!
        cam0_pos = (round(cam0_x, 3), round(cam0_y, 3), round(cam0_z, 3))

        # Cam1: Far right (+X), low shallow elevation (+Z), further back in Y
        # Azimuth ~ +25° to +40°, Elevation ~ 15° to 25°
        cam1_x = float(rng.uniform(0.35, 0.55))
        cam1_y = float(rng.uniform(-1.20, -0.95))
        cam1_z = float(rng.uniform(0.20, 0.32))  # Low shallow angle!
        cam1_pos = (round(cam1_x, 3), round(cam1_y, 3), round(cam1_z, 3))

        # Approximate angular baseline convergence between cameras
        v0 = -np.array(cam0_pos)
        v1 = -np.array(cam1_pos)
        angle_deg = math.degrees(math.acos(float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))))

        print(f"[{exp_id}/010] Camera Convergence Angle: {angle_deg:.1f}°")
        print(f"       Cam0 (steep high left): {cam0_pos}")
        print(f"       Cam1 (low shallow right): {cam1_pos}")

        evals = run_floating_experiment(
            exp_dir, yolo_model, marker_size, marker_id, marker_pos, cans, cam0_pos, cam1_pos
        )

        row = {
            "exp_id": exp_id,
            "convergence_angle_deg": round(angle_deg, 1),
            "cam0_pos": cam0_pos,
            "cam1_pos": cam1_pos,
            "yolo_dlt_err_cm": evals["yolo_dlt"]["mean_corner_distance_error_cm"],
            "yolo_midpoint_err_cm": evals["yolo_midpoint"]["mean_corner_distance_error_cm"],
            "contour_dlt_err_cm": evals["contour_dlt"]["mean_corner_distance_error_cm"],
            "contour_midpoint_err_cm": evals["contour_midpoint"]["mean_corner_distance_error_cm"],
        }
        summary_rows.append(row)

    (reports_dir / "wide_angle_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    with open(reports_dir / "wide_angle_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("WIDE ANGLES FLOATING EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Exp ID':<8} {'Angle':<10} {'YOLO DLT (cm)':<16} {'Contour DLT (cm)':<18}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['exp_id']:<8} {r['convergence_angle_deg']:<10.1f}° {r['yolo_dlt_err_cm']:<16.2f} {r['contour_dlt_err_cm']:<18.2f}")
    print("-" * 80)
    y_mean = np.mean([r["yolo_dlt_err_cm"] for r in summary_rows if r["yolo_dlt_err_cm"] is not None])
    c_mean = np.mean([r["contour_dlt_err_cm"] for r in summary_rows if r["contour_dlt_err_cm"] is not None])
    print(f"Overall Mean 3D Error: YOLO = {y_mean:.2f} cm ({y_mean*10:.1f} mm), Contour = {c_mean:.2f} cm ({c_mean*10:.1f} mm)")
    print("=" * 80)


if __name__ == "__main__":
    main()
