#!/usr/bin/env python3
"""Summarizer and Markdown Report Generator for Experiment 15 (20 Scenes):
Aggregates performance metrics, 3D errors, and uncertainty statistics across all 20 scenes and 5 pipelines.

Outputs:
  - experiment/15_pringles_isolated_cylinder_boxernet/reports/summary.csv
  - experiment/15_pringles_isolated_cylinder_boxernet/reports/summary.json
  - experiment/15_pringles_isolated_cylinder_boxernet/reports/PATTERNS_OBSERVED.md
"""

import json
import csv
from pathlib import Path
import numpy as np

EXP_ROOT = Path("/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/15_pringles_isolated_cylinder_boxernet")
REPORTS_DIR = EXP_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "boxernet_aleatoric_lifting",
    "cylinder_3d_pointcloud_ransac",
    "realsense_surface_depth_compensated",
    "realsense_surface_depth_raw",
    "exp13_feature_matching_transfer"
]

def summarize():
    all_rows = []
    method_stats = {m: {"errors": [], "passes": 0, "total": 0, "sigmas": []} for m in METHODS}

    for i in range(1, 21):
        sid = f"{i:03d}"
        exp_dir = EXP_ROOT / sid
        out_dir = exp_dir / "outputs"

        for m in METHODS:
            rep_path = out_dir / m / "report.json"
            if not rep_path.exists():
                continue
            with open(rep_path) as f:
                rep = json.load(f)

            for ev in rep.get("evaluations", []):
                err = ev["error_3d_cm"]
                passed = ev["pass"]
                method_stats[m]["errors"].append(err)
                method_stats[m]["total"] += 1
                if passed:
                    method_stats[m]["passes"] += 1

                sig = ev.get("aleatoric_sigma", {}) or ev.get("uncertainty", {})

                all_rows.append({
                    "experiment_id": sid,
                    "method": m,
                    "can_name": ev["can_name"],
                    "flavor": ev["flavor"],
                    "state": ev["state"],
                    "error_3d_cm": err,
                    "passed": passed,
                    "sigma_x_cm": sig.get("sigma_x_cm", ""),
                    "sigma_y_cm": sig.get("sigma_y_cm", ""),
                    "sigma_z_cm": sig.get("sigma_z_cm", "")
                })

    # Save summary.csv
    csv_path = REPORTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_id", "method", "can_name", "flavor", "state", "error_3d_cm", "passed",
            "sigma_x_cm", "sigma_y_cm", "sigma_z_cm"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    # Save summary.json
    summary_doc = {}
    for m in METHODS:
        errs = method_stats[m]["errors"]
        tot = method_stats[m]["total"]
        pass_count = method_stats[m]["passes"]
        summary_doc[m] = {
            "total_cans_evaluated": tot,
            "pass_rate_pct": round(pass_count / tot * 100.0, 1) if tot > 0 else 0.0,
            "mean_error_cm": round(float(np.mean(errs)), 2) if errs else 0.0,
            "median_error_cm": round(float(np.median(errs)), 2) if errs else 0.0,
            "p90_error_cm": round(float(np.percentile(errs, 90)), 2) if errs else 0.0,
            "min_error_cm": round(float(np.min(errs)), 2) if errs else 0.0,
            "max_error_cm": round(float(np.max(errs)), 2) if errs else 0.0
        }

    with open(REPORTS_DIR / "summary.json", "w") as f:
        json.dump(summary_doc, f, indent=2)

    # Generate Markdown Report: PATTERNS_OBSERVED.md
    md_content = f"""# PATTERNS OBSERVED: Experiment 15 (Pringles Can 3D Cylinder Construction & BoxerNet Lifting)

## 1. Overview & Setup
Experiment 15 benchmarked 3D cylinder reconstruction, metric localization, and **BoxerNet 2D-to-3D lifting** across **20 distinct scenes** using **strictly isolated, standalone 3D Pringles can models**:
- **Isolated Assets**:
  - `pringles_original.glb`
  - `pringles_bbq.glb`
  - `pringles_extra_hot.glb`
  - `pringles_french_onion.glb`
  - `pringles_paprika.glb`
- **Authentic Dimensions**: Radius $R = 3.18\\text{{ cm}}$, Height $H = 26.95\\text{{ cm}}$.
- **Reference Coordinate Origin**: Tabletop AprilTag #1 ($8.8\\text{{ cm}}$) at $(0, 0, 0)$ via `pupil_apriltags`.
- **Sensor Inputs**: RealSense 1280x720 RGB frame + 32-bit float metric depth map + 16-bit millimeter depth pass.
- **Scenes Span**:
  - **001–004**: Single cans (upright standing, rolling along X, diagonal $45^\\circ$ roll, severe standoff).
  - **005–008**: 3-4 cans (multi-flavor clusters, upright and cross-rolling).
  - **009–012**: 5-6 cans (all 5 flavors, arc clusters around AprilTag, staggered rolls).
  - **013–016**: 7-8 cans (angled camera views, multi-directional rolls).
  - **017–020**: 9-10 cans **extreme stress tests** (dense tabletop layout, ultra-low grazing angle at $0.35\\text{{m}}$ elevation, and severe $15^\\circ$ optical camera roll).

---

## 2. Comparative Benchmark Across All 20 Scenes ({summary_doc['boxernet_aleatoric_lifting']['total_cans_evaluated']} Cans Evaluated)

| Methodology | Model / Mathematical Strategy | Total Evaluated | Pass Rate (<4cm) | Mean 3D Error | Median 3D Error | P90 Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`boxernet_aleatoric_lifting`** | **BoxerNet 2D-to-3D Transformer Lifting + Aleatoric Uncertainty Head** | {summary_doc['boxernet_aleatoric_lifting']['total_cans_evaluated']} | **{summary_doc['boxernet_aleatoric_lifting']['pass_rate_pct']}%** | **{summary_doc['boxernet_aleatoric_lifting']['mean_error_cm']} cm** | **{summary_doc['boxernet_aleatoric_lifting']['median_error_cm']} cm** | **{summary_doc['boxernet_aleatoric_lifting']['p90_error_cm']} cm** |
| **`cylinder_3d_pointcloud_ransac`** | **True 3D Cylinder Mesh Wireframe (Circles + Meridian Ribs) + Table Geometry** | {summary_doc['cylinder_3d_pointcloud_ransac']['total_cans_evaluated']} | **{summary_doc['cylinder_3d_pointcloud_ransac']['pass_rate_pct']}%** | **{summary_doc['cylinder_3d_pointcloud_ransac']['mean_error_cm']} cm** | **{summary_doc['cylinder_3d_pointcloud_ransac']['median_error_cm']} cm** | **{summary_doc['cylinder_3d_pointcloud_ransac']['p90_error_cm']} cm** |
| **`exp13_feature_matching_transfer`** | **Exp-13 Planar Homography Transfer (SIFT / LightGlue) + Metric Depth** | {summary_doc['exp13_feature_matching_transfer']['total_cans_evaluated']} | **{summary_doc['exp13_feature_matching_transfer']['pass_rate_pct']}%** | **{summary_doc['exp13_feature_matching_transfer']['mean_error_cm']} cm** | **{summary_doc['exp13_feature_matching_transfer']['median_error_cm']} cm** | **{summary_doc['exp13_feature_matching_transfer']['p90_error_cm']} cm** |
| **`realsense_surface_depth_compensated`** | **Direct RealSense Surface Depth with Radius Ray Compensation ($+R$)** | {summary_doc['realsense_surface_depth_compensated']['total_cans_evaluated']} | **{summary_doc['realsense_surface_depth_compensated']['pass_rate_pct']}%** | **{summary_doc['realsense_surface_depth_compensated']['mean_error_cm']} cm** | **{summary_doc['realsense_surface_depth_compensated']['median_error_cm']} cm** | **{summary_doc['realsense_surface_depth_compensated']['p90_error_cm']} cm** |
| **`realsense_surface_depth_raw`** | **Uncompensated Front Surface Depth Baseline** | {summary_doc['realsense_surface_depth_raw']['total_cans_evaluated']} | **{summary_doc['realsense_surface_depth_raw']['pass_rate_pct']}%** | **{summary_doc['realsense_surface_depth_raw']['mean_error_cm']} cm** | **{summary_doc['realsense_surface_depth_raw']['median_error_cm']} cm** | **{summary_doc['realsense_surface_depth_raw']['p90_error_cm']} cm** |

---

## 3. Deep Dive into BoxerNet & Aleatoric Uncertainty
1. **Lifting Mechanism**:
   - BoxerNet decouples 2D open-world proposal generation from 3D geometric lifting.
   - Using normalized proposal bounding box $(u, v, w, h)$, aspect ratio, and median patch depth $Z_{{med}}$, the network predicts 3D center translation and 3D extents.
2. **Aleatoric Uncertainty Head**:
   - Outputs log-variances $\\log(\\sigma_x^2, \\sigma_y^2, \\sigma_z^2)$.
   - Objects subject to partial occlusion or extreme camera standoff (Scene 019 & 020) dynamically report increased $\\sigma_z$, providing an explicit confidence gate for downstream robotic grasp planning.
3. **True 3D Cylinder Wireframe vs 3D Box (OBB)**:
   - For rotational symmetry objects like Pringles cans, the 3D circular cylinder model ($16$ meridian segments + circular caps) provides exact collision boundaries compared to standard axis-aligned bounding boxes.
"""

    with open(REPORTS_DIR / "PATTERNS_OBSERVED.md", "w") as f:
        f.write(md_content)

    print("Experiment 15 summary and PATTERNS_OBSERVED.md updated!")

if __name__ == "__main__":
    summarize()
