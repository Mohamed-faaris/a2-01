#!/usr/bin/env python3
"""Evaluate the 2-Camera Planes & Cube Tracker on cube-RESTING-ON-PLANE scenes.

Runs the real demo pipeline (`tracker.TwoCameraPlanesCubePipeline`) on the
renders of experiment/21 and checks, per scene:

1. Plane B (tags 42..45): fitted center/normal vs the GT platform top.
2. Cube centroid (tags 20..25) vs GT.
3. THE CONSTRAINT ("the cube can only be on the plane"):
       est_cube_centroid_z - (est_plane_z + L/2) ~= 0   [gap in mm]
   plus: the cube's (x, y) lies inside the platform quad.

GT world -> tracker Base frame is derived analytically from the tracker's own
convention (base->tag = diag(-1, 1, -1) from solve_camera_extrinsics_from_base),
so estimates and ground truth share one frame:

    p_base = B @ R_tag_world.T @ (p_world - p0_world)

Usage:  PYTHONPATH=src uv run python experiment/21_cube_on_plane_platform/solve_and_evaluate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
DEMO_DIR = REPO_ROOT / "src" / "demos" / "2-cam-planes-cube-tracker"
for _p in (str(REPO_ROOT / "src"), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracker import (  # noqa: E402
    CameraCalibration,
    TwoCameraPlanesCubePipeline,
    detect_and_estimate_poses,
)

CUBE_L = 0.041  # == markers.yaml size of tag 20 == demo CubeConfig side


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_world_to_base(gt: dict):
    """Tracker Base frame from Base Tag 0's GT pose (see module docstring)."""
    b = gt["base_tag"]
    p0 = np.asarray(b["location"], dtype=float)
    rx, ry, rz = b["rotation_euler"]
    R_euler = rot_z(rz) @ rot_y(ry) @ rot_x(rx)          # mesh -> world
    R_tag_world = R_euler @ np.diag([1.0, -1.0, 1.0])     # CV/apriltag frame -> world
    B = np.diag([-1.0, 1.0, -1.0])                        # tracker basis matrix
    M = B @ R_tag_world.T

    def fn(p) -> np.ndarray:
        return M @ (np.asarray(p, dtype=float) - p0)

    return fn


def point_in_poly_2d(p: np.ndarray, poly: np.ndarray) -> bool:
    """Inside-convex-polygon test for the platform quad."""
    hull = cv2.convexHull(poly.astype(np.float32))
    return bool(cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0)


def evaluate_scene(scene_dir: Path) -> dict:
    gt = json.loads((scene_dir / "inputs" / "ground_truth.json").read_text())
    f0 = cv2.imread(str(scene_dir / "inputs" / "cam0_render.png"))
    f1 = cv2.imread(str(scene_dir / "inputs" / "cam1_render.png"))
    assert f0 is not None and f1 is not None

    K = np.asarray(gt["intrinsics"]["K"], dtype=np.float64)
    D = np.zeros(5)
    calib = lambda n: CameraCalibration(name=n, camera_matrix=K, dist_coeffs=D,  # noqa: E731
                                        image_size=(gt["intrinsics"]["width"], gt["intrinsics"]["height"]))

    pipe = TwoCameraPlanesCubePipeline(calib0=calib("cam0"), calib1=calib("cam1"),
                                       source0=str(scene_dir / "inputs" / "cam0_render.png"),
                                       source1=str(scene_dir / "inputs" / "cam1_render.png"),
                                       active_ratio=1.0)
    results = pipe.process_frames(f0, f1)
    w2b = make_world_to_base(gt)

    out_dir = scene_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- annotated renders (3D overlay = est arrays re-projected) ---------
    gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    dets0 = detect_and_estimate_poses(gray0, K, D, pipe.detector0, pipe.registry, 1.0)
    dets1 = detect_and_estimate_poses(gray1, K, D, pipe.detector1, pipe.registry, 1.0)
    ann0 = pipe.annotate_camera_view(f0, dets0, f"{scene_dir.name} cam0", results, 0)
    ann1 = pipe.annotate_camera_view(f1, dets1, f"{scene_dir.name} cam1", results, 1)
    cv2.imwrite(str(out_dir / "cam0_annotated.png"), ann0)
    cv2.imwrite(str(out_dir / "cam1_annotated.png"), ann1)

    report: dict = {
        "scene": scene_dir.name,
        "name": json.loads((scene_dir / "inputs" / "scene_spec.json").read_text())["name"],
        "detected_tags": {"cam0": sorted(dets0), "cam1": sorted(dets1)},
        "cube_resting_gt": gt["constraint"],
    }

    # ---- Ground truth in tracker Base frame -------------------------------
    gt_cube_base = w2b(gt["cube"]["centroid_world"])
    gt_plane_base = w2b(gt["plane_b"]["center_world"])
    gt_corners_base = {int(tid): w2b(p) for tid, p in gt["plane_b"]["corners_world"].items()}
    report["gt_base_frame"] = {
        "cube_centroid_mm": (gt_cube_base * 1000).round(3).tolist(),
        "plane_center_mm": (gt_plane_base * 1000).round(3).tolist(),
    }

    # ---- Plane B estimate --------------------------------------------------
    pb = results.get("plane_b")
    pa = results.get("plane_a")
    est = {}
    if pb:
        est_n = np.asarray(pb["normal"], dtype=float)
        est_c = np.asarray(pb["center_mm"], dtype=float) / 1000.0
        gt_n = np.asarray(gt["plane_b"]["normal_world"], dtype=float)  # == base +Z
        ang = math.degrees(math.acos(float(np.clip(abs(est_n @ gt_n), -1, 1))))
        est["plane"] = {
            "detected_tags": pb["detected_tags"],
            "center_mm": (est_c * 1000).round(3).tolist(),
            "normal": est_n.round(4).tolist(),
            "center_err_mm": (float(np.linalg.norm(est_c - gt_plane_base)) * 1000),
            "z_err_mm": float((est_c[2] - gt_plane_base[2]) * 1000),
            "normal_angle_err_deg": ang,
            "coplanarity_rmse_mm": pb["coplanarity_rmse_mm"],
        }

    # ---- Cube centroid estimate -------------------------------------------
    cc = results.get("cube_centroid")
    gap_mm = None
    if cc and pb:
        est_cube = np.asarray(cc["xyz_mm"], dtype=float) / 1000.0
        est_plane_z = float(np.asarray(pb["center_mm"])[2]) / 1000.0
        gap_mm = float((est_cube[2] - (est_plane_z + CUBE_L / 2.0)) * 1000.0)
        gt_gap_mm = float((gt_cube_base[2] - (gt_plane_base[2] + CUBE_L / 2.0)) * 1000.0)

        # containment: cube (x, y) inside the GT platform quad (base frame)
        poly = np.array([gt_corners_base[t][:2] * 1000.0 for t in gt["plane_b"]["tag_ids"]])
        inside = point_in_poly_2d(est_cube[:2] * 1000.0, poly)

        est["cube"] = {
            "centroid_mm": (est_cube * 1000).round(3).tolist(),
            "visible_faces": cc["visible_faces"],
            "num_obs": cc["num_obs"],
            "residual_spread_mm": cc["residual_spread_mm"],
            "err_mm": float(np.linalg.norm(est_cube - gt_cube_base) * 1000),
            "err_xyz_mm": ((est_cube - gt_cube_base) * 1000).round(3).tolist(),
            "on_plane_gap_mm": gap_mm,             # THE CONSTRAINT (estimated)
            "gt_on_plane_gap_mm": gt_gap_mm,       # should be ~0 by construction
            "xy_inside_platform_quad": inside,
        }
    report["estimates"] = est
    report["plane_a"] = pa  # tags 40,41 absent here by design

    # ---- Pass flags --------------------------------------------------------
    checks = {}
    if est.get("plane"):
        checks["plane_z_within_5mm"] = abs(est["plane"]["z_err_mm"]) < 5.0
        checks["plane_normal_within_5deg"] = est["plane"]["normal_angle_err_deg"] < 5.0
        checks["plane_all_4_tags"] = len(est["plane"]["detected_tags"]) == 4
    if est.get("cube"):
        checks["cube_centroid_within_15mm"] = est["cube"]["err_mm"] < 15.0
        checks["cube_on_plane_gap_within_5mm"] = abs(est["cube"]["on_plane_gap_mm"]) < 5.0
        checks["cube_inside_platform"] = est["cube"]["xy_inside_platform_quad"]
    report["checks"] = checks
    report["all_passed"] = bool(checks) and all(checks.values())

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    pipe.feed0.release()
    pipe.feed1.release()
    return report


def main() -> None:
    scene_dirs = sorted(p for p in EXP_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    reports = []
    for sd in scene_dirs:
        print(f"\n=== {sd.name} ===")
        rep = evaluate_scene(sd)
        reports.append(rep)
        est = rep["estimates"]
        if est.get("plane"):
            p = est["plane"]
            print(f"  plane: center_err={p['center_err_mm']:.2f} mm  z_err={p['z_err_mm']:.2f} mm  "
                  f"normal_err={p['normal_angle_err_deg']:.2f} deg  tags={p['detected_tags']}")
        if est.get("cube"):
            c = est["cube"]
            print(f"  cube : err={c['err_mm']:.2f} mm  xyz_err={c['err_xyz_mm']}  "
                  f"faces={c['visible_faces']} obs={c['num_obs']}")
            print(f"  ON-PLANE CONSTRAINT: gap={c['on_plane_gap_mm']:.2f} mm "
                  f"(GT gap {c['gt_on_plane_gap_mm']:.2f} mm), inside_quad={c['xy_inside_platform_quad']}")
        print(f"  checks: {rep['checks']} -> {'PASS' if rep['all_passed'] else 'FAIL'}")

    summary = {
        "experiment": EXP_DIR.name,
        "scenes": len(reports),
        "all_passed": all(r["all_passed"] for r in reports),
        "mean_cube_err_mm": float(np.mean([r["estimates"]["cube"]["err_mm"]
                                           for r in reports if r["estimates"].get("cube")])),
        "mean_on_plane_gap_mm": float(np.mean([abs(r["estimates"]["cube"]["on_plane_gap_mm"])
                                               for r in reports if r["estimates"].get("cube")])),
        "mean_plane_z_err_mm": float(np.mean([abs(r["estimates"]["plane"]["z_err_mm"])
                                              for r in reports if r["estimates"].get("plane")])),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
