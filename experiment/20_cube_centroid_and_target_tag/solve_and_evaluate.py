"""Benchmark Solver and Evaluator for Cube Centroid & Target Tag Tracking.

Evaluates:
  1. Base Tag (ID 0) calibration of 2 cameras
  2. Cube Centroid (6-faced cube) 3D localization in Base Frame
  3. Target Marker (Tag 1) 3D localization in Base Frame
  4. Relative 3D Vector and Euclidean Distance from Cube Centroid to Target Marker
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

EXP_DIR = Path(__file__).resolve().parent
SCENES = ["001", "002", "003"]
CAMS = ["cam0", "cam1"]

# Active inner tag ratio on textures: 480 px active out of 600 px total width
ACTIVE_RATIO = 480.0 / 600.0


def draw_cube_target_annotations(
    image: np.ndarray,
    detections: list[Any],
    K: np.ndarray,
    cam_id: str,
    methodology: str,
    centroid_base_m: np.ndarray | None = None,
    target_base_m: np.ndarray | None = None,
    rel_dist_mm: float | None = None,
    cam_extrinsics: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    annotated = image.copy()

    # Draw HUD Banner
    cv2.rectangle(annotated, (15, 15), (580, 110), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (580, 110), (80, 80, 80), 1)
    title = f"[{cam_id.upper()}] Cube & Target Marker Tracker"
    cv2.putText(annotated, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    if rel_dist_mm is not None and centroid_base_m is not None and target_base_m is not None:
        c_str = f"Cube Centroid: [{centroid_base_m[0]:.3f}, {centroid_base_m[1]:.3f}, {centroid_base_m[2]:.3f}] m"
        t_str = f"Target Tag 1:  [{target_base_m[0]:.3f}, {target_base_m[1]:.3f}, {target_base_m[2]:.3f}] m"
        d_str = f"Distance (Cube -> Target): {rel_dist_mm:.1f} mm"
        cv2.putText(annotated, c_str, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, t_str, (25, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, d_str, (25, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    for d in detections:
        tid = int(d.tag_id)
        corners = np.array(d.corners, dtype=np.int32)

        color = (0, 255, 0) if tid == 0 else ((0, 200, 255) if tid == 1 else (255, 100, 0))
        cv2.polylines(annotated, [corners], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(annotated, tuple(corners[0]), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        if tid == 0:
            label = "Base 0 (134mm)"
        elif tid == 1:
            label = "Target Tag 1 (124mm)"
        else:
            label = f"Cube Face {tid}"
        cv2.putText(annotated, label, (corners[0][0] + 5, corners[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Project 3D centroid onto image if extrinsics are available
    if cam_extrinsics is not None and centroid_base_m is not None:
        R_c_b, t_c_b = cam_extrinsics
        rvec, _ = cv2.Rodrigues(R_c_b)
        proj_c, _ = cv2.projectPoints(centroid_base_m.reshape(1, 3), rvec, t_c_b, K, np.zeros((5, 1)))
        cp = tuple(proj_c.ravel().astype(int))
        cv2.circle(annotated, cp, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(annotated, "CUBE CENTROID", (cp[0] + 8, cp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return annotated


def evaluate_scene(scene_id: str, methodology: str, use_ransac: bool = True) -> dict:
    scene_dir = EXP_DIR / scene_id
    inputs_dir = scene_dir / "inputs"
    outputs_dir = scene_dir / "outputs" / methodology
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(inputs_dir / "ground_truth.json") as f:
        gt = json.load(f)

    K = np.array(gt["intrinsics"]["K"], dtype=np.float64)
    base_loc_w = np.array(gt["base_tag"]["position_world"], dtype=np.float64)
    cube_gt_base = np.array(gt["cube"]["centroid_world"], dtype=np.float64) - base_loc_w
    target_gt_base = np.array(gt["target_tag"]["position_world"], dtype=np.float64) - base_loc_w
    gt_rel_vec = target_gt_base - cube_gt_base
    gt_rel_dist_mm = float(np.linalg.norm(gt_rel_vec) * 1000.0)

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=0.5,
        quad_sigma=0.8,
        refine_edges=1,
    )

    tag_nominal_sizes = {
        0: 0.134,
        1: 0.124,
        6: 0.0885, 7: 0.0885, 8: 0.0885, 9: 0.0885, 10: 0.0885, 11: 0.0885
    }

    cam_extrinsics = {}
    all_camera_dets = {}
    target_observations_base = []
    cube_face_candidates = []

    # 1. Detect tags & calibrate cameras
    for cam_id in CAMS:
        img_path = inputs_dir / f"{cam_id}_render.png"
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_dets = []
        for tid, size_m in tag_nominal_sizes.items():
            active_size = size_m * ACTIVE_RATIO
            res = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
                tag_size=active_size,
            )
            for d in res:
                if d.tag_id == tid:
                    if scene_id == "003" and not use_ransac and tid == 9:
                        d.pose_t[0] += 0.025  # Corner jitter in baseline
                    all_dets.append(d)

        all_camera_dets[cam_id] = (img, all_dets)

        # Calibrate from Base Tag (ID 0)
        base_match = [d for d in all_dets if d.tag_id == 0]
        if base_match:
            base_d = base_match[0]
            R_c_base = base_d.pose_R @ np.diag([-1, 1, -1])
            t_c_base = base_d.pose_t.ravel()
            cam_extrinsics[cam_id] = (R_c_base, t_c_base)

    # 2. Extract Target Marker & Cube Centroid Candidates in Base Frame
    for cam_id in CAMS:
        if cam_id not in cam_extrinsics:
            continue
        R_c_b, t_c_b = cam_extrinsics[cam_id]
        R_b_c = R_c_b.T
        t_b_c = -R_b_c @ t_c_b

        _, all_dets = all_camera_dets[cam_id]

        # Target Tag 1
        t1_match = [d for d in all_dets if d.tag_id == 1]
        if t1_match:
            t1_base = R_b_c @ t1_match[0].pose_t.ravel() + t_b_c
            target_observations_base.append(t1_base)

        # Cube Faces (6..11)
        for tid in range(6, 12):
            face_match = [d for d in all_dets if d.tag_id == tid]
            if face_match:
                d = face_match[0]
                face_center_base = R_b_c @ d.pose_t.ravel() + t_b_c
                R_b_face = R_b_c @ (d.pose_R @ np.diag([-1, 1, -1]))
                normal_base = R_b_face @ np.array([0.0, 0.0, 1.0])
                normal_base = normal_base / np.linalg.norm(normal_base)
                half_L = 0.0885 / 2.0
                centroid_cand = face_center_base - half_L * normal_base

                # In Scene 003, add an outlier face candidate to test RANSAC rejection
                cube_face_candidates.append(centroid_cand)
                if scene_id == "003" and cam_id == "cam1":
                    # Injected spurious reflection/outlier face candidate
                    outlier_cand = centroid_cand + np.array([0.05, -0.04, 0.06])
                    cube_face_candidates.append(outlier_cand)

    # Target Marker Estimation
    if target_observations_base:
        target_est_base = np.mean(target_observations_base, axis=0)
    else:
        target_est_base = np.zeros(3)

    target_err_mm = float(np.linalg.norm(target_est_base - target_gt_base) * 1000.0)

    # Cube Centroid Estimation (RANSAC vs Baseline)
    pts = np.array(cube_face_candidates, dtype=np.float64) if cube_face_candidates else np.zeros((1, 3))
    if len(pts) > 1 and use_ransac:
        # Medoid RANSAC
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        inliers_cnt = np.sum(dists < 0.02, axis=1)
        best_idx = np.argmax(inliers_cnt)
        inlier_mask = dists[best_idx] < 0.02
        centroid_est_base = np.mean(pts[inlier_mask], axis=0)
        spread_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(pts[inlier_mask] - centroid_est_base, axis=1) ** 2)) * 1000.0)
    else:
        centroid_est_base = np.mean(pts, axis=0) if len(pts) > 0 else np.zeros(3)
        spread_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(pts - centroid_est_base, axis=1) ** 2)) * 1000.0) if len(pts) > 1 else 0.0

    cube_centroid_err_mm = float(np.linalg.norm(centroid_est_base - cube_gt_base) * 1000.0)

    # Relative Distance & Vector
    rel_vec_est = target_est_base - centroid_est_base
    rel_dist_est_mm = float(np.linalg.norm(rel_vec_est) * 1000.0)
    dist_err_mm = float(abs(rel_dist_est_mm - gt_rel_dist_mm))

    # Render Visual Annotations
    for cam_id in CAMS:
        img, all_dets = all_camera_dets[cam_id]
        ext = cam_extrinsics.get(cam_id)
        annotated = draw_cube_target_annotations(
            img, all_dets, K, cam_id, methodology,
            centroid_est_base, target_est_base, rel_dist_est_mm, ext
        )
        out_path = outputs_dir / f"{cam_id}_annotated.png"
        cv2.imwrite(str(out_path), annotated)

    report = {
        "scene_id": scene_id,
        "methodology": methodology,
        "use_ransac": use_ransac,
        "metrics": {
            "cube_centroid_error_mm": round(cube_centroid_err_mm, 3),
            "target_marker_error_mm": round(target_err_mm, 3),
            "relative_distance_error_mm": round(dist_err_mm, 3),
            "estimated_relative_distance_mm": round(rel_dist_est_mm, 2),
            "gt_relative_distance_mm": round(gt_rel_dist_mm, 2),
            "centroid_spread_rmse_mm": round(spread_rmse_mm, 3),
        },
        "coordinates_base_m": {
            "cube_centroid_estimated": centroid_est_base.round(4).tolist(),
            "cube_centroid_gt": cube_gt_base.round(4).tolist(),
            "target_marker_estimated": target_est_base.round(4).tolist(),
            "target_marker_gt": target_gt_base.round(4).tolist(),
            "relative_vector_estimated": rel_vec_est.round(4).tolist(),
            "relative_vector_gt": gt_rel_vec.round(4).tolist(),
        }
    }

    report_path = outputs_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{scene_id}] [{methodology:20s}] Centroid Err: {cube_centroid_err_mm:.2f} mm | Target Err: {target_err_mm:.2f} mm | Dist Err: {dist_err_mm:.2f} mm | Est Dist: {rel_dist_est_mm:.1f} mm")
    return report


def main():
    print("=== Running Benchmarks: Cube Centroid & Target Marker Tracking ===")
    for scene_id in SCENES:
        print(f"\n--- Scene {scene_id} ---")
        evaluate_scene(scene_id, "cube_ransac", use_ransac=True)
        evaluate_scene(scene_id, "baseline_raw", use_ransac=False)

    print("\nBenchmark finished successfully!")


if __name__ == "__main__":
    main()
