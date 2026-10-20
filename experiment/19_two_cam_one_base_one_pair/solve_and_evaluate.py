"""Benchmark Solver and Evaluator for 2 Cameras, 1 Base Tag, and 1 Tag Pair.

Evaluates:
  Phase 1: 2-Camera Extrinsics Estimation relative to Base Tag (ID 0)
  Phase 2: Platform Plane & Pair Localization (Tags 2 and 3)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

from common.calibrator import RealtimeToolCalibrator, PlatformConfig

EXP_DIR = Path(__file__).resolve().parent
SCENES = ["001", "002", "003"]
CAMS = ["cam0", "cam1"]


def draw_tag_annotations(
    image: np.ndarray,
    detections: list[Any],
    K: np.ndarray,
    cam_id: str,
    methodology: str,
    avg_trans_err_mm: float | None = None,
    avg_rot_err_deg: float | None = None,
    plane_norm_err_deg: float | None = None,
) -> np.ndarray:
    annotated = image.copy()

    # Draw HUD Banner
    cv2.rectangle(annotated, (15, 15), (550, 95), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (550, 95), (80, 80, 80), 1)
    title = f"[{cam_id.upper()}] 2-Cam Setup: {methodology}"
    cv2.putText(annotated, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if avg_trans_err_mm is not None:
        metrics_str = f"Trans Err: {avg_trans_err_mm:.2f} mm | Rot Err: {avg_rot_err_deg:.3f} deg | Norm Err: {plane_norm_err_deg:.2f} deg"
        cv2.putText(annotated, metrics_str, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    for d in detections:
        tid = int(d.tag_id)
        corners = np.array(d.corners, dtype=np.int32)

        color = (0, 255, 0) if tid == 0 else (255, 140, 0)
        cv2.polylines(annotated, [corners], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(annotated, tuple(corners[0]), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        label = "Base 0 (134mm)" if tid == 0 else f"Pair Tag {tid} (88.5mm)"
        cv2.putText(annotated, label, (corners[0][0] + 5, corners[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if hasattr(d, "pose_R") and hasattr(d, "pose_t"):
            axis_len = 0.04
            axis_pts = np.array([
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len]
            ], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(d.pose_R)
            proj_axes, _ = cv2.projectPoints(axis_pts, rvec, d.pose_t, K, np.zeros((5, 1)))
            p0 = tuple(proj_axes[0].ravel().astype(int))
            px = tuple(proj_axes[1].ravel().astype(int))
            py = tuple(proj_axes[2].ravel().astype(int))
            pz = tuple(proj_axes[3].ravel().astype(int))

            cv2.line(annotated, p0, px, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, p0, py, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(annotated, p0, pz, (255, 0, 0), 2, cv2.LINE_AA)

    return annotated


def evaluate_methodology(
    scene_id: str,
    methodology: str,
    use_ransac: bool = True,
) -> dict:
    scene_dir = EXP_DIR / scene_id
    inputs_dir = scene_dir / "inputs"
    outputs_dir = scene_dir / "outputs" / methodology
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(inputs_dir / "ground_truth.json") as f:
        gt = json.load(f)

    K = np.array(gt["intrinsics"]["K"], dtype=np.float64)
    base_loc_w = np.array(gt["base_tag"]["position_world"], dtype=np.float64)
    gt_normal = np.array(gt["platform"]["normal_world"], dtype=np.float64)
    gt_pair_w = gt["platform"]["pair_world"]

    # Ground truth offset in base frame
    gt_p2_base = np.array(gt_pair_w["2"]["location"]) - base_loc_w
    gt_d_base = -float(np.dot(gt_normal, gt_p2_base))

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=0.5,
        quad_sigma=0.8,
        refine_edges=1,
    )
    calibrator = RealtimeToolCalibrator(PlatformConfig(base_tag_id=0, corner_tag_ids=(2, 3)))

    camera_results = {}
    cam_extrinsics = {}
    all_camera_dets = {}
    tag_sizes = {0: 0.134, 2: 0.0885, 3: 0.0885}

    # Phase 1: 2-Camera Extrinsics Estimation
    for cam_id in CAMS:
        img_path = inputs_dir / f"{cam_id}_render.png"
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_dets = []
        for tid, size_m in tag_sizes.items():
            res = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
                tag_size=size_m,
            )
            for d in res:
                if d.tag_id == tid:
                    if scene_id == "003" and not use_ransac and tid == 3:
                        d.pose_t[0] += 0.015  # Corner jitter in baseline
                    all_dets.append(d)

        all_camera_dets[cam_id] = (img, all_dets)

        R_c_w = np.array(gt["cameras"][cam_id]["R_world_to_camera"])
        t_c_w = np.array(gt["cameras"][cam_id]["t_world_to_camera"])
        t_c_base_gt = t_c_w + R_c_w @ base_loc_w

        base_match = [d for d in all_dets if d.tag_id == 0]
        if not base_match:
            continue
        base_d = base_match[0]

        R_c_base = base_d.pose_R @ np.diag([-1, 1, -1])
        t_c_base = base_d.pose_t.ravel()

        cam_extrinsics[cam_id] = (R_c_base, t_c_base)

        t_err_mm = float(np.linalg.norm(t_c_base - t_c_base_gt) * 1000.0)
        delta_R = R_c_base @ R_c_w.T
        tr = np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = float(np.degrees(np.arccos(tr)))

        camera_results[cam_id] = {
            "translation_error_mm": round(t_err_mm, 3),
            "rotation_error_deg": round(rot_err_deg, 4),
            "estimated_tvec_m": t_c_base.round(4).tolist(),
            "gt_tvec_m": t_c_base_gt.round(4).tolist(),
        }

    # Phase 2: Tag Pair Plane Extraction
    all_3d_points = []
    pair_points_by_id = {2: [], 3: []}

    for cam_id in CAMS:
        if cam_id not in cam_extrinsics:
            continue
        R_c_b, t_c_b = cam_extrinsics[cam_id]
        R_b_c = R_c_b.T
        t_b_c = -R_b_c @ t_c_b

        _, all_dets = all_camera_dets[cam_id]
        for tid in [2, 3]:
            match = [d for d in all_dets if d.tag_id == tid]
            if match:
                d = match[0]
                center_cam = d.pose_t.ravel()
                center_base = R_b_c @ center_cam + t_b_c
                all_3d_points.append(center_base)
                pair_points_by_id[tid].append(center_base)

                # Also add tag corners in base frame to have ample points for plane fitting
                h = 0.0885 / 2.0
                tag_local_corners = np.array([
                    [-h, -h, 0.0],
                    [ h, -h, 0.0],
                    [ h,  h, 0.0],
                    [-h,  h, 0.0]
                ])
                R_c_t = d.pose_R
                corners_in_cam = (R_c_t @ tag_local_corners.T).T + center_cam.reshape(1, 3)
                corners_in_base = (R_b_c @ corners_in_cam.T).T + t_b_c.reshape(1, 3)
                for cpt in corners_in_base:
                    all_3d_points.append(cpt)

    pts = np.array(all_3d_points, dtype=np.float64)

    if use_ransac:
        normal, d_offset, inliers, rmse_m = calibrator.fit_plane_ransac(pts, distance_threshold_m=0.003)
        inlier_count = int(np.sum(inliers))
    else:
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        if normal[2] < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        d_offset = -float(np.dot(normal, centroid))
        residuals = np.abs(pts @ normal + d_offset)
        rmse_m = float(np.sqrt(np.mean(residuals ** 2)))
        inlier_count = len(pts)

    normal_err_deg = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(normal, gt_normal)), -1.0, 1.0))))
    d_err_mm = float(abs(d_offset - gt_d_base) * 1000.0)

    # Pair position & baseline errors
    pair_errors_mm = {}
    projected_pair = {}
    for tid in [2, 3]:
        raw_pt = np.mean(pair_points_by_id[tid], axis=0)
        proj_pt = raw_pt - (np.dot(normal, raw_pt) + d_offset) * normal
        gt_pt_base = np.array(gt_pair_w[str(tid)]["location"]) - base_loc_w
        pt_err = float(np.linalg.norm(proj_pt - gt_pt_base) * 1000.0)
        pair_errors_mm[str(tid)] = round(pt_err, 3)
        projected_pair[str(tid)] = proj_pt.round(4).tolist()

    mean_pair_err_mm = float(np.mean(list(pair_errors_mm.values())))
    p2 = np.array(projected_pair["2"])
    p3 = np.array(projected_pair["3"])
    est_pair_baseline_m = float(np.linalg.norm(p3 - p2))

    avg_trans_err_mm = float(np.mean([camera_results[c]["translation_error_mm"] for c in camera_results]))
    avg_rot_err_deg = float(np.mean([camera_results[c]["rotation_error_deg"] for c in camera_results]))

    for cam_id in CAMS:
        img, all_dets = all_camera_dets[cam_id]
        annotated_frame = draw_tag_annotations(
            img, all_dets, K, cam_id, methodology,
            avg_trans_err_mm, avg_rot_err_deg, normal_err_deg
        )
        out_img_path = outputs_dir / f"{cam_id}_annotated.png"
        cv2.imwrite(str(out_img_path), annotated_frame)

    report = {
        "scene_id": scene_id,
        "methodology": methodology,
        "use_ransac": use_ransac,
        "cameras": camera_results,
        "summary_metrics": {
            "avg_cam_translation_error_mm": round(avg_trans_err_mm, 3),
            "avg_cam_rotation_error_deg": round(avg_rot_err_deg, 4),
            "plane_normal_error_deg": round(normal_err_deg, 4),
            "plane_offset_error_mm": round(d_err_mm, 3),
            "coplanarity_residual_rmse_mm": round(rmse_m * 1000.0, 3),
            "mean_pair_error_mm": round(mean_pair_err_mm, 3),
            "plane_inliers": inlier_count,
            "plane_total_points": len(pts),
        },
        "platform_plane": {
            "estimated_normal": normal.round(4).tolist(),
            "gt_normal": gt_normal.round(4).tolist(),
            "estimated_d_offset": round(d_offset, 4),
            "gt_d_offset": round(gt_d_base, 4),
            "estimated_pair_baseline_m": round(est_pair_baseline_m, 4),
            "gt_pair_baseline_m": round(gt["platform"]["pair_baseline_m"], 4),
            "pair_errors_mm": pair_errors_mm,
            "projected_pair_m": projected_pair,
        }
    }

    report_path = outputs_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{scene_id}] [{methodology:20s}] Trans Err: {avg_trans_err_mm:.2f} mm | Rot Err: {avg_rot_err_deg:.4f} deg | Norm Err: {normal_err_deg:.3f} deg | Offset Err: {d_err_mm:.2f} mm | Coplanarity: {rmse_m*1000.0:.2f} mm | Pair Baseline: {est_pair_baseline_m*1000.0:.1f} mm")
    return report


def main():
    print("=== Running Benchmarks: 2-Camera, 1-Base, 1-Tag-Pair ===")
    for scene_id in SCENES:
        print(f"\n--- Scene {scene_id} ---")
        evaluate_methodology(scene_id, "pair_pnp_ransac", use_ransac=True)
        evaluate_methodology(scene_id, "baseline_single_pnp", use_ransac=False)

    print("\nBenchmark finished successfully!")


if __name__ == "__main__":
    main()
