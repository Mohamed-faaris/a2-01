import os
import sys
import json
import csv
import cv2
import numpy as np

# Mock kornia for clean LightGlue & SuperPoint import without heavy native dependencies
import types
kornia = types.ModuleType('kornia')
kornia.__path__ = []
kornia.color = types.ModuleType('kornia.color')
kornia.color.grayscale_to_rgb = lambda x: x
kornia.color.rgb_to_grayscale = lambda x: x
kornia.feature = types.ModuleType('kornia.feature')
kornia.feature.HardNet = None
kornia.feature.LAFDescriptor = None
kornia.feature.laf_from_center_scale_ori = None
sys.modules['kornia'] = kornia
sys.modules['kornia.color'] = kornia.color
sys.modules['kornia.feature'] = kornia.feature

import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

BASE_DIR = "/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/13_box_feature_point_matching"
SHARED_DIR = os.path.join(BASE_DIR, "shared")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# All 25 scenes
SCENES = [f"{i:03d}" for i in range(1, 26)]
METHODS = ["sift_homography", "orb_homography", "superpoint_lightglue", "template_patch_matching", "fundamental_matrix_guided"]

# Initialize deep learning models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sp_extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
lg_matcher = LightGlue(features='superpoint').eval().to(device)

def evaluate_superpoint_lightglue(cam0_img, cam1_img, gt_points):
    """Deep feature extraction with SuperPoint and transformer attention matching with LightGlue"""
    # Convert BGR to PyTorch RGB tensors [1, 3, H, W] normalized to [0, 1]
    img0_t = torch.from_numpy(cam0_img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img1_t = torch.from_numpy(cam1_img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # Convert to grayscale for SuperPoint
    gray0_t = (0.299 * img0_t[:, 2:3] + 0.587 * img0_t[:, 1:2] + 0.114 * img0_t[:, 0:1])
    gray1_t = (0.299 * img1_t[:, 2:3] + 0.587 * img1_t[:, 1:2] + 0.114 * img1_t[:, 0:1])

    with torch.no_grad():
        feats0 = sp_extractor({'image': gray0_t})
        feats1 = sp_extractor({'image': gray1_t})
        
        # Check if enough keypoints were extracted in both images
        kpts0 = feats0.get('keypoints', torch.empty(0))
        kpts1 = feats1.get('keypoints', torch.empty(0))
        if kpts0.shape[1] < 4 or kpts1.shape[1] < 4:
            return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_INSUFFICIENT_FEATURES"} for pt in gt_points}
            
        matches01 = lg_matcher({'image0': feats0, 'image1': feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # indices with shape (K, 2)
    mpts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    mpts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

    results = {}
    if len(mpts0) < 4:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_INSUFFICIENT_MATCHES"} for pt in gt_points}

    for plane_name in ["choco_pie", "kelloggs"]:
        plane_targets = [p for p in gt_points if p["plane"] == plane_name]
        target_cam0_coords = np.array([p["cam0_px"] for p in plane_targets])
        min_x, max_x = target_cam0_coords[:, 0].min() - 80, target_cam0_coords[:, 0].max() + 80
        min_y, max_y = target_cam0_coords[:, 1].min() - 80, target_cam0_coords[:, 1].max() + 80

        mask_plane = (mpts0[:, 0] >= min_x) & (mpts0[:, 0] <= max_x) & (mpts0[:, 1] >= min_y) & (mpts0[:, 1] <= max_y)
        plane_pts0 = mpts0[mask_plane]
        plane_pts1 = mpts1[mask_plane]

        if len(plane_pts0) >= 4:
            H, inlier_mask = cv2.findHomography(plane_pts0, plane_pts1, cv2.RANSAC, 3.5)
        else:
            H = None

        for pt in plane_targets:
            p_name = pt["name"]
            gt_cam1 = np.array(pt["cam1_px"])
            p0 = np.array([pt["cam0_px"][0], pt["cam0_px"][1], 1.0])

            if H is not None:
                p1_proj = H @ p0
                if abs(p1_proj[2]) > 1e-6:
                    est_cam1 = np.array([p1_proj[0] / p1_proj[2], p1_proj[1] / p1_proj[2]])
                    err = float(np.linalg.norm(est_cam1 - gt_cam1))
                    status = "PASS" if err < 15.0 else "FAIL_HIGH_ERROR"
                    results[p_name] = {
                        "est_cam1_px": [round(float(est_cam1[0]), 2), round(float(est_cam1[1]), 2)],
                        "error_px": round(err, 2),
                        "status": status,
                        "num_plane_matches": int(len(plane_pts0))
                    }
                    continue
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_HOMOGRAPHY_FAILED"}

    return results

def evaluate_sift_homography(cam0_img, cam1_img, gt_points):
    gray0 = cv2.cvtColor(cam0_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cam1_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_NO_FEATURES"} for pt in gt_points}

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des0, des1, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)

    results = {}
    for plane_name in ["choco_pie", "kelloggs"]:
        plane_targets = [p for p in gt_points if p["plane"] == plane_name]
        target_cam0_coords = np.array([p["cam0_px"] for p in plane_targets])
        min_x, max_x = target_cam0_coords[:, 0].min() - 80, target_cam0_coords[:, 0].max() + 80
        min_y, max_y = target_cam0_coords[:, 1].min() - 80, target_cam0_coords[:, 1].max() + 80

        mask_plane = (pts0[:, 0] >= min_x) & (pts0[:, 0] <= max_x) & (pts0[:, 1] >= min_y) & (pts0[:, 1] <= max_y)
        plane_pts0 = pts0[mask_plane]
        plane_pts1 = pts1[mask_plane]

        if len(plane_pts0) >= 4:
            H, _ = cv2.findHomography(plane_pts0, plane_pts1, cv2.RANSAC, 4.0)
        else:
            H = None

        for pt in plane_targets:
            p_name = pt["name"]
            gt_cam1 = np.array(pt["cam1_px"])
            p0 = np.array([pt["cam0_px"][0], pt["cam0_px"][1], 1.0])

            if H is not None:
                p1_proj = H @ p0
                if abs(p1_proj[2]) > 1e-6:
                    est_cam1 = np.array([p1_proj[0] / p1_proj[2], p1_proj[1] / p1_proj[2]])
                    err = float(np.linalg.norm(est_cam1 - gt_cam1))
                    status = "PASS" if err < 15.0 else "FAIL_HIGH_ERROR"
                    results[p_name] = {
                        "est_cam1_px": [round(float(est_cam1[0]), 2), round(float(est_cam1[1]), 2)],
                        "error_px": round(err, 2),
                        "status": status,
                        "num_plane_matches": int(len(plane_pts0))
                    }
                    continue
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_HOMOGRAPHY_FAILED"}

    return results

def evaluate_orb_homography(cam0_img, cam1_img, gt_points):
    gray0 = cv2.cvtColor(cam0_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cam1_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    kp0, des0 = orb.detectAndCompute(gray0, None)
    kp1, des1 = orb.detectAndCompute(gray1, None)

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_NO_FEATURES"} for pt in gt_points}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des0, des1, k=2)

    good = [m for m, n in matches if m.distance < 0.78 * n.distance]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)

    results = {}
    for plane_name in ["choco_pie", "kelloggs"]:
        plane_targets = [p for p in gt_points if p["plane"] == plane_name]
        target_cam0_coords = np.array([p["cam0_px"] for p in plane_targets])
        min_x, max_x = target_cam0_coords[:, 0].min() - 80, target_cam0_coords[:, 0].max() + 80
        min_y, max_y = target_cam0_coords[:, 1].min() - 80, target_cam0_coords[:, 1].max() + 80

        mask_plane = (pts0[:, 0] >= min_x) & (pts0[:, 0] <= max_x) & (pts0[:, 1] >= min_y) & (pts0[:, 1] <= max_y)
        plane_pts0 = pts0[mask_plane]
        plane_pts1 = pts1[mask_plane]

        if len(plane_pts0) >= 4:
            H, _ = cv2.findHomography(plane_pts0, plane_pts1, cv2.RANSAC, 5.0)
        else:
            H = None

        for pt in plane_targets:
            p_name = pt["name"]
            gt_cam1 = np.array(pt["cam1_px"])
            p0 = np.array([pt["cam0_px"][0], pt["cam0_px"][1], 1.0])

            if H is not None:
                p1_proj = H @ p0
                if abs(p1_proj[2]) > 1e-6:
                    est_cam1 = np.array([p1_proj[0] / p1_proj[2], p1_proj[1] / p1_proj[2]])
                    err = float(np.linalg.norm(est_cam1 - gt_cam1))
                    status = "PASS" if err < 15.0 else "FAIL_HIGH_ERROR"
                    results[p_name] = {
                        "est_cam1_px": [round(float(est_cam1[0]), 2), round(float(est_cam1[1]), 2)],
                        "error_px": round(err, 2),
                        "status": status,
                        "num_plane_matches": int(len(plane_pts0))
                    }
                    continue
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_HOMOGRAPHY_FAILED"}

    return results

def evaluate_template_patch_matching(cam0_img, cam1_img, gt_points):
    gray0 = cv2.cvtColor(cam0_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cam1_img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray0.shape

    results = {}
    patch_radius = 16
    search_radius_x = 90
    search_radius_y = 60

    for pt in gt_points:
        p_name = pt["name"]
        u0, v0 = int(round(pt["cam0_px"][0])), int(round(pt["cam0_px"][1]))
        gt_cam1 = np.array(pt["cam1_px"])

        if u0 - patch_radius < 0 or u0 + patch_radius >= w_img or v0 - patch_radius < 0 or v0 + patch_radius >= h_img:
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_OUT_OF_BOUNDS"}
            continue

        template = gray0[v0 - patch_radius : v0 + patch_radius + 1, u0 - patch_radius : u0 + patch_radius + 1]

        win_x0 = max(0, u0 - search_radius_x)
        win_x1 = min(w_img - 1, u0 + search_radius_x)
        win_y0 = max(0, v0 - search_radius_y)
        win_y1 = min(h_img - 1, v0 + search_radius_y)

        search_roi = gray1[win_y0:win_y1, win_x0:win_x1]
        if search_roi.shape[0] < template.shape[0] or search_roi.shape[1] < template.shape[1]:
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_ROI_SMALL"}
            continue

        match_res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_res)

        est_u1 = win_x0 + max_loc[0] + patch_radius
        est_v1 = win_y0 + max_loc[1] + patch_radius
        est_cam1 = np.array([float(est_u1), float(est_v1)])

        err = float(np.linalg.norm(est_cam1 - gt_cam1))
        status = "PASS" if err < 15.0 else "FAIL_HIGH_ERROR"

        results[p_name] = {
            "est_cam1_px": [round(float(est_cam1[0]), 2), round(float(est_cam1[1]), 2)],
            "error_px": round(err, 2),
            "status": status,
            "ncc_score": round(float(max_val), 3)
        }

    return results

def evaluate_fundamental_matrix_guided(cam0_img, cam1_img, gt_points):
    gray0 = cv2.cvtColor(cam0_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(cam1_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    results = {}
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_NO_FEATURES"} for pt in gt_points}

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des0, des1, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_INSUFFICIENT_MATCHES"} for pt in gt_points}

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)

    F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 2.0, 0.99)
    if F is None:
        return {pt["name"]: {"est_cam1_px": None, "error_px": None, "status": "FAIL_F_ESTIMATION"} for pt in gt_points}

    patch_radius = 16
    for pt in gt_points:
        p_name = pt["name"]
        u0, v0 = pt["cam0_px"]
        gt_cam1 = np.array(pt["cam1_px"])
        p0 = np.array([u0, v0, 1.0])

        l1 = F @ p0
        line_norm = np.sqrt(l1[0]**2 + l1[1]**2)
        if line_norm < 1e-7:
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_DEGENERATE_LINE"}
            continue
        l1 /= line_norm

        epipolar_dist = abs(l1[0] * gt_cam1[0] + l1[1] * gt_cam1[1] + l1[2])

        best_val = -1.0
        best_pt = None
        u0_int, v0_int = int(round(u0)), int(round(v0))

        if (u0_int - patch_radius >= 0 and u0_int + patch_radius < gray0.shape[1] and
            v0_int - patch_radius >= 0 and v0_int + patch_radius < gray0.shape[0]):
            template = gray0[v0_int - patch_radius : v0_int + patch_radius + 1, u0_int - patch_radius : u0_int + patch_radius + 1]

            for sample_u in range(max(patch_radius, u0_int - 100), min(gray1.shape[1] - patch_radius - 1, u0_int + 100), 2):
                if abs(l1[1]) > 1e-4:
                    sample_v = int(round(-(l1[0] * sample_u + l1[2]) / l1[1]))
                else:
                    sample_v = v0_int

                if sample_v - patch_radius >= 0 and sample_v + patch_radius < gray1.shape[0]:
                    roi = gray1[sample_v - patch_radius : sample_v + patch_radius + 1, sample_u - patch_radius : sample_u + patch_radius + 1]
                    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)[0][0]
                    if res > best_val:
                        best_val = res
                        best_pt = np.array([float(sample_u), float(sample_v)])

        if best_pt is not None:
            err = float(np.linalg.norm(best_pt - gt_cam1))
            status = "PASS" if err < 15.0 else "FAIL_HIGH_ERROR"
            results[p_name] = {
                "est_cam1_px": [round(float(best_pt[0]), 2), round(float(best_pt[1]), 2)],
                "error_px": round(err, 2),
                "epipolar_line_residual_px": round(float(epipolar_dist), 2),
                "ncc_score": round(float(best_val), 3),
                "status": status
            }
        else:
            results[p_name] = {"est_cam1_px": None, "error_px": None, "status": "FAIL_EPIPOLAR_SEARCH"}

    return results

def render_annotated_images(cam0_img, cam1_img, gt_points, est_results, out_dir):
    ann0 = cam0_img.copy()
    ann1 = cam1_img.copy()

    for pt in gt_points:
        p_name = pt["name"]
        u0, v0 = int(round(pt["cam0_px"][0])), int(round(pt["cam0_px"][1]))
        u1_gt, v1_gt = int(round(pt["cam1_px"][0])), int(round(pt["cam1_px"][1]))

        cv2.circle(ann0, (u0, v0), 6, (255, 255, 0), 2)
        cv2.line(ann0, (u0 - 8, v0), (u0 + 8, v0), (255, 255, 0), 1)
        cv2.line(ann0, (u0, v0 - 8), (u0, v0 + 8), (255, 255, 0), 1)
        cv2.putText(ann0, p_name.replace("choco_", "C:").replace("kelloggs_", "K:"), (u0 + 8, v0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ann0, p_name.replace("choco_", "C:").replace("kelloggs_", "K:"), (u0 + 8, v0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)

        cv2.circle(ann1, (u1_gt, v1_gt), 6, (0, 255, 0), 2)
        cv2.putText(ann1, "GT", (u1_gt + 8, v1_gt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        res = est_results.get(p_name, {})
        est = res.get("est_cam1_px")
        if est is not None:
            u1_est, v1_est = int(round(est[0])), int(round(est[1]))
            cv2.circle(ann1, (u1_est, v1_est), 5, (255, 0, 255), -1)
            err = res.get("error_px", 0)
            col = (0, 255, 0) if err < 5.0 else ((0, 255, 255) if err < 15.0 else (0, 0, 255))
            cv2.line(ann1, (u1_gt, v1_gt), (u1_est, v1_est), col, 2)
            cv2.putText(ann1, f"err:{err:.1f}px", (u1_est + 8, v1_est + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, "cam0_annotated.png"), ann0)
    cv2.imwrite(os.path.join(out_dir, "cam1_annotated.png"), ann1)

def run_evaluation():
    all_scene_results = []
    csv_rows = []

    for s_id in SCENES:
        scene_dir = os.path.join(BASE_DIR, s_id)
        inp_dir = os.path.join(scene_dir, "inputs")
        gt_file = os.path.join(inp_dir, "ground_truth.json")

        if not os.path.exists(gt_file):
            continue

        with open(gt_file, "r") as f:
            gt_data = json.load(f)

        cam0_path = os.path.join(inp_dir, "cam0_render.png")
        cam1_path = os.path.join(inp_dir, "cam1_render.png")
        cam0_img = cv2.imread(cam0_path)
        cam1_img = cv2.imread(cam1_path)

        gt_points = gt_data["points"]

        for method in METHODS:
            method_out_dir = os.path.join(scene_dir, "outputs", method)
            os.makedirs(method_out_dir, exist_ok=True)

            if method == "superpoint_lightglue":
                est_results = evaluate_superpoint_lightglue(cam0_img, cam1_img, gt_points)
            elif method == "sift_homography":
                est_results = evaluate_sift_homography(cam0_img, cam1_img, gt_points)
            elif method == "orb_homography":
                est_results = evaluate_orb_homography(cam0_img, cam1_img, gt_points)
            elif method == "template_patch_matching":
                est_results = evaluate_template_patch_matching(cam0_img, cam1_img, gt_points)
            elif method == "fundamental_matrix_guided":
                est_results = evaluate_fundamental_matrix_guided(cam0_img, cam1_img, gt_points)

            render_annotated_images(cam0_img, cam1_img, gt_points, est_results, method_out_dir)

            errors = [v["error_px"] for v in est_results.values() if v.get("error_px") is not None]
            passes = [v for v in est_results.values() if v.get("status") == "PASS"]
            pass_rate = (len(passes) / len(gt_points)) * 100.0 if gt_points else 0.0

            mean_err = float(np.mean(errors)) if errors else 999.0
            median_err = float(np.median(errors)) if errors else 999.0
            max_err = float(np.max(errors)) if errors else 999.0

            rep_data = {
                "scene_id": s_id,
                "method": method,
                "description": gt_data["description"],
                "total_points": len(gt_points),
                "matched_points": len(errors),
                "passed_points": len(passes),
                "pass_rate_pct": round(pass_rate, 1),
                "mean_error_px": round(mean_err, 2),
                "median_error_px": round(median_err, 2),
                "max_error_px": round(max_err, 2),
                "point_details": est_results
            }

            with open(os.path.join(method_out_dir, "report.json"), "w") as f:
                json.dump(rep_data, f, indent=2)

            all_scene_results.append(rep_data)

            csv_rows.append({
                "scene_id": s_id,
                "method": method,
                "pass_rate_pct": rep_data["pass_rate_pct"],
                "mean_error_px": rep_data["mean_error_px"],
                "median_error_px": rep_data["median_error_px"],
                "max_error_px": rep_data["max_error_px"],
                "matched_points": rep_data["matched_points"],
                "total_points": rep_data["total_points"]
            })

            print(f"[{s_id}] {method:<26} | Pass: {pass_rate:5.1f}% | Mean Err: {mean_err:5.2f}px")

    with open(os.path.join(REPORTS_DIR, "summary.json"), "w") as f:
        json.dump(all_scene_results, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "method", "pass_rate_pct", "mean_error_px", "median_error_px", "max_error_px", "matched_points", "total_points"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nSummary reports saved to reports/summary.json and reports/summary.csv")

if __name__ == "__main__":
    run_evaluation()
