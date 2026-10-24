#!/usr/bin/env python3
"""Cross-validation, error analysis, and report generation for Hand-Eye-0 Demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from detector import PoseObservation
from solver import HandEyeCalibrationResult, solve_handeye_pipeline


def run_cross_validation(observations: list[PoseObservation], k_folds: int = 5) -> dict[str, Any]:
    """Runs k-fold cross-validation on poses to assess real held-out generalization."""
    n = len(observations)
    if n < 4:
        return {"status": "skipped", "reason": "not enough poses"}

    indices = list(range(n))
    fold_size = max(1, n // k_folds)
    held_out_pos_errs = []
    held_out_rot_errs = []

    for fold in range(k_folds):
        val_idx = set(indices[fold * fold_size : min((fold + 1) * fold_size, n)])
        train_obs = [observations[i] for i in range(n) if i not in val_idx]
        val_obs = [observations[i] for i in range(n) if i in val_idx]

        if not train_obs or not val_obs:
            continue

        try:
            res_fold = solve_handeye_pipeline(train_obs)
        except Exception:
            continue

        T_b_c = res_fold.T_base_d415
        mounts = res_fold.tool_mounts

        for v in val_obs:
            for tid, m_T in mounts.items():
                if tid in v.d415_tags:
                    T_meas = T_b_c @ v.d415_tags[tid].T_cam_tag
                    T_pred = v.T_base_tcp @ m_T
                    pe = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0)
                    re = float(np.linalg.norm(Rotation.from_matrix(T_pred[:3, :3].T @ T_meas[:3, :3]).as_rotvec()) * 180.0 / np.pi)
                    held_out_pos_errs.append(pe)
                    held_out_rot_errs.append(re)

    if not held_out_pos_errs:
        return {"status": "failed", "reason": "no held-out tool sightings"}

    pos_rms = float(np.sqrt(np.mean(np.array(held_out_pos_errs) ** 2)))
    pos_max = float(np.max(held_out_pos_errs))
    rot_rms = float(np.sqrt(np.mean(np.array(held_out_rot_errs) ** 2)))
    rot_max = float(np.max(held_out_rot_errs))

    return {
        "status": "success",
        "folds": k_folds,
        "held_out_samples": len(held_out_pos_errs),
        "pos_rms_mm": pos_rms,
        "pos_max_mm": pos_max,
        "rot_rms_deg": rot_rms,
        "rot_max_deg": rot_max,
    }


def generate_markdown_report(
    result: HandEyeCalibrationResult,
    cv_stats: dict[str, Any],
    output_path: Path,
) -> str:
    """Generates formatted markdown summary of the entire calibration and verification."""
    t_d415 = result.T_base_d415[:3, 3]
    q_d415 = Rotation.from_matrix(result.T_base_d415[:3, :3]).as_quat()

    lines = [
        "# Hand-Eye Multi-Pose Calibration Report (Demo `hand-eye-0`)",
        "",
        "## 1. Executive Summary",
        f"- **Training Position Residual**: `{result.pos_rms_mm:.2f} mm RMS` (Max: `{result.pos_max_mm:.2f} mm`)",
        f"- **Training Rotation Residual**: `{result.rot_rms_deg:.2f}° RMS` (Max: `{result.rot_max_deg:.2f}°`)",
        f"- **Total Tool Tag Sightings**: `{result.num_tool_sightings}` across 18 arm poses",
    ]

    if cv_stats.get("status") == "success":
        lines += [
            f"- **Held-Out Cross Validation ({cv_stats['folds']}-fold)**: `{cv_stats['pos_rms_mm']:.2f} mm RMS` (Max: `{cv_stats['pos_max_mm']:.2f} mm`), `{cv_stats['rot_rms_deg']:.2f}° RMS`",
        ]

    lines += [
        "",
        "---",
        "",
        "## 2. Solved Camera Extrinsics in `base_link` Frame",
        "",
        "### D415 Camera (`T_base_link -> d415`):",
        f"- **Position $(X, Y, Z)$**: `[{t_d415[0]:+7.3f}, {t_d415[1]:+7.3f}, {t_d415[2]:+7.3f}] m`",
        f"- **Quaternion $(qx, qy, qz, qw)$**: `[{q_d415[0]:+.4f}, {q_d415[1]:+.4f}, {q_d415[2]:+.4f}, {q_d415[3]:+.4f}]`",
        "",
        "```yaml",
        "T_base_d415:",
        f"  xyz: [{t_d415[0]:.6f}, {t_d415[1]:.6f}, {t_d415[2]:.6f}]",
        f"  quat: [{q_d415[0]:.6f}, {q_d415[1]:.6f}, {q_d415[2]:.6f}, {q_d415[3]:.6f}]",
        "```",
    ]

    if result.T_base_usb is not None:
        t_usb = result.T_base_usb[:3, 3]
        q_usb = Rotation.from_matrix(result.T_base_usb[:3, :3]).as_quat()
        lines += [
            "",
            "### USB Camera (`T_base_link -> usb`):",
            f"- **Position $(X, Y, Z)$**: `[{t_usb[0]:+7.3f}, {t_usb[1]:+7.3f}, {t_usb[2]:+7.3f}] m`",
            f"- **Quaternion $(qx, qy, qz, qw)$**: `[{q_usb[0]:+.4f}, {q_usb[1]:+.4f}, {q_usb[2]:+.4f}, {q_usb[3]:+.4f}]`",
            "",
            "```yaml",
            "T_base_usb:",
            f"  xyz: [{t_usb[0]:.6f}, {t_usb[1]:.6f}, {t_usb[2]:.6f}]",
            f"  quat: [{q_usb[0]:.6f}, {q_usb[1]:.6f}, {q_usb[2]:.6f}, {q_usb[3]:.6f}]",
            "```",
        ]

    lines += [
        "",
        "---",
        "",
        "## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)",
        "",
        "| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |",
        "|:---:|:---:|:---|:---:|",
    ]

    for tid, M in sorted(result.tool_mounts.items()):
        t_mm = M[:3, 3] * 1000.0
        dist = np.linalg.norm(t_mm)
        lines.append(f"| **Tag {tid}** | `tcp` | `[{t_mm[0]:+6.1f}, {t_mm[1]:+6.1f}, {t_mm[2]:+6.1f}]` | `{dist:.1f} mm` |")

    lines += [
        "",
        "---",
        "",
        "## 4. Static Tags in `base_link` Reference Frame",
        "",
        "| Tag ID | Role / Link | Position $(X,Y,Z)$ [mm] | Height $Z$ [mm] |",
        "|:---:|:---|:---|:---:|",
    ]

    for tid, T in sorted(result.static_tag_poses.items()):
        p_mm = T[:3, 3] * 1000.0
        role = "Base Cluster (`base_link`)" if tid in (0, 1, 4, 5, 41) else "Workpiece Box (`box`)"
        lines.append(f"| **Tag {tid:2d}** | {role} | `[{p_mm[0]:+7.1f}, {p_mm[1]:+7.1f}, {p_mm[2]:+7.1f}]` | `{p_mm[2]:+7.1f}` |")

    if result.box_plane_normal is not None:
        bn = result.box_plane_normal
        bc = result.box_plane_center * 1000.0
        lines += [
            "",
            "### Workpiece Platform Plane (Tags 42, 43, 44, 45):",
            f"- **Consensus Surface Normal**: `[{bn[0]:+.3f}, {bn[1]:+.3f}, {bn[2]:+.3f}]`",
            f"- **Center Position**: `[{bc[0]:+7.1f}, {bc[1]:+7.1f}, {bc[2]:+7.1f}] mm`",
            f"- **Coplanarity RMSE**: `{result.box_coplanarity_rmse_mm:.2f} mm`",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Per-Pose Sighting Residuals",
        "",
        "| Pose Name | Tag ID | Position Error [mm] | Rotation Error [deg] |",
        "|:---|:---:|:---:|:---:|",
    ]

    for err in result.per_pose_errors:
        lines.append(f"| `{err['pose']}` | Tag {err['tag_id']} | `{err['pos_err_mm']:.2f} mm` | `{err['rot_err_deg']:.2f}°` |")

    report_str = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_str, encoding="utf-8")
    return report_str
