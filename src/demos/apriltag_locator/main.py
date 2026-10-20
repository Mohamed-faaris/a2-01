"""AprilTag Batch Locator & Annotator.

Processes single images, multiple images, glob patterns, or folders,
detects AprilTags using marker configurations from calibrations/markers.yaml,
optionally uses camera calibration (.npz / .json / .yaml) for solvePnP 3D pose,
annotates the image with tag IDs and detection boxes,
and outputs structured JSON with tag 2D coordinates and 3D poses.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

# Import centralized marker configurations
try:
    from common.markers import (
        MarkerRegistry,
        get_marker_size,
        get_marker_sizes,
        get_marker_metadata,
        get_tag_family,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from common.markers import (
        MarkerRegistry,
        get_marker_size,
        get_marker_sizes,
        get_marker_metadata,
        get_tag_family,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("apriltag_locator")


@contextmanager
def suppress_native_noise():
    """Silence C-level printf chatter from the AprilTag detector."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull)
    except Exception:
        yield


def load_calibration_file(path: Path | str | None) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int] | None, float | None]:
    """Load camera calibration from npz, json, or yaml (optional)."""
    if not path:
        return None, None, None, None

    p = Path(path)
    if not p.is_absolute():
        candidates = [
            Path.cwd() / p,
            Path(__file__).resolve().parent / p,
            Path.cwd() / "calibrations" / p,
            Path.cwd() / "calibrations" / "calibarotor0" / "outputs" / p,
        ]
        for c in candidates:
            if c.exists():
                p = c
                break

    if p.is_dir():
        npz_cand = p / "camera_calibration.npz"
        json_cand = p / "camera_calibration.json"
        yaml_cand = p / "camera_info.yaml"
        if npz_cand.exists():
            p = npz_cand
        elif json_cand.exists():
            p = json_cand
        elif yaml_cand.exists():
            p = yaml_cand

    if not p.exists():
        log.warning(f"Calibration file not found at: {path}")
        return None, None, None, None

    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=True) as data:
            keys = set(data.files)
            camera = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            size = None
            if "image_size" in keys and np.asarray(data["image_size"]).size == 2:
                w, h = (int(v) for v in np.asarray(data["image_size"]).ravel()[:2])
                size = (w, h)
            rms = None
            for key in ("rms", "rms_error", "rms_reprojection_error_px"):
                if key in keys:
                    rms = float(np.asarray(data[key]).ravel()[0])
                    break
            log.info(f"Loaded calibration from NPZ: {p} (rms: {rms})")
            return camera, dist, size, rms

    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            info = json.load(f)
        mat = info.get("camera_matrix", {})
        if isinstance(mat, dict) and "matrix_3x3" in mat:
            camera = np.asarray(mat["matrix_3x3"], dtype=np.float64).reshape(3, 3)
        else:
            fx = float(mat.get("fx", info.get("fx", 0.0)))
            fy = float(mat.get("fy", info.get("fy", 0.0)))
            cx = float(mat.get("cx", info.get("cx", 0.0)))
            cy = float(mat.get("cy", info.get("cy", 0.0)))
            camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray(info.get("distortion_coefficients", info.get("dist_coeffs", [])), dtype=np.float64).reshape(-1, 1)
        cal = info.get("calibration_info", {}) if isinstance(info.get("calibration_info"), dict) else {}
        size = None
        if "image_width" in cal and "image_height" in cal:
            size = (int(cal["image_width"]), int(cal["image_height"]))
        rms = cal.get("rms_reprojection_error_px", info.get("rms_error"))
        log.info(f"Loaded calibration from JSON: {p}")
        return camera, dist, size, (float(rms) if rms is not None else None)

    if suffix in (".yaml", ".yml"):
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)
        camera = np.asarray(info["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(info["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
        size = None
        if "image_width" in info and "image_height" in info:
            size = (int(info["image_width"]), int(info["image_height"]))
        log.info(f"Loaded calibration from YAML: {p}")
        return camera, dist, size, None

    log.warning(f"Unsupported calibration format: {p}")
    return None, None, None, None


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=float) @ np.asarray(second, dtype=float).T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def solvepnp_pose(
    corners: np.ndarray,
    size_m: float,
    camera: np.ndarray,
    distortion: np.ndarray,
    prior_rotation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Robust solvePnP pose disambiguation against 180-deg flip."""
    half = float(size_m) / 2.0
    variants = [
        np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float64),
        np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64),
    ]
    image = np.ascontiguousarray(np.asarray(corners, dtype=np.float64).reshape(4, 2))
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for obj_points in variants:
        for reverse in (False, True):
            points = image[::-1] if reverse else image
            for shift in range(4):
                ordered = np.ascontiguousarray(np.roll(points, shift, axis=0))
                ok, rvec, tvec = cv2.solvePnP(obj_points, ordered, camera, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                rotation = cv2.Rodrigues(rvec)[0]
                score = rotation_error_degrees(rotation, prior_rotation) if prior_rotation is not None else 0.0
                projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera, distortion)
                reproj = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - ordered, axis=1)))
                if best is None or score < best[0]:
                    best = (score, rotation, tvec.reshape(3), reproj)
    if best is None:
        return None
    return best[1], best[2], best[3]


def euler_xyz_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    ry = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-7:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        rz = 0.0
    return tuple(math.degrees(v) for v in (rx, ry, rz))


def annotate_image(
    image: np.ndarray,
    detections_data: list[dict[str, Any]],
    camera: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
    draw_axes: bool = True,
    axis_len_m: float = 0.04,
) -> np.ndarray:
    """Draw bounding boxes, corners, tag ID labels, and optional 3D axes on image."""
    annotated = image.copy()
    for det in detections_data:
        tag_id = det["id"]
        corners = np.asarray(det["corners_px"], dtype=np.float32).reshape(4, 2)
        center = tuple(np.round(np.asarray(det["center_px"])).astype(int))

        # Draw quadrilateral border
        poly = np.round(corners).astype(np.int32)
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 230, 90), thickness=2, lineType=cv2.LINE_AA)

        # Draw top-right corner indicator (corners[0] in pupil_apriltags convention)
        tr = tuple(np.round(corners[0]).astype(int))
        cv2.circle(annotated, tr, 6, (0, 180, 255), -1, cv2.LINE_AA)

        # Draw tag ID label with filled background
        label = f"ID: {tag_id}"
        size_mm = det.get("size_mm")
        if size_mm:
            label += f" ({size_mm:.0f}mm)"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pt_text = (tr[0] + 8, max(20, tr[1] - 4))
        cv2.rectangle(
            annotated,
            (pt_text[0] - 2, pt_text[1] - th - 3),
            (pt_text[0] + tw + 4, pt_text[1] + baseline + 1),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            pt_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

        # Draw 3D axes if pose is available and camera intrinsics exist
        if draw_axes and camera is not None and distortion is not None and "pose" in det:
            pose = det["pose"]
            tvec = np.asarray(pose["translation_cam_xyz_m"], dtype=np.float64).reshape(3, 1)
            rot = np.asarray(pose["rotation_cam_matrix"], dtype=np.float64).reshape(3, 3)
            rvec, _ = cv2.Rodrigues(rot)
            tag_size_m = det.get("size_m", axis_len_m)
            cur_axis = min(axis_len_m, tag_size_m * 0.75)
            cv2.drawFrameAxes(annotated, camera, distortion, rvec, tvec, cur_axis, 2)

    return annotated


def process_image(
    image_path: Path,
    detector: Detector,
    marker_registry: MarkerRegistry,
    camera: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
    tag_family: str = "tag36h11",
    draw_axes: bool = True,
    output_annotated_dir: Path | None = None,
) -> dict[str, Any]:
    """Process a single image, locate all tags, and return dictionary report."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        log.warning(f"Failed to read image: {image_path}")
        return {
            "image_path": str(image_path),
            "status": "error_reading_image",
            "tags": [],
        }

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    intrinsics = None
    if camera is not None:
        intrinsics = (float(camera[0, 0]), float(camera[1, 1]), float(camera[0, 2]), float(camera[1, 2]))

    with suppress_native_noise():
        if intrinsics:
            raw_dets = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=intrinsics,
                tag_size=marker_registry.default_size_m,
            )
        else:
            raw_dets = detector.detect(gray)

    tags_list: list[dict[str, Any]] = []

    for det in raw_dets:
        tag_id = int(det.tag_id)
        tag_size_m = marker_registry.get_size(tag_id)
        tag_meta = marker_registry.metadata.get(tag_id, {})
        corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2).tolist()
        center = [float(det.center[0]), float(det.center[1])]

        tag_record: dict[str, Any] = {
            "id": tag_id,
            "family": tag_meta.get("family", tag_family),
            "size_m": tag_size_m,
            "size_mm": round(tag_size_m * 1000.0, 2),
            "description": tag_meta.get("description", ""),
            "center_px": center,
            "corners_px": corners,
            "decision_margin": float(getattr(det, "decision_margin", 0.0)),
        }

        # 3D pose calculation if camera model is provided
        if camera is not None and distortion is not None and hasattr(det, "pose_R") and det.pose_R is not None:
            init_R = np.asarray(det.pose_R, dtype=np.float64)
            refined = solvepnp_pose(np.asarray(det.corners), tag_size_m, camera, distortion, init_R)
            if refined is not None:
                rot_mat, tvec, reproj_err = refined
                rx, ry, rz = euler_xyz_degrees(rot_mat)
                dist_m = float(np.linalg.norm(tvec))
                tag_record["pose"] = {
                    "translation_cam_xyz_m": [round(float(v), 5) for v in tvec],
                    "translation_cam_xyz_mm": [round(float(v) * 1000.0, 2) for v in tvec],
                    "distance_m": round(dist_m, 5),
                    "distance_mm": round(dist_m * 1000.0, 2),
                    "euler_xyz_deg": [round(rx, 2), round(ry, 2), round(rz, 2)],
                    "rotation_cam_matrix": rot_mat.round(6).tolist(),
                    "reprojection_error_px": round(reproj_err, 3),
                }

        tags_list.append(tag_record)

    # Sort tags by ID
    tags_list.sort(key=lambda t: t["id"])

    # If annotated output requested, create and save image
    annotated_rel_path = None
    if output_annotated_dir is not None:
        annotated = annotate_image(
            img_bgr,
            tags_list,
            camera=camera,
            distortion=distortion,
            draw_axes=draw_axes,
        )
        output_annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_filename = f"{image_path.stem}_annotated{image_path.suffix}"
        annotated_path = output_annotated_dir / annotated_filename
        cv2.imwrite(str(annotated_path), annotated)
        annotated_rel_path = str(annotated_path)

    return {
        "image_path": str(image_path),
        "image_size": {"width": w, "height": h},
        "tags_count": len(tags_list),
        "tags": tags_list,
        "annotated_image": annotated_rel_path,
    }


def collect_image_paths(inputs: list[str]) -> list[Path]:
    """Collect image paths from individual paths, glob patterns, or directories."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    collected: list[Path] = []

    for item in inputs:
        # Check if direct directory
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in extensions:
                    collected.append(f)
            continue

        # Check glob
        matches = glob.glob(item)
        if matches:
            for m in sorted(matches):
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() in extensions:
                    collected.append(mp)
            continue

        if p.is_file() and p.suffix.lower() in extensions:
            collected.append(p)
        else:
            log.warning(f"No valid image found for input: {item}")

    # Remove duplicates preserving order
    seen = set()
    result = []
    for f in collected:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(f)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AprilTag Batch Locator: Detects and annotates AprilTag IDs, 2D corners, and 3D locations into JSON.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["assets"],
        help="Input image files, folders, or glob patterns (e.g. 'images/*.jpg' or 'assets/').",
    )
    parser.add_argument(
        "--markers-yaml",
        type=Path,
        default=None,
        help="Path to markers.yaml configuration file (defaults to calibrations/markers.yaml).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional camera calibration file (.npz, .json, or .yaml).",
    )
    parser.add_argument(
        "--tag-family",
        default="tag36h11",
        help="AprilTag family (default: tag36h11).",
    )
    parser.add_argument(
        "--quad-decimate",
        type=float,
        default=0.5,
        help="Detector quad decimation factor (default: 0.5).",
    )
    parser.add_argument(
        "--quad-sigma",
        type=float,
        default=0.8,
        help="Detector blur sigma (default: 0.8).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("apriltag_locations.json"),
        help="Path for output JSON file (default: apriltag_locations.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("annotated_tags"),
        help="Directory to save annotated images (default: annotated_tags).",
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Disable drawing 3D axes on annotated images.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence non-error progress output.",
    )

    args = parser.parse_args()

    # Load Marker configuration from markers.yaml
    registry = MarkerRegistry(args.markers_yaml)
    family = registry.default_family or args.tag_family
    clean_family = family.replace("tag", "") if family.startswith("tag") else family

    log.info(f"Loaded marker registry with {len(registry.sizes_m)} markers defined.")
    if registry.config_file:
        log.info(f"Using markers configuration: {registry.config_file}")

    # Load optional calibration
    camera, distortion = None, None
    if args.calibration:
        camera, distortion, _, _ = load_calibration_file(args.calibration)
        if camera is not None:
            log.info(f"Using camera calibration for 3D pose extraction: {args.calibration}")
        else:
            log.warning(f"Could not load calibration from {args.calibration}, 3D poses will be skipped.")

    # Gather images
    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        log.error(f"No matching images found for inputs: {args.inputs}")
        return 1

    log.info(f"Found {len(image_paths)} images to process.")

    # Initialize pupil-apriltags detector
    detector = Detector(
        families=f"tag{clean_family}",
        nthreads=4,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    # Process all images
    all_results = []
    total_tags_detected = 0

    for i, img_path in enumerate(image_paths, 1):
        if not args.quiet:
            print(f"[{i}/{len(image_paths)}] Processing {img_path.name} ...", end=" ", flush=True)

        res = process_image(
            img_path,
            detector,
            registry,
            camera=camera,
            distortion=distortion,
            tag_family=family,
            draw_axes=not args.no_axes,
            output_annotated_dir=args.out_dir,
        )
        tags_cnt = res.get("tags_count", 0)
        total_tags_detected += tags_cnt
        all_results.append(res)

        if not args.quiet:
            ids = [t["id"] for t in res.get("tags", [])]
            print(f"-> found {tags_cnt} tags (IDs: {ids})")

    # Build top-level JSON report
    report: dict[str, Any] = {
        "metadata": {
            "markers_yaml": str(registry.config_file) if registry.config_file else None,
            "default_family": family,
            "default_size_m": registry.default_size_m,
            "calibration_applied": str(args.calibration) if camera is not None else None,
            "total_images_processed": len(image_paths),
            "total_tags_detected": total_tags_detected,
        },
        "images": all_results,
    }

    # Write output JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(f"Results successfully saved to JSON: {args.out_json}")
    log.info(f"Annotated images saved to directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
