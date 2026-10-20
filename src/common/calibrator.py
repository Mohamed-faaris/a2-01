"""Real-time 3D Multi-Camera Tool Calibrator & Robust RANSAC Plane Estimator.

Phase 1: Multi-Camera Extrinsics via Joint Multi-Tag PnP RANSAC.
Phase 2: Robust Platform Plane Coordinates via 3D Plane Fitting RANSAC.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from pupil_apriltags import Detector

from common.markers import MarkerRegistry

log = logging.getLogger("tool-calibrator")

# Standard Tag Corners in Tag Local Frame (Z = 0, tag center at (0, 0, 0))
# Order: [-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2] (matching AprilTag corner order)
def get_tag_model_corners_3d(tag_size_m: float) -> np.ndarray:
    h = tag_size_m / 2.0
    return np.array([
        [-h, -h, 0.0],
        [ h, -h, 0.0],
        [ h,  h, 0.0],
        [-h,  h, 0.0]
    ], dtype=np.float64)


@dataclass
class PlatformConfig:
    """Configuration for base frame and platform corner tags."""
    base_tag_id: int = 0
    corner_tag_ids: tuple[int, ...] = (2, 3, 4, 5)
    markers_yaml_path: Path | str | None = None
    tag_family: str = "tag36h11"
    # Custom overrides if provided; otherwise loaded from MarkerRegistry
    tag_sizes_m: dict[int, float] = field(default_factory=dict)
    # Known metric layout on platform plane if available: tag_id -> (X, Y) in meters
    known_platform_layout_m: dict[int, tuple[float, float]] | None = None

    def __post_init__(self):
        registry = MarkerRegistry(self.markers_yaml_path)
        for tid in [self.base_tag_id, *self.corner_tag_ids]:
            if tid not in self.tag_sizes_m:
                self.tag_sizes_m[tid] = registry.get_size(tid)


@dataclass
class TagDetection:
    tag_id: int
    corners_px: np.ndarray      # (4, 2)
    center_px: np.ndarray       # (2,)
    tag_size_m: float
    decision_margin: float = 0.0


@dataclass
class CameraExtrinsics:
    """Camera extrinsics relative to base frame: X_cam = R @ X_base + t."""
    camera_id: str | int
    rvec: np.ndarray            # (3, 1) Rodrigues vector
    tvec: np.ndarray            # (3, 1) Translation in meters
    R: np.ndarray               # (3, 3) Rotation matrix
    reprojection_error_px: float
    num_inliers: int
    total_points: int
    inlier_ratio: float
    # Inverse transform: X_base = R_inv @ X_cam + t_inv (Camera pose in base frame)
    pos_in_base_m: np.ndarray   # (3,)
    euler_xyz_deg: np.ndarray   # (3,)


@dataclass
class PlaneCoordinates:
    """Robust 3D platform plane estimation in Base Reference Frame."""
    normal: np.ndarray          # (3,) Unit normal vector [nx, ny, nz] (nz > 0)
    d_offset: float             # Plane equation: normal . X + d = 0
    origin_m: np.ndarray        # (3,) Center of platform in base frame
    corners_3d_m: dict[int, np.ndarray] # tag_id -> projected (3,) position on plane
    raw_corners_3d_m: dict[int, np.ndarray] # tag_id -> raw 3D position before projection
    width_m: float              # Estimated platform width (m)
    height_m: float             # Estimated platform length / height (m)
    coplanarity_rmse_mm: float  # Residual distance to plane (mm)
    inlier_count: int
    total_points: int


def rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """Extract Euler XYZ angles (degrees) from rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


class RealtimeToolCalibrator:
    """Realtime 3D Tool Calibrator with RANSAC multi-camera & plane estimation."""

    def __init__(self, config: PlatformConfig | None = None):
        self.config = config or PlatformConfig()
        self.detector = Detector(
            families=self.config.tag_family,
            nthreads=2,
            quad_decimate=0.5,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
        )

    def detect_tags(self, image: np.ndarray) -> dict[int, TagDetection]:
        """Detect AprilTags in grayscale or BGR image."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        raw_detections = self.detector.detect(gray)
        detections: dict[int, TagDetection] = {}
        for d in raw_detections:
            tid = int(d.tag_id)
            tag_size = self.config.tag_sizes_m.get(tid, 0.0885)
            detections[tid] = TagDetection(
                tag_id=tid,
                corners_px=np.array(d.corners, dtype=np.float64),
                center_px=np.array(d.center, dtype=np.float64),
                tag_size_m=tag_size,
                decision_margin=float(getattr(d, "decision_margin", 0.0)),
            )
        return detections

    # =========================================================================
    # Phase 1: Robust Camera Extrinsics via Multi-Tag PnP RANSAC
    # =========================================================================

    def estimate_camera_extrinsics_ransac(
        self,
        camera_id: str | int,
        detections: dict[int, TagDetection],
        K: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
        known_base_to_platform: tuple[np.ndarray, np.ndarray] | None = None,
        reproj_threshold_px: float = 3.0,
        iterations_count: int = 1000,
    ) -> CameraExtrinsics | None:
        """Estimate camera extrinsics relative to Base Tag (ID 0) using RANSAC.

        If platform tags have a known or previously estimated relative pose
        to the base frame, all corners are pooled into a single high-accuracy
        multi-tag PnP RANSAC solver.
        Otherwise, base tag corners are solved with PnP RANSAC.
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        object_points_list: list[np.ndarray] = []
        image_points_list: list[np.ndarray] = []

        # 1. Base Tag (Direct base frame coordinates)
        if self.config.base_tag_id in detections:
            base_det = detections[self.config.base_tag_id]
            corners_3d = get_tag_model_corners_3d(base_det.tag_size_m)
            for pt3d, pt2d in zip(corners_3d, base_det.corners_px):
                object_points_list.append(pt3d)
                image_points_list.append(pt2d)

        # 2. Platform Corner Tags (if relative transform to base is provided)
        if known_base_to_platform is not None and self.config.known_platform_layout_m:
            R_plat_base, t_plat_base = known_base_to_platform
            for tid in self.config.corner_tag_ids:
                if tid in detections and tid in self.config.known_platform_layout_m:
                    det = detections[tid]
                    tag_local_corners = get_tag_model_corners_3d(det.tag_size_m)
                    lx, ly = self.config.known_platform_layout_m[tid]
                    # Transform tag local corners into platform frame, then into base frame
                    plat_corners = tag_local_corners + np.array([lx, ly, 0.0])
                    base_corners = (R_plat_base @ plat_corners.T).T + t_plat_base.reshape(1, 3)
                    for pt3d, pt2d in zip(base_corners, det.corners_px):
                        object_points_list.append(pt3d)
                        image_points_list.append(pt2d)

        if len(object_points_list) < 4:
            log.warning(f"Cam {camera_id}: Not enough 3D-2D points ({len(object_points_list)}) for PnP RANSAC.")
            return None

        obj_pts = np.array(object_points_list, dtype=np.float64)
        img_pts = np.array(image_points_list, dtype=np.float64)

        # Run PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            dist_coeffs,
            reprojectionError=reproj_threshold_px,
            iterationsCount=iterations_count,
            flags=cv2.SOLVEPNP_SQPNP if len(obj_pts) > 4 else cv2.SOLVEPNP_IPPE,
        )

        if not success or rvec is None or tvec is None:
            return None

        # Refine with Levenberg-Marquardt using inliers
        inlier_indices = inliers.flatten() if inliers is not None else np.arange(len(obj_pts))
        if len(inlier_indices) >= 4:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                obj_pts[inlier_indices],
                img_pts[inlier_indices],
                K,
                dist_coeffs,
                rvec,
                tvec,
            )
            rvec, tvec = refined_rvec, refined_tvec

        # Calculate reprojection error on inliers
        proj_pts, _ = cv2.projectPoints(obj_pts[inlier_indices], rvec, tvec, K, dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(proj_pts.squeeze() - img_pts[inlier_indices], axis=1)))

        R, _ = cv2.Rodrigues(rvec)
        # Invert: Cam pose in Base frame
        R_inv = R.T
        pos_in_base = -R_inv @ tvec.ravel()
        euler = rotation_matrix_to_euler_xyz_deg(R_inv)

        return CameraExtrinsics(
            camera_id=camera_id,
            rvec=rvec,
            tvec=tvec,
            R=R,
            reprojection_error_px=reproj_err,
            num_inliers=len(inlier_indices),
            total_points=len(obj_pts),
            inlier_ratio=float(len(inlier_indices) / len(obj_pts)),
            pos_in_base_m=pos_in_base,
            euler_xyz_deg=euler,
        )

    # =========================================================================
    # Phase 2: Robust Platform Plane Estimation via RANSAC 3D Plane Fitting
    # =========================================================================

    def estimate_single_tag_pose_in_camera(
        self,
        det: TagDetection,
        K: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate 3D pose of single tag relative to camera: (R, t)."""
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        obj_pts = get_tag_model_corners_3d(det.tag_size_m)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            det.corners_px,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel()

    @staticmethod
    def fit_plane_ransac(
        points_3d: np.ndarray,
        distance_threshold_m: float = 0.003,
        max_iterations: int = 1000,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Fit a plane (normal . X + d = 0) using RANSAC and SVD refinement.

        Returns:
            normal: Unit normal vector (3,) pointing towards +Z (nz > 0).
            d: Plane scalar offset.
            inlier_mask: Boolean array indicating consensus inliers.
            rmse_m: Root Mean Squared residual of inliers to plane.
        """
        N = len(points_3d)
        if N < 3:
            raise ValueError(f"Need at least 3 points to fit a plane, got {N}")

        best_inliers = np.zeros(N, dtype=bool)
        best_count = -1

        for _ in range(max_iterations):
            # Sample 3 non-collinear points
            idx = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = points_3d[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            d = -np.dot(normal, p1)

            # Distances of all points to plane
            distances = np.abs(points_3d @ normal + d)
            inliers = distances < distance_threshold_m
            count = np.sum(inliers)
            if count > best_count:
                best_count = count
                best_inliers = inliers

        inlier_points = points_3d[best_inliers]
        if len(inlier_points) < 3:
            inlier_points = points_3d
            best_inliers = np.ones(N, dtype=bool)

        # SVD Refinement on consensus inliers
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]  # Eigenvector with smallest singular value

        # Ensure normal points in +Z direction
        if normal[2] < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, centroid))

        # Inlier residual RMSE
        residuals = np.abs(inlier_points @ normal + d)
        rmse_m = float(np.sqrt(np.mean(residuals ** 2)))

        return normal, d, best_inliers, rmse_m

    def estimate_platform_plane_ransac(
        self,
        cameras_data: list[tuple[str | int, dict[int, TagDetection], CameraExtrinsics, np.ndarray]],
        distance_threshold_m: float = 0.003,
    ) -> PlaneCoordinates | None:
        """Extract Platform Plane coordinates using multi-camera observations.

        cameras_data: List of (camera_id, detections, extrinsics, K)
        """
        # Collect 3D points for each platform corner tag in Base Frame
        tag_3d_observations: dict[int, list[np.ndarray]] = {tid: [] for tid in self.config.corner_tag_ids}
        tag_corners_3d_obs: dict[int, list[np.ndarray]] = {tid: [] for tid in self.config.corner_tag_ids}

        for cam_id, detections, extrinsics, K in cameras_data:
            R_base_cam = extrinsics.R.T
            t_base_cam = -R_base_cam @ extrinsics.tvec.ravel()

            for tid in self.config.corner_tag_ids:
                if tid in detections:
                    det = detections[tid]
                    pose = self.estimate_single_tag_pose_in_camera(det, K)
                    if pose is not None:
                        R_c_t, t_c_t = pose
                        # Tag center in base frame
                        center_in_base = R_base_cam @ t_c_t + t_base_cam
                        tag_3d_observations[tid].append(center_in_base)

                        # Tag corners in base frame
                        corners_tag = get_tag_model_corners_3d(det.tag_size_m)
                        corners_in_cam = (R_c_t @ corners_tag.T).T + t_c_t.reshape(1, 3)
                        corners_in_base = (R_base_cam @ corners_in_cam.T).T + t_base_cam.reshape(1, 3)
                        tag_corners_3d_obs[tid].append(corners_in_base)

        # Average raw multi-view observations per tag
        raw_centers: dict[int, np.ndarray] = {}
        all_3d_points: list[np.ndarray] = []

        for tid in self.config.corner_tag_ids:
            obs = tag_3d_observations[tid]
            if obs:
                avg_center = np.mean(obs, axis=0)
                raw_centers[tid] = avg_center
                all_3d_points.append(avg_center)

            corner_obs = tag_corners_3d_obs[tid]
            if corner_obs:
                avg_corners = np.mean(corner_obs, axis=0)  # (4, 3)
                for pt in avg_corners:
                    all_3d_points.append(pt)

        if len(all_3d_points) < 4:
            log.warning(f"Not enough 3D observations for plane fitting: {len(all_3d_points)}")
            return None

        points_array = np.array(all_3d_points, dtype=np.float64)

        # Fit robust plane using RANSAC
        normal, d, inliers, rmse_m = self.fit_plane_ransac(
            points_array,
            distance_threshold_m=distance_threshold_m,
            max_iterations=1000,
        )

        # Orthogonally project tag centers onto the fitted plane:
        # P_proj = P - (normal . P + d) * normal
        projected_centers: dict[int, np.ndarray] = {}
        for tid, pt in raw_centers.items():
            dist = float(np.dot(normal, pt) + d)
            projected_centers[tid] = pt - dist * normal

        # Compute platform center origin
        proj_list = list(projected_centers.values())
        platform_origin = np.mean(proj_list, axis=0)

        # Compute metric dimensions (width & height across corners)
        # Assuming corner tag ordering: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        width_m, height_m = 0.0, 0.0
        if len(self.config.corner_tag_ids) >= 4 and all(tid in projected_centers for tid in self.config.corner_tag_ids[:4]):
            t0 = projected_centers[self.config.corner_tag_ids[0]]
            t1 = projected_centers[self.config.corner_tag_ids[1]]
            t2 = projected_centers[self.config.corner_tag_ids[2]]
            t3 = projected_centers[self.config.corner_tag_ids[3]]
            width_m = float((np.linalg.norm(t1 - t0) + np.linalg.norm(t2 - t3)) / 2.0)
            height_m = float((np.linalg.norm(t3 - t0) + np.linalg.norm(t2 - t1)) / 2.0)
        elif len(self.config.corner_tag_ids) == 2 and all(tid in projected_centers for tid in self.config.corner_tag_ids[:2]):
            t0 = projected_centers[self.config.corner_tag_ids[0]]
            t1 = projected_centers[self.config.corner_tag_ids[1]]
            width_m = float(np.linalg.norm(t1 - t0))
            height_m = 0.0

        return PlaneCoordinates(
            normal=normal,
            d_offset=d,
            origin_m=platform_origin,
            corners_3d_m=projected_centers,
            raw_corners_3d_m=raw_centers,
            width_m=width_m,
            height_m=height_m,
            coplanarity_rmse_mm=rmse_m * 1000.0,
            inlier_count=int(np.sum(inliers)),
            total_points=len(points_array),
        )
