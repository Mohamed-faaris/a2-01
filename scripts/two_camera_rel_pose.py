#!/usr/bin/env python3
"""One AprilTag, two cameras: compute relative camera pose via AprilTag detection.

World Frame is defined as Tag Frame:
  - AprilTag is at origin (0, 0, 0) facing +Z (in camera/view space) or standard plane.
  - Camera 0 is placed at (x0, y0, z0) oriented towards the tag.
  - Camera 1 is placed at (x1, y1, z1) oriented towards the tag.
  - Both cameras render the scene with the AprilTag in view.
  - pupil-apriltags detects the tag in cam0 and cam1 images, outputting:
        T_c0_tag = (R_c0_t, t_c0_t)
        T_c1_tag = (R_c1_t, t_c1_t)
  - Relative transform from cam0 to cam1:
        R_rel_est = R_c1_t @ R_c0_t.T
        t_rel_est = t_c1_t - R_rel_est @ t_c0_t
  - Ground truth relative camera transform is calculated from Blender camera world matrices.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector


WIDTH, HEIGHT = 640, 480
FOCAL_PIXELS = 500.0
TAG_ID = 0
TAG_SIZE_M = 0.08  # 8 cm tag

MAX_REL_CAM_TRANS_ERROR_M = 0.03   # 3 cm
MAX_REL_CAM_ROT_ERROR_DEG = 3.0    # 3 degrees


def rotation_xyz_matrix(angles: tuple[float, float, float] | list[float]) -> np.ndarray:
    rx, ry, rz = angles
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def rotation_error_deg(r_est: np.ndarray, r_exp: np.ndarray) -> float:
    delta = np.asarray(r_est, dtype=float) @ np.asarray(r_exp, dtype=float).T
    trace_val = (np.trace(delta) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(trace_val, -1.0, 1.0))))


def ensure_tag_image(tag_dir: Path, tag_id: int = TAG_ID) -> Path:
    tag_dir.mkdir(parents=True, exist_ok=True)
    tag_path = tag_dir / f"tag_{tag_id}.png"
    if not tag_path.exists():
        family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(family, tag_id, 512, borderBits=1)
        cv2.imwrite(str(tag_path), marker)
    return tag_path


def generate_blender_scene_script(
    out_dir: Path,
    tag_path: Path,
    cam0_cv_pos: tuple[float, float, float],
    cam0_rot_xyz: tuple[float, float, float],
    cam1_cv_pos: tuple[float, float, float],
    cam1_rot_xyz: tuple[float, float, float],
) -> str:
    """
    OpenCV Camera coordinates:
      Tag is at (0, 0, 0).
      Camera 0 is at cam0_cv_pos, with camera orientation.
      In Blender:
      Tag mesh is at (0, 0, 0) in XY plane.
      Camera 0 is positioned in Blender coordinates.
    """
    escaped_out = str(out_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_tag = str(tag_path.resolve()).replace("\\", "\\\\").replace("'", "\\'")

    return f'''import bpy
import math
import os
import mathutils

out = r'{escaped_out}'
tag_img_path = r'{escaped_tag}'
os.makedirs(out, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {{item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = {WIDTH}
scene.render.resolution_y = {HEIGHT}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("Neutral background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.52, 0.52, 0.52, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Tag Material
image = bpy.data.images.load(tag_img_path, check_existing=True)
image.colorspace_settings.name = "Non-Color"
mat = bpy.data.materials.new("TagMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
tex = nodes.new("ShaderNodeTexImage")
tex.image = image
tex.interpolation = "Closest"
emission = nodes.new("ShaderNodeEmission")
out_node = nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

# Tag mesh centered at origin
h = {TAG_SIZE_M} / 2.0
mesh = bpy.data.meshes.new("TagMesh")
mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
mesh.update()
uv = mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

tag_obj = bpy.data.objects.new("AprilTag", mesh)
scene.collection.objects.link(tag_obj)
tag_obj.location = (0.0, 0.0, 0.0)
tag_obj.rotation_euler = (0.0, 0.0, 0.0)
tag_obj.data.materials.append(mat)

# Setup cameras pointing at tag
def create_cam(name, loc, target_obj, roll=0.0):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = {FOCAL_PIXELS} * 36.0 / {WIDTH}
    c_data.sensor_width = 36.0
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 100.0
    c_obj = bpy.data.objects.new(name, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = loc
    
    # Track to constraint to look at tag
    track = c_obj.constraints.new(type='TRACK_TO')
    track.target = target_obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    return c_obj

# Camera positions in Blender world (tag is at 0,0,0, facing +Z in Blender)
cam0 = create_cam("Cam0", {cam0_cv_pos}, tag_obj)
cam1 = create_cam("Cam1", {cam1_cv_pos}, tag_obj)

bpy.context.view_layer.update()

# Save scene
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))

# Render cam0
scene.camera = cam0
scene.render.filepath = os.path.join(out, "cam0_render.png")
bpy.ops.render.render(write_still=True)

# Render cam1
scene.camera = cam1
scene.render.filepath = os.path.join(out, "cam1_render.png")
bpy.ops.render.render(write_still=True)

# Output camera world matrices for exact ground truth
import json
def get_cv_extrinsics(cam_obj):
    # cam_obj.matrix_world maps camera coords to world coords
    mat_world = cam_obj.matrix_world
    R_world = np_mat = [list(row) for row in mat_world]
    return R_world

with open(os.path.join(out, "blender_cams.json"), "w") as f:
    json.dump({{
        "cam0_matrix_world": [list(row) for row in cam0.matrix_world],
        "cam1_matrix_world": [list(row) for row in cam1.matrix_world],
    }}, f, indent=2)
'''


def get_cv_camera_extrinsics_from_blender_matrix(M_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Blender camera local coordinate:
      Right: +X, Up: +Y, Forward (view direction): -Z.
    OpenCV camera coordinate:
      Right: +X, Down: -Y, Forward (view direction): +Z.
    Change of basis matrix from CV camera to Blender camera:
      P_blender_cam = R_cv2b @ P_cv_cam
      where R_cv2b = diag([1, -1, -1]).
    World matrix M_world has [R_b | t_b] where P_world = R_b @ P_blender_cam + t_b.
    Therefore:
      P_world = R_b @ R_cv2b @ P_cv_cam + t_b.
    In OpenCV camera frame:
      P_cv_cam = (R_b @ R_cv2b)^T @ (P_world - t_b)
               = R_w2c @ P_world + t_w2c
      where R_w2c = (R_b @ R_cv2b)^T = R_cv2b.T @ R_b.T
            t_w2c = -R_w2c @ t_b
    """
    R_b = M_world[:3, :3]
    t_b = M_world[:3, 3]

    R_cv2b = np.diag([1.0, -1.0, -1.0])
    R_w2c = R_cv2b @ R_b.T
    t_w2c = -R_w2c @ t_b
    return R_w2c, t_w2c


def detect_tag_in_image(img_bgr: np.ndarray, detector: Detector, tag_size: float = TAG_SIZE_M):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    intrinsics = (FOCAL_PIXELS, FOCAL_PIXELS, WIDTH / 2.0, HEIGHT / 2.0)
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=tag_size)
    for det in detections:
        if det.tag_id == TAG_ID:
            R = np.asarray(det.pose_R, dtype=float)
            t = np.asarray(det.pose_t, dtype=float).reshape(3)
            return det, R, t
    return None, None, None


def annotate_detection(img_bgr: np.ndarray, det, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    annotated = img_bgr.copy()
    corners = np.asarray(det.corners, dtype=float).reshape(-1, 2).astype(int)
    cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
    center = tuple(np.asarray(det.center, dtype=float).astype(int).tolist())
    cv2.circle(annotated, center, 4, (0, 0, 255), -1)

    axis_len = TAG_SIZE_M * 0.75
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # X: blue, Y: green, Z: red

    def project(pt3):
        x, y, z = pt3
        z = max(z, 1e-6)
        u = int(FOCAL_PIXELS * x / z + WIDTH / 2.0)
        v = int(FOCAL_PIXELS * y / z + HEIGHT / 2.0)
        return (u, v)

    origin = project(t)
    for axis, color in zip(np.eye(3) * axis_len, colors):
        tip = project(t + R @ axis)
        cv2.arrowedLine(annotated, origin, tip, color, 2, tipLength=0.15)
    return annotated


def run_experiment(
    exp_dir: Path,
    blender_bin: str = "/snap/bin/blender",
    cam0_blender_pos: tuple[float, float, float] = (-0.18, -0.05, 0.95),
    cam1_blender_pos: tuple[float, float, float] = (0.22, 0.08, 1.10),
) -> dict:
    exp_dir.mkdir(parents=True, exist_ok=True)
    tags_dir = exp_dir.parent / "_shared_tags"
    tag_img_path = ensure_tag_image(tags_dir, TAG_ID)

    # 1. Blender scene script
    script_str = generate_blender_scene_script(
        out_dir=exp_dir,
        tag_path=tag_img_path,
        cam0_cv_pos=cam0_blender_pos,
        cam0_rot_xyz=(0, 0, 0),
        cam1_cv_pos=cam1_blender_pos,
        cam1_rot_xyz=(0, 0, 0),
    )
    script_path = exp_dir / "build_scene.py"
    script_path.write_text(script_str, encoding="utf-8")

    # 2. Run Blender to build scene and render both camera images
    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(script_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender execution failed:\n{res.stderr}\n{res.stdout}")

    cam0_img_path = exp_dir / "cam0_render.png"
    cam1_img_path = exp_dir / "cam1_render.png"
    if not cam0_img_path.exists() or not cam1_img_path.exists():
        raise RuntimeError("Blender failed to produce render images.")

    # 3. Detect tag in both camera frames
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    img0 = cv2.imread(str(cam0_img_path))
    img1 = cv2.imread(str(cam1_img_path))

    det0, R_c0_t, t_c0_t = detect_tag_in_image(img0, detector)
    det1, R_c1_t, t_c1_t = detect_tag_in_image(img1, detector)

    if det0 is None or det1 is None:
        raise RuntimeError(f"Tag detection failed (det0={det0 is not None}, det1={det1 is not None})")

    # Annotate renders
    cv2.imwrite(str(exp_dir / "cam0_annotated.png"), annotate_detection(img0, det0, R_c0_t, t_c0_t))
    cv2.imwrite(str(exp_dir / "cam1_annotated.png"), annotate_detection(img1, det1, R_c1_t, t_c1_t))

    # 4. Relative camera pose calculation from AprilTag detections:
    # Camera 0 sees tag: P_c0 = R_c0_t * P_t + t_c0_t  =>  P_t = R_c0_t^T (P_c0 - t_c0_t)
    # Camera 1 sees tag: P_c1 = R_c1_t * P_t + t_c1_t
    # Substitute P_t:
    # P_c1 = R_c1_t * R_c0_t^T * P_c0 + (t_c1_t - R_c1_t * R_c0_t^T * t_c0_t)
    #
    # Estimated relative transform from cam0 to cam1 (P_c1 = R_rel * P_c0 + t_rel):
    R_rel_est = R_c1_t @ R_c0_t.T
    t_rel_est = t_c1_t - R_rel_est @ t_c0_t

    # 5. Ground truth relative camera pose from Blender camera world matrices
    b_cams = json.loads((exp_dir / "blender_cams.json").read_text(encoding="utf-8"))
    M0 = np.array(b_cams["cam0_matrix_world"])
    M1 = np.array(b_cams["cam1_matrix_world"])

    R_w2c0, t_w2c0 = get_cv_camera_extrinsics_from_blender_matrix(M0)
    R_w2c1, t_w2c1 = get_cv_camera_extrinsics_from_blender_matrix(M1)

    # Relative transform from cam0 to cam1 in ground truth:
    # P_c0 = R0 * P_w + t0  =>  P_w = R0.T (P_c0 - t0)
    # P_c1 = R1 * P_w + t1 = R1 * R0.T * P_c0 + (t1 - R1 * R0.T * t0)
    R_rel_gt = R_w2c1 @ R_w2c0.T
    t_rel_gt = t_w2c1 - R_rel_gt @ t_w2c0

    # 6. Errors
    trans_err_m = float(np.linalg.norm(t_rel_est - t_rel_gt))
    rot_err_deg = float(rotation_error_deg(R_rel_est, R_rel_gt))
    gt_dist_m = float(np.linalg.norm(t_rel_gt))
    est_dist_m = float(np.linalg.norm(t_rel_est))
    dist_err_m = abs(est_dist_m - gt_dist_m)

    passed = (trans_err_m <= MAX_REL_CAM_TRANS_ERROR_M) and (rot_err_deg <= MAX_REL_CAM_ROT_ERROR_DEG)

    report = {
        "tag_id": TAG_ID,
        "tag_size_m": TAG_SIZE_M,
        "ground_truth": {
            "cam0_pos_blender": list(cam0_blender_pos),
            "cam1_pos_blender": list(cam1_blender_pos),
            "rel_cam0_to_cam1_trans_m": t_rel_gt.tolist(),
            "rel_cam0_to_cam1_rot_matrix": R_rel_gt.tolist(),
            "rel_cam0_to_cam1_distance_m": gt_dist_m,
        },
        "estimated": {
            "cam0_tag_t_m": t_c0_t.tolist(),
            "cam1_tag_t_m": t_c1_t.tolist(),
            "rel_cam0_to_cam1_trans_m": t_rel_est.tolist(),
            "rel_cam0_to_cam1_rot_matrix": R_rel_est.tolist(),
            "rel_cam0_to_cam1_distance_m": est_dist_m,
        },
        "errors": {
            "translation_error_m": trans_err_m,
            "translation_error_cm": trans_err_m * 100.0,
            "distance_error_m": dist_err_m,
            "distance_error_cm": dist_err_m * 100.0,
            "rotation_error_deg": rot_err_deg,
        },
        "thresholds": {
            "max_translation_error_m": MAX_REL_CAM_TRANS_ERROR_M,
            "max_rotation_error_deg": MAX_REL_CAM_ROT_ERROR_DEG,
        },
        "test_passed": passed,
    }

    report_path = exp_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiment"))
    parser.add_argument("--name", default="two_camera_rel_pose")
    parser.add_argument("--id", default="001")
    args = parser.parse_args()

    target_dir = args.experiment_root / args.name / args.id
    print(f"Running experiment in: {target_dir}")
    report = run_experiment(target_dir, blender_bin=args.blender)
    print(json.dumps(report, indent=2))
    return 0 if report["test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
