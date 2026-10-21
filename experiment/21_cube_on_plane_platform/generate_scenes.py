#!/usr/bin/env python3
"""Synthetic Scene Generator: Cube RESTING ON the Plane-B Platform.

Constraint under test ("the cube can only be on the plane"): the 6-faced
AprilTag cube (20..25) sits flat on the platform plane defined by tags
42..45, so GT guarantees:

    cube_bottom_z == platform_top_z          (no floating, no sinking)
    cube_centroid_z == platform_top_z + L/2
    cube (x, y) inside the platform quad

Layout (world frame, metres, table at z=0):
- Base Tag 0 (134 mm) flat on the table
- Platform box top at z = 0.05, corners carry Plane-B Tags 42..45 (66 mm)
- Cube side L = 41 mm (registry size of tag 20 = demo's CubeConfig),
  centroid z = 0.05 + L/2, yaw-only rotation variants across scenes
- 2 cameras (1280x720, fx=fy=900) like experiment/20

Usage:  PYTHONPATH=src uv run python experiment/21_cube_on_plane_platform/generate_scenes.py
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 900.0
SENSOR_WIDTH_MM = 36.0
LENS_MM = FOCAL_PX * SENSOR_WIDTH_MM / WIDTH

EXP_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXP_DIR / "shared"

# --- Physical layout --------------------------------------------------------
PLATFORM_TOP_Z = 0.05
CUBE_L = 0.041           # = markers.yaml size for tag 20 (demo CubeConfig)
CUBE_Z = PLATFORM_TOP_Z + CUBE_L / 2.0   # resting exactly on the plane
BASE_TAG_LOC = [0.0, -0.16, 0.0005]
PLANE_TAG_IDS = [42, 43, 44, 45]
PLANE_TAG_SIZE = 0.066
PLANE_TAG_CORNERS = [                      # on the platform top, lifted 1 mm to avoid z-fighting
    [-0.07, 0.06, PLATFORM_TOP_Z + 0.001],
    [0.07, 0.06, PLATFORM_TOP_Z + 0.001],
    [0.07, 0.18, PLATFORM_TOP_Z + 0.001],
    [-0.07, 0.18, PLATFORM_TOP_Z + 0.001],
]
PLATFORM_BOX = {                           # outer dims of the platform
    "x_min": -0.13, "x_max": 0.13,
    "y_min": 0.0, "y_max": 0.24,
    "z_min": 0.0, "z_max": PLATFORM_TOP_Z,
}
CUBE_TAG_IDS = [20, 21, 22, 23, 24, 25]    # 20 top, 21 bottom, 22 +Y, 23 -Y, 24 +X, 25 -X
LOOK_AT = [0.0, -0.02, 0.05]
CAMERAS = {
    # Elevated viewpoint: the cube's TOP face must be seen steeply enough
    # (>= ~45 deg) for the 41 mm tags to keep enough pixels per module.
    "cam0": {"location": [-0.30, -0.45, 0.85]},
    "cam1": {"location": [0.30, -0.45, 0.85]},
}


def ensure_tag_images() -> None:
    """Generate tag36h11 textures for every tag used by this experiment."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in [0, *CUBE_TAG_IDS, *PLANE_TAG_IDS]:
        marker = cv2.aruco.generateImageMarker(family, tag_id, 512, borderBits=1)
        if not cv2.imwrite(str(SHARED_DIR / f"tag_{tag_id}.png"), marker):
            raise RuntimeError(f"Could not save generated tag ID {tag_id}")


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def generate_cube_faces(center: list[float], yaw: float) -> dict:
    """Cube face quads for the DEMO's tag mapping, cube yaw-only on the plane.

    Local frame (yaw = 0): +X right, +Y front, +Z top, matching
    CubeConfig.face_normals in the tracker (20 top, 21 bottom, 22 +Y,
    23 -Y, 24 +X, 25 -X).
    """
    half = CUBE_L / 2.0
    R = rot_z(yaw)
    c = np.asarray(center, dtype=float)

    # local offset (outward centre), local euler (Blender XYZ)
    local_defs = {
        20: ([0.0, 0.0, half], [0.0, 0.0, 0.0]),             # Top (+Z)
        21: ([0.0, 0.0, -half], [math.pi, 0.0, 0.0]),        # Bottom (-Z)
        22: ([0.0, half, 0.0], [-math.pi / 2, 0.0, 0.0]),    # Front (+Y)
        23: ([0.0, -half, 0.0], [math.pi / 2, 0.0, 0.0]),    # Back (-Y)
        24: ([half, 0.0, 0.0], [0.0, math.pi / 2, 0.0]),     # Right (+X)
        25: ([-half, 0.0, 0.0], [0.0, -math.pi / 2, 0.0]),   # Left (-X)
    }

    faces = {}
    for tid, (lpos, lrot) in local_defs.items():
        wpos = R @ np.asarray(lpos, dtype=float) + c
        faces[str(tid)] = {
            "size_m": CUBE_L,
            "location": [float(x) for x in wpos],
            # yaw-only cube rotation: Rz(yaw) @ R_local == euler (rx, ry, rz+yaw)
            "rotation_euler": [lrot[0], lrot[1], lrot[2] + yaw],
            "local_offset": [float(x) for x in lpos],
        }
    return faces


def build_blender_script(scene_dir: Path) -> str:
    """Template with __TOKEN__ placeholders (avoids f-string brace escaping)."""
    script = r'''
import bpy
import json
import math
import os
import numpy as np

out_dir = r"__SCENE_DIR__"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

spec = json.loads(r"""__SPEC_JSON__""")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = __WIDTH__ * 2
scene.render.resolution_y = __HEIGHT__ * 2   # 2x supersampling: the 41 mm cube
# faces alias badly at this distance with nearest-neighbour tag textures
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("NeutralWorld")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

def create_tag_material(name, img_path):
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    nodes.clear()
    img = bpy.data.images.load(img_path, check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat

def get_white_mat():
    mat = bpy.data.materials.get("WhiteBackingMat")
    if mat:
        return mat
    mat = bpy.data.materials.new("WhiteBackingMat")
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat

def get_platform_mat():
    mat = bpy.data.materials.get("PlatformMat")
    if mat:
        return mat
    mat = bpy.data.materials.new("PlatformMat")
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.30, 0.31, 0.34, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat

def create_tag_object(name, size_m, mat, location, rotation_euler, white_plate=True,
                      plate_scale=1.2, plate_offset=-0.0005):
    h = size_m / 2.0
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop_idx, c in zip(mesh.polygons[0].loop_indices, uv_coords):
        uv.data[loop_idx].uv = c
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.data.materials.append(mat)

    if white_plate:
        hb = h * plate_scale
        oz = plate_offset
        back_mesh = bpy.data.meshes.new(f"{name}_BackMesh")
        back_mesh.from_pydata([(-hb, -hb, oz), (hb, -hb, oz), (hb, hb, oz), (-hb, hb, oz)], [], [(0, 1, 2, 3)])
        back_mesh.update()
        back_obj = bpy.data.objects.new(f"{name}_Back", back_mesh)
        scene.collection.objects.link(back_obj)
        back_obj.location = location
        back_obj.rotation_euler = rotation_euler
        back_obj.data.materials.append(get_white_mat())
    return obj

shared_dir = r"__SHARED_DIR__"

# --- Platform box (the plane the cube must rest on) ------------------------
box = spec["platform_box"]
cx = (box["x_min"] + box["x_max"]) / 2.0
cy = (box["y_min"] + box["y_max"]) / 2.0
cz = (box["z_min"] + box["z_max"]) / 2.0
sx = (box["x_max"] - box["x_min"]) / 2.0
sy = (box["y_max"] - box["y_min"]) / 2.0
sz = (box["z_max"] - box["z_min"]) / 2.0
verts = [
    (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
    (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
    (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
    (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
]
faces_idx = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
pmesh = bpy.data.meshes.new("Platform_Mesh")
pmesh.from_pydata(verts, [], faces_idx)
pmesh.update()
pobj = bpy.data.objects.new("Platform", pmesh)
scene.collection.objects.link(pobj)
pobj.data.materials.append(get_platform_mat())

# --- Base Tag 0 -------------------------------------------------------------
base = spec["base_tag"]
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, f"tag_{base['id']}.png"))
create_tag_object("Tag_0_Base", base["size_m"], base_mat, base["location"], (0.0, 0.0, 0.0))

# --- Plane-B corner tags 42..45 --------------------------------------------
for tid_str, pdata in spec["plane_tags"].items():
    mat = create_tag_material(f"Mat_Tag_{tid_str}", os.path.join(shared_dir, f"tag_{tid_str}.png"))
    create_tag_object(f"Plane_Tag_{tid_str}", pdata["size_m"], mat, pdata["location"], (0.0, 0.0, 0.0))

# --- Cube faces (RESTING on the platform; tags flush with the 41 mm faces,
# quiet zone supplied by plates recessed 0.2 mm behind each face plane) ----
for tid_str, fdata in spec["cube"]["faces"].items():
    mat = create_tag_material(f"Mat_Cube_{tid_str}", os.path.join(shared_dir, f"tag_{tid_str}.png"))
    create_tag_object(f"Cube_Face_{tid_str}", fdata["size_m"], mat,
                      fdata["location"], fdata["rotation_euler"],
                      white_plate=True, plate_scale=1.08, plate_offset=-0.0002)

# --- Cameras ----------------------------------------------------------------
target_dummy = bpy.data.objects.new("LookAtDummy", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = spec["look_at"]

cam_objects = {}
for cam_id, ccfg in spec["cameras"].items():
    c_data = bpy.data.cameras.new(cam_id)
    c_data.lens = __LENS_MM__
    c_data.sensor_width = __SENSOR_MM__
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 20.0
    c_obj = bpy.data.objects.new(cam_id, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = ccfg["location"]
    track = c_obj.constraints.new(type="TRACK_TO")
    track.target = target_dummy
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam_objects[cam_id] = c_obj

bpy.context.view_layer.update()

blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

gt = json.loads(json.dumps(spec["gt_static"]))
gt["cameras"] = {}

R_b_cv = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])

for cam_id, c_obj in cam_objects.items():
    mw = np.array(c_obj.matrix_world)
    R_b_cam = mw[:3, :3]
    t_b_cam = mw[:3, 3]
    R_w_c = np.array(R_b_cam @ R_b_cv, dtype=float)
    t_w_c = np.array(t_b_cam, dtype=float)
    R_c_w = R_w_c.T
    t_c_w = -R_c_w @ t_w_c
    tr = np.trace(R_c_w)
    angle = math.acos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        rvec = [0.0, 0.0, 0.0]
    else:
        rx = float((R_c_w[2, 1] - R_c_w[1, 2]) / (2.0 * math.sin(angle)))
        ry = float((R_c_w[0, 2] - R_c_w[2, 0]) / (2.0 * math.sin(angle)))
        rz = float((R_c_w[1, 0] - R_c_w[0, 1]) / (2.0 * math.sin(angle)))
        rvec = [rx * angle, ry * angle, rz * angle]
    gt["cameras"][cam_id] = {
        "position_world": [float(x) for x in t_b_cam],
        "R_camera_to_world": [[float(x) for x in row] for row in R_w_c],
        "R_world_to_camera": [[float(x) for x in row] for row in R_c_w],
        "t_world_to_camera": [float(x) for x in t_c_w],
        "rvec": [float(x) for x in rvec]
    }

    scene.camera = c_obj
    render_path = os.path.join(inputs_dir, f"{cam_id}_render.png")
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    # Box-filter the 2x supersampled render down to the nominal 1280x720 (K stays fx=900)
    img = bpy.data.images.load(render_path)
    img.scale(__WIDTH__, __HEIGHT__)
    img.filepath_raw = render_path
    img.file_format = "PNG"
    img.save()
    print(f"Rendered {render_path}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {gt_path}")
'''
    return script


def make_spec(name: str, cube_center: list[float], yaw_deg: float) -> dict:
    yaw = math.radians(yaw_deg)
    faces = generate_cube_faces(cube_center, yaw)

    plane_tags = {
        str(tid): {"size_m": PLANE_TAG_SIZE, "location": loc}
        for tid, loc in zip(PLANE_TAG_IDS, PLANE_TAG_CORNERS)
    }

    gt_static = {
        "intrinsics": {
            "width": WIDTH, "height": HEIGHT,
            "fx": FOCAL_PX, "fy": FOCAL_PX,
            "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
            "K": [[FOCAL_PX, 0.0, WIDTH / 2.0], [0.0, FOCAL_PX, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        },
        "base_tag": {"id": 0, "size_m": 0.134, "location": BASE_TAG_LOC, "rotation_euler": [0.0, 0.0, 0.0]},
        "plane_b": {
            "tag_ids": PLANE_TAG_IDS,
            "tag_size_m": PLANE_TAG_SIZE,
            "top_z_world": PLATFORM_TOP_Z,
            "normal_world": [0.0, 0.0, 1.0],
            "center_world": [0.0, 0.12, PLATFORM_TOP_Z],
            "corners_world": {str(tid): loc for tid, loc in zip(PLANE_TAG_IDS, PLANE_TAG_CORNERS)},
        },
        "cube": {
            "side_length_m": CUBE_L,
            "centroid_world": cube_center,
            "yaw_rad": yaw,
            "yaw_deg": yaw_deg,
            "resting_on_plane": True,
            "bottom_z_world": cube_center[2] - CUBE_L / 2.0,
            "faces": faces,
        },
        "constraint": {
            "cube_bottom_z_equals_plane_top": abs((cube_center[2] - CUBE_L / 2.0) - PLATFORM_TOP_Z) < 1e-9,
            "plane_top_z": PLATFORM_TOP_Z,
            "expected_centroid_z": CUBE_Z,
        },
    }

    return {
        "name": name,
        "base_tag": {"id": 0, "size_m": 0.134, "location": BASE_TAG_LOC},
        "platform_box": PLATFORM_BOX,
        "plane_tags": plane_tags,
        "cube": {"side_length_m": CUBE_L, "center": cube_center, "yaw_deg": yaw_deg, "faces": faces},
        "look_at": LOOK_AT,
        "cameras": CAMERAS,
        "gt_static": gt_static,
    }


def run_scene_generation() -> None:
    ensure_tag_images()

    scenes = {
        "001": ("Cube-On-Plane Nominal (yaw 0)", [0.0, 0.12, CUBE_Z], 0.0),
        "002": ("Cube-On-Plane Yawed 30deg, offset", [0.035, 0.10, CUBE_Z], 30.0),
        "003": ("Cube-On-Plane Yawed -45deg, near edge", [-0.03, 0.145, CUBE_Z], -45.0),
    }

    for scene_id, (name, center, yaw) in scenes.items():
        sdir = EXP_DIR / scene_id
        inputs_dir = sdir / "inputs"
        artifacts_dir = sdir / "artifacts"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        spec = make_spec(name, center, yaw)
        with open(inputs_dir / "scene_spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        script = (build_blender_script(sdir)
                  .replace("__SCENE_DIR__", str(sdir.resolve()))
                  .replace("__SPEC_JSON__", json.dumps(spec))
                  .replace("__SHARED_DIR__", str(SHARED_DIR.resolve()))
                  .replace("__WIDTH__", str(WIDTH))
                  .replace("__HEIGHT__", str(HEIGHT))
                  .replace("__LENS_MM__", str(LENS_MM))
                  .replace("__SENSOR_MM__", str(SENSOR_WIDTH_MM)))
        build_py_path = artifacts_dir / "build_scene.py"
        with open(build_py_path, "w") as f:
            f.write(script)

        print(f"\n--- Generating Scene {scene_id}: {name} ---")
        res = subprocess.run(["/snap/bin/blender", "-b", "-P", str(build_py_path)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error rendering scene {scene_id}:\n", res.stderr[-3000:])
            raise SystemExit(1)
        print(f"Scene {scene_id} generated successfully!")


if __name__ == "__main__":
    run_scene_generation()
