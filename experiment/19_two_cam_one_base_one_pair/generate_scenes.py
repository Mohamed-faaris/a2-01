"""Synthetic Scene Generator for 2-Camera, 1 Base Tag, and 1 Tag Pair Benchmark.

Layout according to TESTING_PROCEDURE.md:
- 2 Cameras (cam0, cam1)
- 1 Large Base Marker: Tag 0 (134 mm)
- 1 Tag Pair on Platform Plane: Tag 2 and Tag 3 (88.5 mm each, defining a platform plane)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 900.0
SENSOR_WIDTH_MM = 36.0
LENS_MM = FOCAL_PX * SENSOR_WIDTH_MM / WIDTH

EXP_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXP_DIR.parent / "18_three_cam_platform_ransac" / "shared"


def build_blender_script(
    scene_dir: Path,
    scene_spec: dict,
) -> str:
    """Generate standalone reproducible build_scene.py for Blender."""
    return f'''import bpy
import json
import math
import os
import numpy as np

out_dir = r"{scene_dir.resolve()}"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Clean Scene
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

def create_tag_object(name, size_m, mat, location, rotation_euler):
    h = size_m / 2.0
    mesh = bpy.data.meshes.new(f"{{name}}_Mesh")
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

    # White backing plate
    hb = h * 1.2
    back_mesh = bpy.data.meshes.new(f"{{name}}_BackMesh")
    back_mesh.from_pydata([(-hb, -hb, -0.0005), (hb, -hb, -0.0005), (hb, hb, -0.0005), (-hb, hb, -0.0005)], [], [(0, 1, 2, 3)])
    back_mesh.update()
    back_obj = bpy.data.objects.new(f"{{name}}_Back", back_mesh)
    scene.collection.objects.link(back_obj)
    back_obj.location = location
    back_obj.rotation_euler = rotation_euler
    back_obj.data.materials.append(get_white_mat())
    return obj

shared_dir = r"{SHARED_DIR.resolve()}"
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, "tag_0.png"))
base_loc = {scene_spec["base_tag_location"]}
base_obj = create_tag_object("Tag_0_Base", {scene_spec["base_tag_size_m"]}, base_mat, base_loc, (0.0, 0.0, 0.0))

# Tag Pair
pair_specs = {json.dumps(scene_spec["platform_pair"])}
for tid_str, pdata in pair_specs.items():
    tid = int(tid_str)
    mat = create_tag_material(f"Mat_Tag_{{tid}}", os.path.join(shared_dir, f"tag_{{tid}}.png"))
    create_tag_object(
        f"Tag_{{tid}}",
        pdata["size_m"],
        mat,
        pdata["location"],
        pdata["rotation_euler"]
    )

target_dummy = bpy.data.objects.new("TargetCenter", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = {scene_spec["target_look_at"]}

# 2 Cameras
cam_configs = {json.dumps(scene_spec["cameras"])}
cam_objects = {{}}

for cam_id, ccfg in cam_configs.items():
    c_data = bpy.data.cameras.new(cam_id)
    c_data.lens = {LENS_MM}
    c_data.sensor_width = {SENSOR_WIDTH_MM}
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

# Save scene blend
blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

K = [
    [{FOCAL_PX}, 0.0, {WIDTH / 2.0}],
    [0.0, {FOCAL_PX}, {HEIGHT / 2.0}],
    [0.0, 0.0, 1.0]
]

gt = {{
    "intrinsics": {{
        "width": {WIDTH},
        "height": {HEIGHT},
        "fx": {FOCAL_PX},
        "fy": {FOCAL_PX},
        "cx": {WIDTH / 2.0},
        "cy": {HEIGHT / 2.0},
        "K": K
    }},
    "base_tag": {{
        "id": 0,
        "size_m": {scene_spec["base_tag_size_m"]},
        "position_world": {scene_spec["base_tag_location"]},
        "rotation_euler": [0.0, 0.0, 0.0]
    }},
    "platform": {{
        "normal_world": {scene_spec["platform_normal"]},
        "d_offset": {scene_spec["platform_d_offset"]},
        "pair_world": {json.dumps(scene_spec["platform_pair"])},
        "pair_baseline_m": {scene_spec["pair_baseline_m"]}
    }},
    "cameras": {{}}
}}

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

    gt["cameras"][cam_id] = {{
        "position_world": [float(x) for x in t_b_cam],
        "R_camera_to_world": [[float(x) for x in row] for row in R_w_c],
        "R_world_to_camera": [[float(x) for x in row] for row in R_c_w],
        "t_world_to_camera": [float(x) for x in t_c_w],
        "rvec": [float(x) for x in rvec]
    }}

    scene.camera = c_obj
    render_path = os.path.join(inputs_dir, f"{{cam_id}}_render.png")
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {{render_path}}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {{gt_path}}")
'''


def run_scene_generation():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Scene 002 exact tilted platform plane for tag pair
    rx, ry = math.radians(10), math.radians(-6)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    R_tilt = np.array([
        [cy,  0, sy],
        [sx*sy, cx, -sx*cy],
        [-cx*sy, sx, cx*cy]
    ])
    plat_center_002 = np.array([0.0, 0.10, 0.15])
    normal_002 = R_tilt @ np.array([0.0, 0.0, 1.0])
    normal_002 = normal_002 / np.linalg.norm(normal_002)
    d_002 = -float(np.dot(normal_002, plat_center_002))

    # Pair Tag 2 and Tag 3 offsets in plane
    p2_local = np.array([-0.12, 0.0, 0.0])
    p3_local = np.array([ 0.12, 0.0, 0.0])
    p2_world_002 = R_tilt @ p2_local + plat_center_002
    p3_world_002 = R_tilt @ p3_local + plat_center_002

    scenes_spec = {
        "001": {
            "name": "Flat Platform Plane with Tag Pair (12 cm elevation)",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [0.0, 0.0, 1.0],
            "platform_d_offset": -0.12,
            "pair_baseline_m": 0.24,
            "target_look_at": [0.0, -0.05, 0.06],
            "platform_pair": {
                "2": {"size_m": 0.0885, "location": [-0.12, 0.10, 0.12], "rotation_euler": [0.0, 0.0, 0.0]},
                "3": {"size_m": 0.0885, "location": [ 0.12, 0.10, 0.12], "rotation_euler": [0.0, 0.0, 0.0]}
            },
            "cameras": {
                "cam0": {"location": [-0.35, -0.70, 0.75]},
                "cam1": {"location": [ 0.35, -0.70, 0.75]}
            }
        },
        "002": {
            "name": "Tilted Platform Plane with Tag Pair (10 deg X, -6 deg Y tilt)",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [float(x) for x in normal_002],
            "platform_d_offset": float(d_002),
            "pair_baseline_m": 0.24,
            "target_look_at": [0.0, -0.05, 0.08],
            "platform_pair": {
                "2": {"size_m": 0.0885, "location": [float(x) for x in p2_world_002], "rotation_euler": [rx, ry, 0.0]},
                "3": {"size_m": 0.0885, "location": [float(x) for x in p3_world_002], "rotation_euler": [rx, ry, 0.0]}
            },
            "cameras": {
                "cam0": {"location": [-0.38, -0.72, 0.78]},
                "cam1": {"location": [ 0.38, -0.72, 0.78]}
            }
        },
        "003": {
            "name": "Oblique 2-Camera Setup with Tag Pair and Perturbation",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [0.0, 0.0, 1.0],
            "platform_d_offset": -0.10,
            "pair_baseline_m": 0.24,
            "target_look_at": [0.0, -0.05, 0.05],
            "platform_pair": {
                "2": {"size_m": 0.0885, "location": [-0.12, 0.10, 0.10], "rotation_euler": [0.0, 0.0, 0.0]},
                "3": {"size_m": 0.0885, "location": [ 0.12, 0.10, 0.10], "rotation_euler": [0.0, 0.0, 0.0]}
            },
            "cameras": {
                "cam0": {"location": [-0.48, -0.62, 0.62]},
                "cam1": {"location": [ 0.48, -0.62, 0.62]}
            }
        }
    }

    for scene_id, spec in scenes_spec.items():
        sdir = EXP_DIR / scene_id
        inputs_dir = sdir / "inputs"
        artifacts_dir = sdir / "artifacts"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(inputs_dir / "scene_spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        script = build_blender_script(sdir, spec)
        build_py_path = artifacts_dir / "build_scene.py"
        with open(build_py_path, "w") as f:
            f.write(script)

        print(f"\n--- Generating Scene {scene_id}: {spec['name']} ---")
        cmd = ["/snap/bin/blender", "-b", "-P", str(build_py_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error rendering scene {scene_id}:\n", res.stderr)
        else:
            print(f"Scene {scene_id} generated successfully!")


if __name__ == "__main__":
    run_scene_generation()
