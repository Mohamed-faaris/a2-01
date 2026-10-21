
import bpy
import json
import math
import os
import numpy as np

out_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/21_cube_on_plane_platform/001"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

spec = json.loads(r"""{"name": "Cube-On-Plane Nominal (yaw 0)", "base_tag": {"id": 0, "size_m": 0.134, "location": [0.0, -0.16, 0.0005]}, "platform_box": {"x_min": -0.13, "x_max": 0.13, "y_min": 0.0, "y_max": 0.24, "z_min": 0.0, "z_max": 0.05}, "plane_tags": {"42": {"size_m": 0.066, "location": [-0.07, 0.06, 0.051000000000000004]}, "43": {"size_m": 0.066, "location": [0.07, 0.06, 0.051000000000000004]}, "44": {"size_m": 0.066, "location": [0.07, 0.18, 0.051000000000000004]}, "45": {"size_m": 0.066, "location": [-0.07, 0.18, 0.051000000000000004]}}, "cube": {"side_length_m": 0.041, "center": [0.0, 0.12, 0.07050000000000001], "yaw_deg": 0.0, "faces": {"20": {"size_m": 0.041, "location": [0.0, 0.12, 0.09100000000000001], "rotation_euler": [0.0, 0.0, 0.0], "local_offset": [0.0, 0.0, 0.0205]}, "21": {"size_m": 0.041, "location": [0.0, 0.12, 0.05], "rotation_euler": [3.141592653589793, 0.0, 0.0], "local_offset": [0.0, 0.0, -0.0205]}, "22": {"size_m": 0.041, "location": [0.0, 0.14049999999999999, 0.07050000000000001], "rotation_euler": [-1.5707963267948966, 0.0, 0.0], "local_offset": [0.0, 0.0205, 0.0]}, "23": {"size_m": 0.041, "location": [0.0, 0.09949999999999999, 0.07050000000000001], "rotation_euler": [1.5707963267948966, 0.0, 0.0], "local_offset": [0.0, -0.0205, 0.0]}, "24": {"size_m": 0.041, "location": [0.0205, 0.12, 0.07050000000000001], "rotation_euler": [0.0, 1.5707963267948966, 0.0], "local_offset": [0.0205, 0.0, 0.0]}, "25": {"size_m": 0.041, "location": [-0.0205, 0.12, 0.07050000000000001], "rotation_euler": [0.0, -1.5707963267948966, 0.0], "local_offset": [-0.0205, 0.0, 0.0]}}}, "look_at": [0.0, -0.02, 0.05], "cameras": {"cam0": {"location": [-0.3, -0.45, 0.85]}, "cam1": {"location": [0.3, -0.45, 0.85]}}, "gt_static": {"intrinsics": {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0, "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]}, "base_tag": {"id": 0, "size_m": 0.134, "location": [0.0, -0.16, 0.0005], "rotation_euler": [0.0, 0.0, 0.0]}, "plane_b": {"tag_ids": [42, 43, 44, 45], "tag_size_m": 0.066, "top_z_world": 0.05, "normal_world": [0.0, 0.0, 1.0], "center_world": [0.0, 0.12, 0.05], "corners_world": {"42": [-0.07, 0.06, 0.051000000000000004], "43": [0.07, 0.06, 0.051000000000000004], "44": [0.07, 0.18, 0.051000000000000004], "45": [-0.07, 0.18, 0.051000000000000004]}}, "cube": {"side_length_m": 0.041, "centroid_world": [0.0, 0.12, 0.07050000000000001], "yaw_rad": 0.0, "yaw_deg": 0.0, "resting_on_plane": true, "bottom_z_world": 0.05, "faces": {"20": {"size_m": 0.041, "location": [0.0, 0.12, 0.09100000000000001], "rotation_euler": [0.0, 0.0, 0.0], "local_offset": [0.0, 0.0, 0.0205]}, "21": {"size_m": 0.041, "location": [0.0, 0.12, 0.05], "rotation_euler": [3.141592653589793, 0.0, 0.0], "local_offset": [0.0, 0.0, -0.0205]}, "22": {"size_m": 0.041, "location": [0.0, 0.14049999999999999, 0.07050000000000001], "rotation_euler": [-1.5707963267948966, 0.0, 0.0], "local_offset": [0.0, 0.0205, 0.0]}, "23": {"size_m": 0.041, "location": [0.0, 0.09949999999999999, 0.07050000000000001], "rotation_euler": [1.5707963267948966, 0.0, 0.0], "local_offset": [0.0, -0.0205, 0.0]}, "24": {"size_m": 0.041, "location": [0.0205, 0.12, 0.07050000000000001], "rotation_euler": [0.0, 1.5707963267948966, 0.0], "local_offset": [0.0205, 0.0, 0.0]}, "25": {"size_m": 0.041, "location": [-0.0205, 0.12, 0.07050000000000001], "rotation_euler": [0.0, -1.5707963267948966, 0.0], "local_offset": [-0.0205, 0.0, 0.0]}}}, "constraint": {"cube_bottom_z_equals_plane_top": true, "plane_top_z": 0.05, "expected_centroid_z": 0.07050000000000001}}}""")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1280 * 2
scene.render.resolution_y = 720 * 2   # 2x supersampling: the 41 mm cube
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

shared_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/21_cube_on_plane_platform/shared"

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
    c_data.lens = 25.3125
    c_data.sensor_width = 36.0
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
    img.scale(1280, 720)
    img.filepath_raw = render_path
    img.file_format = "PNG"
    img.save()
    print(f"Rendered {render_path}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {gt_path}")
