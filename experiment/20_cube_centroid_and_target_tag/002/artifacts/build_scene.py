import bpy
import json
import math
import os
import numpy as np

out_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/20_cube_centroid_and_target_tag/002"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
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

    # For planar tags (base, target), add quiet white backing plate
    if "Cube" not in name:
        hb = h * 1.2
        back_mesh = bpy.data.meshes.new(f"{name}_BackMesh")
        back_mesh.from_pydata([(-hb, -hb, -0.0005), (hb, -hb, -0.0005), (hb, hb, -0.0005), (-hb, hb, -0.0005)], [], [(0, 1, 2, 3)])
        back_mesh.update()
        back_obj = bpy.data.objects.new(f"{name}_Back", back_mesh)
        scene.collection.objects.link(back_obj)
        back_obj.location = location
        back_obj.rotation_euler = rotation_euler
        back_obj.data.materials.append(get_white_mat())
    return obj

shared_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/18_three_cam_platform_ransac/shared"

# 1. Base Tag (ID 0)
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, "tag_0.png"))
create_tag_object("Tag_0_Base", 0.134, base_mat, [0.0, -0.22, 0.0], (0.0, 0.0, 0.0))

# 2. Target Marker (Tag 1)
target_mat = create_tag_material("TargetMat", os.path.join(shared_dir, "tag_1.png"))
create_tag_object("Tag_1_Target", 0.124, target_mat, [-0.16, 0.1, 0.12], [0.13962634015954636, -0.08726646259971647, 0.0])

# 3. Rigid 6-Faced Cube (Tags 6..11)
cube_spec = {"side_length_m": 0.0885, "center": [0.1, 0.08, 0.16], "rotation_euler": [0.4, 0.35, -0.2], "faces": {"6": {"size_m": 0.0885, "location": [0.11027346697815306, 0.060335229368698506, 0.1982859657350719], "rotation_euler": [0.4, 0.35, -0.2], "local_offset": [0.0, 0.0, 0.04425]}, "7": {"size_m": 0.0885, "location": [0.08972653302184695, 0.0996647706313015, 0.12171403426492809], "rotation_euler": [3.541592653589793, 0.35, -0.2], "local_offset": [0.0, 0.0, -0.04425]}, "8": {"size_m": 0.0885, "location": [0.08611189220761159, 0.04122936057344122, 0.14381295331436997], "rotation_euler": [1.9707963267948965, 0.35, -0.2], "local_offset": [0.0, -0.04425, 0.0]}, "9": {"size_m": 0.0885, "location": [0.11388810779238842, 0.11877063942655879, 0.17618704668563004], "rotation_euler": [-1.1707963267948966, 0.35, -0.2], "local_offset": [0.0, 0.04425, 0.0]}, "10": {"size_m": 0.0885, "location": [0.14073866514990152, 0.07174186374088755, 0.14482677202009628], "rotation_euler": [0.4, 1.9207963267948966, -0.2], "local_offset": [0.04425, 0.0, 0.0]}, "11": {"size_m": 0.0885, "location": [0.059261334850098504, 0.08825813625911245, 0.17517322797990373], "rotation_euler": [0.4, -1.2207963267948965, -0.2], "local_offset": [-0.04425, 0.0, 0.0]}}}
side_L = cube_spec["side_length_m"]
faces = cube_spec["faces"]

# Create each cube face quad facing outward
for tid_str, fdata in faces.items():
    tid = int(tid_str)
    mat = create_tag_material(f"Mat_Tag_{tid}", os.path.join(shared_dir, f"tag_{tid}.png"))
    create_tag_object(
        f"Cube_Face_{tid}",
        side_L,
        mat,
        fdata["location"],
        fdata["rotation_euler"]
    )

target_dummy = bpy.data.objects.new("LookAtDummy", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = [0.0, -0.05, 0.08]

# 2 Cameras
cam_configs = {"cam0": {"location": [-0.4, -0.7, 0.75]}, "cam1": {"location": [0.4, -0.7, 0.75]}}
cam_objects = {}

for cam_id, ccfg in cam_configs.items():
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

# Save scene blend
blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

K = [
    [900.0, 0.0, 640.0],
    [0.0, 900.0, 360.0],
    [0.0, 0.0, 1.0]
]

gt = {
    "intrinsics": {
        "width": 1280,
        "height": 720,
        "fx": 900.0,
        "fy": 900.0,
        "cx": 640.0,
        "cy": 360.0,
        "K": K
    },
    "base_tag": {
        "id": 0,
        "size_m": 0.134,
        "position_world": [0.0, -0.22, 0.0],
        "rotation_euler": [0.0, 0.0, 0.0]
    },
    "target_tag": {
        "id": 1,
        "size_m": 0.124,
        "position_world": [-0.16, 0.1, 0.12],
        "rotation_euler": [0.13962634015954636, -0.08726646259971647, 0.0]
    },
    "cube": {
        "side_length_m": side_L,
        "centroid_world": cube_center,
        "rotation_euler": cube_spec["rotation_euler"],
        "faces": faces
    },
    "cameras": {}
}

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
    print(f"Rendered {render_path}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {gt_path}")
