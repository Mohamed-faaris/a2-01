import bpy
import json
import math
import os
import mathutils
import numpy as np

out_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/18_three_cam_platform_ransac/002"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Clean Scene
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

# Neutral ambient lighting
world = bpy.data.worlds.new("NeutralWorld")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

def create_tag_material(name, img_path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
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
    mat.use_nodes = True
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
    # Tag corners matching OpenCV AprilTag orientation:
    # 0: [-h, -h], 1: [h, -h], 2: [h, h], 3: [-h, h]
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

    # White quiet margin plate
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

# Create Base Tag (ID 0)
shared_dir = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/18_three_cam_platform_ransac/shared"
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, "tag_0.png"))
base_loc = [0.0, -0.22, 0.0]
base_obj = create_tag_object("Tag_0_Base", 0.134, base_mat, base_loc, (0.0, 0.0, 0.0))

# Create Platform Corner Tags
corner_specs = {"2": {"size_m": 0.0885, "location": [-0.13923306535155827, 0.2010219401037262, 0.15295315602159185], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "3": {"size_m": 0.0885, "location": [0.13923306535155827, 0.19593961049871544, 0.1817764795117942], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "4": {"size_m": 0.0885, "location": [0.13923306535155827, -0.0010219401037261794, 0.14704684397840814], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "5": {"size_m": 0.0885, "location": [-0.13923306535155827, 0.004060389501284581, 0.11822352048820578], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}}
platform_objs = []
for tid_str, cdata in corner_specs.items():
    tid = int(tid_str)
    mat = create_tag_material(f"Mat_Tag_{tid}", os.path.join(shared_dir, f"tag_{tid}.png"))
    obj = create_tag_object(
        f"Tag_{tid}",
        cdata["size_m"],
        mat,
        cdata["location"],
        cdata["rotation_euler"]
    )
    platform_objs.append(obj)

# Create Target Dummy Object for Camera Look-At
target_dummy = bpy.data.objects.new("TargetCenter", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = [0.0, -0.05, 0.08]

# Create Cameras
cam_configs = {"cam0": {"location": [-0.45, -0.75, 0.8]}, "cam1": {"location": [0.05, -0.72, 0.9]}, "cam2": {"location": [0.45, -0.75, 0.8]}}
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
    
    # Track-to constraint to look at target
    track = c_obj.constraints.new(type="TRACK_TO")
    track.target = target_dummy
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam_objects[cam_id] = c_obj

bpy.context.view_layer.update()

# Save scene blend file
blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

# Export Ground Truth
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
    "platform": {
        "normal_world": [-0.10452846326765348, -0.17269691478056223, 0.9794128730990715],
        "d_offset": -0.12964223948680448,
        "corners_world": {"2": {"size_m": 0.0885, "location": [-0.13923306535155827, 0.2010219401037262, 0.15295315602159185], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "3": {"size_m": 0.0885, "location": [0.13923306535155827, 0.19593961049871544, 0.1817764795117942], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "4": {"size_m": 0.0885, "location": [0.13923306535155827, -0.0010219401037261794, 0.14704684397840814], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}, "5": {"size_m": 0.0885, "location": [-0.13923306535155827, 0.004060389501284581, 0.11822352048820578], "rotation_euler": [0.17453292519943295, -0.10471975511965978, 0.0]}},
        "width_m": 0.28,
        "height_m": 0.2
    },
    "cameras": {}
}

# Compute OpenCV Extrinsics for each camera
# Change of basis from Blender camera coords (Right, Up, Back) to OpenCV coords (Right, Down, Forward)
R_b_cv = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])

for cam_id, c_obj in cam_objects.items():
    # Blender camera world matrix: (4, 4)
    mw = np.array(c_obj.matrix_world)
    R_b_cam = mw[:3, :3]
    t_b_cam = mw[:3, 3]

    # Convert to OpenCV Camera Pose in World coordinates
    R_w_c = np.array(R_b_cam @ R_b_cv, dtype=float)
    t_w_c = np.array(t_b_cam, dtype=float)

    # OpenCV Extrinsics: X_c = R_c_w @ X_w + t_c_w
    R_c_w = R_w_c.T
    t_c_w = -R_c_w @ t_w_c

    # Rodrigues vector
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

    # Render Frame
    scene.camera = c_obj
    render_path = os.path.join(inputs_dir, f"{cam_id}_render.png")
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {render_path}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {gt_path}")
