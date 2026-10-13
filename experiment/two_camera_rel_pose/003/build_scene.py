import bpy
import math
import os
import mathutils

out = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/two_camera_rel_pose/003'
tag_img_path = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/two_camera_rel_pose/_shared_tags/tag_0.png'
os.makedirs(out, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 480
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
h = 0.08 / 2.0
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
    c_data.lens = 500.0 * 36.0 / 640
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
cam0 = create_cam("Cam0", (-0.1912269759838671, 0.064552322654166, 0.9773656795309325), tag_obj)
cam1 = create_cam("Cam1", (0.17226490601049868, 0.010916957403166963, 0.8319086280520878), tag_obj)

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
    json.dump({
        "cam0_matrix_world": [list(row) for row in cam0.matrix_world],
        "cam1_matrix_world": [list(row) for row in cam1.matrix_world],
    }, f, indent=2)
