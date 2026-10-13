import bpy
import math
import os
import json
import mathutils

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/can_triangulation/001/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/can_triangulation/001/inputs'
tex_path = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/can_triangulation/shared/can_label.png'
os.makedirs(art_dir, exist_ok=True)
os.makedirs(inp_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1080
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

# Ambient world lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.78, 0.78, 0.80, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Wooden table surface / floor
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("TableMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.22, 0.18, 1.0)
mat_f.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.4
floor.data.materials.append(mat_f)

# Can material
mat_can = bpy.data.materials.new("CanMaterial")
mat_can.use_nodes = True
nodes = mat_can.node_tree.nodes
tex = nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(tex_path, check_existing=True)
bsdf = nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.25
bsdf.inputs["Metallic"].default_value = 0.35
mat_can.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

# Top rim material (metallic silver)
mat_rim = bpy.data.materials.new("RimMaterial")
mat_rim.use_nodes = True
bsdf_rim = mat_rim.node_tree.nodes["Principled BSDF"]
bsdf_rim.inputs["Base Color"].default_value = (0.85, 0.85, 0.85, 1.0)
bsdf_rim.inputs["Metallic"].default_value = 0.95
bsdf_rim.inputs["Roughness"].default_value = 0.15

cans = [{'name': 'can_01', 'position': [-0.16, 0.0, 0.06], 'radius': 0.033, 'height': 0.12}, {'name': 'can_02', 'position': [0.0, 0.06, 0.06], 'radius': 0.033, 'height': 0.12}, {'name': 'can_03', 'position': [0.18, -0.04, 0.06], 'radius': 0.033, 'height': 0.12}]
can_objects = []

for c in cans:
    pos = c["position"]
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    # Base cylinder
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=height, location=(pos[0], pos[1], pos[2]))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat_can)
    
    # Top rim
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius*0.96, depth=0.005,
                                        location=(pos[0], pos[1], pos[2] + height/2.0))
    rim = bpy.context.active_object
    rim.name = name + "_rim"
    rim.data.materials.append(mat_rim)
    
    # Join into single object
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    rim.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    can_objects.append(bpy.context.active_object)

# Key Light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 4.5
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2.0, -2.5, 4.0)
light_obj.rotation_euler = (math.radians(40), math.radians(20), math.radians(35))

# Target for cameras (center of cans)
avg_x = sum(c["position"][0] for c in cans) / len(cans)
avg_y = sum(c["position"][1] for c in cans) / len(cans)
avg_z = sum(c["position"][2] for c in cans) / len(cans)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

def create_cam(name, loc):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = 935.3074360871938 * 36.0 / 1080
    c_data.sensor_width = 36.0
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 100.0
    c_obj = bpy.data.objects.new(name, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = loc
    dir_vec = target_loc - mathutils.Vector(loc)
    c_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
    return c_obj

cam0 = create_cam("Camera_0", (-0.28, -0.85, 0.28))
cam1 = create_cam("Camera_1", (0.28, -0.85, 0.28))

bpy.context.view_layer.update()

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

scene.camera = cam0
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

scene.camera = cam1
scene.render.filepath = os.path.join(inp_dir, "cam1_render.png")
bpy.ops.render.render(write_still=True)

with open(os.path.join(inp_dir, "ground_truth_cameras.json"), "w") as f:
    json.dump({
        "cam0_matrix_world": [list(row) for row in cam0.matrix_world],
        "cam1_matrix_world": [list(row) for row in cam1.matrix_world],
        "cam0_position": list(cam0.location),
        "cam1_position": list(cam1.location),
    }, f, indent=2)
