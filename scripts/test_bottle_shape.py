import bpy
import math
import os

out_dir = "/tmp/test_coke_can"
os.makedirs(out_dir, exist_ok=True)
render_path = os.path.join(out_dir, "render_bottle.png")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1080
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = render_path
scene.view_settings.view_transform = "Standard"

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)
scene.world = world

# Floor
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("Floor")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.35, 1.0)
floor.data.materials.append(mat_f)

# Cylinder Body
dims = (0.07, 0.07, 0.22)
center = (0.0, 0.0, 0.11)
bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=dims[0]*0.46, depth=dims[2]*0.68, location=(0, 0, 0.11 - dims[2]*0.10))
body = bpy.context.active_object

bpy.ops.mesh.primitive_cone_add(vertices=40, radius1=dims[0]*0.46, radius2=dims[0]*0.22, depth=dims[2]*0.20, location=(0, 0, 0.11 + dims[2]*0.34))
shoulder = bpy.context.active_object

bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=dims[0]*0.22, depth=dims[2]*0.24, location=(0, 0, 0.11 + dims[2]*0.55))
neck = bpy.context.active_object

# Join into one object
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
shoulder.select_set(True)
neck.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
bottle = bpy.context.active_object

mat_b = bpy.data.materials.new("BottleMat")
mat_b.use_nodes = True
mat_b.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.6, 0.2, 1.0)
bottle.data.materials.append(mat_b)

# Light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2, -2, 4)

# Camera
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 35.0
cam_data.sensor_width = 36.0
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = (0, -0.9, 0.35)

track = cam_obj.constraints.new(type='TRACK_TO')
track.target = bottle
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

scene.camera = cam_obj
bpy.ops.render.render(write_still=True)
print("Bottle rendered:", render_path)
