import bpy
import math
import os

out_dir = "/tmp/test_coke_can"
os.makedirs(out_dir, exist_ok=True)
render_path = os.path.join(out_dir, "render_can.png")

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

# Cylinder can: radius 0.033, height 0.12
center = (0.0, 0.0, 0.06)
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.033, depth=0.12, location=center)
can = bpy.context.active_object
mat_c = bpy.data.materials.new("CanMat")
mat_c.use_nodes = True
mat_c.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1.0)
mat_c.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.5
mat_c.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.3
can.data.materials.append(mat_c)

# Add silver rim on top (cylinder)
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.031, depth=0.005, location=(0.0, 0.0, 0.12))
rim = bpy.context.active_object
mat_rim = bpy.data.materials.new("RimMat")
mat_rim.use_nodes = True
mat_rim.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
mat_rim.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.9
rim.data.materials.append(mat_rim)

# Sun Light
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
cam_obj.location = (0, -0.85, 0.25)

track = cam_obj.constraints.new(type='TRACK_TO')
track.target = can
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

scene.camera = cam_obj
bpy.ops.render.render(write_still=True)
print("Can rendered:", render_path)
