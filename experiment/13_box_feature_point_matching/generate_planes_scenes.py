import bpy
import bpy_extras
import math
import os
import json
import mathutils

BASE_DIR = "/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/13_box_feature_point_matching"
SHARED_DIR = os.path.join(BASE_DIR, "shared")

# 15 Total Scene Configurations (Scenes 001 - 015)
# Zero occlusion, both image planes in every scene, diverse angles, lighting, distances, tilt, rolls
SCENE_CONFIGS = [
    # Initial 5 scenes
    {
        "id": "001",
        "description": "Frontal stereo baseline (12 deg), soft ambient daylight, both planes side-by-side",
        "cam0": {"loc": (-0.12, -0.85, 0.05), "roll": 0.0},
        "cam1": {"loc": (0.12, -0.85, 0.05), "roll": 0.0},
        "choco_plane": {"loc": (-0.18, 0.0, 0.0), "rot": (0.0, 0.0, 0.0)},
        "kelloggs_plane": {"loc": (0.16, 0.0, 0.0), "rot": (0.0, 0.0, 0.0)},
        "light": {"type": "SUN", "energy": 3.0, "loc": (1.0, -1.0, 2.5), "rot": (45, 15, 30), "ambient": 0.6}
    },
    {
        "id": "002",
        "description": "Medium asymmetric stereo baseline (24 deg), angled planes (toe-in / slant), standard diffuse light",
        "cam0": {"loc": (-0.24, -0.80, 0.10), "roll": 2.0},
        "cam1": {"loc": (0.18, -0.82, -0.04), "roll": -1.5},
        "choco_plane": {"loc": (-0.18, 0.03, 0.0), "rot": (4.0, -12.0, 3.0)},
        "kelloggs_plane": {"loc": (0.16, -0.03, 0.0), "rot": (-3.0, 15.0, -2.0)},
        "light": {"type": "SUN", "energy": 3.5, "loc": (-1.5, -1.0, 2.5), "rot": (35, -20, -45), "ambient": 0.5}
    },
    {
        "id": "003",
        "description": "Elevated pitch angle (top-down view ~26 deg), warm direct sun, zero occlusion",
        "cam0": {"loc": (-0.15, -0.76, 0.35), "roll": 0.0},
        "cam1": {"loc": (0.18, -0.76, 0.32), "roll": 0.0},
        "choco_plane": {"loc": (-0.17, 0.0, 0.0), "rot": (-15.0, 6.0, 0.0)},
        "kelloggs_plane": {"loc": (0.17, 0.0, 0.0), "rot": (-15.0, -6.0, 0.0)},
        "light": {"type": "SUN", "energy": 4.5, "loc": (0.5, -2.0, 3.0), "rot": (60, 0, 10), "ambient": 0.4}
    },
    {
        "id": "004",
        "description": "Wide baseline angle (~36 deg convergence), low-angle harsh directional side light",
        "cam0": {"loc": (-0.36, -0.72, 0.0), "roll": -3.0},
        "cam1": {"loc": (0.32, -0.75, 0.08), "roll": 4.0},
        "choco_plane": {"loc": (-0.19, 0.0, 0.0), "rot": (0.0, 10.0, -5.0)},
        "kelloggs_plane": {"loc": (0.17, 0.0, 0.0), "rot": (0.0, -10.0, 5.0)},
        "light": {"type": "SUN", "energy": 5.0, "loc": (-2.5, -0.3, 0.9), "rot": (20, -50, -80), "ambient": 0.3}
    },
    {
        "id": "005",
        "description": "Low ambient / dim lighting with sharp specular point glare, slight camera roll",
        "cam0": {"loc": (-0.16, -0.88, -0.06), "roll": 3.0},
        "cam1": {"loc": (0.19, -0.86, -0.03), "roll": -3.0},
        "choco_plane": {"loc": (-0.18, 0.02, 0.0), "rot": (-5.0, -8.0, 4.0)},
        "kelloggs_plane": {"loc": (0.16, -0.02, 0.0), "rot": (6.0, 10.0, -3.0)},
        "light": {"type": "POINT", "energy": 140.0, "loc": (0.0, -0.5, 0.3), "rot": (0, 0, 0), "ambient": 0.25}
    },
    # 10 NEW EXPERIMENTS (006 - 015)
    {
        "id": "006",
        "description": "Extreme yaw rotation (planes rotated 28 deg), narrow stereo baseline",
        "cam0": {"loc": (-0.08, -0.82, 0.02), "roll": 0.0},
        "cam1": {"loc": (0.10, -0.82, 0.02), "roll": 0.0},
        "choco_plane": {"loc": (-0.17, 0.04, 0.0), "rot": (0.0, 28.0, 0.0)},
        "kelloggs_plane": {"loc": (0.16, -0.04, 0.0), "rot": (0.0, -28.0, 0.0)},
        "light": {"type": "SUN", "energy": 3.2, "loc": (1.5, -1.2, 2.2), "rot": (40, 20, 20), "ambient": 0.55}
    },
    {
        "id": "007",
        "description": "Substantial height disparity (cam0 looking upward from desk, cam1 looking downward)",
        "cam0": {"loc": (-0.15, -0.78, -0.16), "roll": -2.0},
        "cam1": {"loc": (0.18, -0.76, 0.22), "roll": 2.0},
        "choco_plane": {"loc": (-0.18, 0.0, 0.0), "rot": (8.0, -6.0, 0.0)},
        "kelloggs_plane": {"loc": (0.16, 0.0, 0.0), "rot": (-6.0, 8.0, 0.0)},
        "light": {"type": "SUN", "energy": 4.0, "loc": (-1.0, -1.8, 2.8), "rot": (55, -10, -20), "ambient": 0.5}
    },
    {
        "id": "008",
        "description": "Close-up zoom distance (0.58m), maximum pixel resolution on package details",
        "cam0": {"loc": (-0.11, -0.58, 0.02), "roll": 0.0},
        "cam1": {"loc": (0.11, -0.58, 0.02), "roll": 0.0},
        "choco_plane": {"loc": (-0.16, 0.0, 0.0), "rot": (2.0, -5.0, 2.0)},
        "kelloggs_plane": {"loc": (0.15, 0.0, 0.0), "rot": (-2.0, 5.0, -2.0)},
        "light": {"type": "SUN", "energy": 3.5, "loc": (0.8, -1.0, 2.0), "rot": (45, 10, 15), "ambient": 0.6}
    },
    {
        "id": "009",
        "description": "Far distance (1.25m), smaller packaging footprint in image plane",
        "cam0": {"loc": (-0.20, -1.25, 0.08), "roll": 0.0},
        "cam1": {"loc": (0.22, -1.22, 0.06), "roll": 0.0},
        "choco_plane": {"loc": (-0.19, 0.0, 0.0), "rot": (0.0, 10.0, 0.0)},
        "kelloggs_plane": {"loc": (0.18, 0.0, 0.0), "rot": (0.0, -10.0, 0.0)},
        "light": {"type": "SUN", "energy": 3.0, "loc": (0.0, -2.0, 2.5), "rot": (50, 0, 0), "ambient": 0.65}
    },
    {
        "id": "010",
        "description": "Significant in-plane optical roll (Cam 0 rolled +15 deg, Cam 1 rolled -14 deg)",
        "cam0": {"loc": (-0.18, -0.80, 0.04), "roll": 15.0},
        "cam1": {"loc": (0.18, -0.78, 0.04), "roll": -14.0},
        "choco_plane": {"loc": (-0.18, 0.0, 0.0), "rot": (5.0, 5.0, 10.0)},
        "kelloggs_plane": {"loc": (0.16, 0.0, 0.0), "rot": (-5.0, -5.0, -8.0)},
        "light": {"type": "SUN", "energy": 3.8, "loc": (1.2, -1.5, 2.4), "rot": (45, 20, 20), "ambient": 0.5}
    },
    {
        "id": "011",
        "description": "Strong dual-source cross-lighting with dark ambient, casting opposing hard shadows",
        "cam0": {"loc": (-0.16, -0.82, 0.05), "roll": 0.0},
        "cam1": {"loc": (0.16, -0.82, 0.05), "roll": 0.0},
        "choco_plane": {"loc": (-0.18, 0.0, 0.0), "rot": (0.0, 12.0, 0.0)},
        "kelloggs_plane": {"loc": (0.16, 0.0, 0.0), "rot": (0.0, -12.0, 0.0)},
        "light": {"type": "POINT", "energy": 220.0, "loc": (-0.6, -0.5, 0.4), "rot": (0, 0, 0), "ambient": 0.15}
    },
    {
        "id": "012",
        "description": "Asymmetric focal positioning: Cam 0 centered on Kellogg's, Cam 1 centered on Choco Pie",
        "cam0": {"loc": (0.14, -0.80, 0.02), "roll": 1.0},
        "cam1": {"loc": (-0.14, -0.80, 0.02), "roll": -1.0},
        "choco_plane": {"loc": (-0.19, 0.0, 0.0), "rot": (3.0, -10.0, 0.0)},
        "kelloggs_plane": {"loc": (0.17, 0.0, 0.0), "rot": (-3.0, 10.0, 0.0)},
        "light": {"type": "SUN", "energy": 3.5, "loc": (0.5, -1.5, 2.2), "rot": (40, 10, 15), "ambient": 0.55}
    },
    {
        "id": "013",
        "description": "Grazing incidence angle (both planes yawed 34 deg away from camera axis)",
        "cam0": {"loc": (-0.22, -0.74, 0.06), "roll": 0.0},
        "cam1": {"loc": (0.22, -0.74, 0.06), "roll": 0.0},
        "choco_plane": {"loc": (-0.18, 0.06, 0.0), "rot": (0.0, 34.0, 0.0)},
        "kelloggs_plane": {"loc": (0.16, -0.06, 0.0), "rot": (0.0, -34.0, 0.0)},
        "light": {"type": "SUN", "energy": 4.2, "loc": (-2.0, -0.8, 1.5), "rot": (30, -40, -60), "ambient": 0.4}
    },
    {
        "id": "014",
        "description": "Very wide convergence baseline (44 deg), steep perspective disparity between views",
        "cam0": {"loc": (-0.42, -0.68, 0.08), "roll": -4.0},
        "cam1": {"loc": (0.38, -0.70, -0.04), "roll": 5.0},
        "choco_plane": {"loc": (-0.18, 0.0, 0.0), "rot": (-8.0, 12.0, -5.0)},
        "kelloggs_plane": {"loc": (0.16, 0.0, 0.0), "rot": (8.0, -12.0, 5.0)},
        "light": {"type": "SUN", "energy": 4.0, "loc": (1.8, -1.0, 2.6), "rot": (50, 25, 30), "ambient": 0.45}
    },
    {
        "id": "015",
        "description": "High pitch steep overhead perspective (~38 deg pitch down) under bright diffused daylight",
        "cam0": {"loc": (-0.16, -0.65, 0.52), "roll": 2.0},
        "cam1": {"loc": (0.18, -0.65, 0.48), "roll": -2.0},
        "choco_plane": {"loc": (-0.17, 0.0, 0.0), "rot": (-20.0, 8.0, 0.0)},
        "kelloggs_plane": {"loc": (0.17, 0.0, 0.0), "rot": (-20.0, -8.0, 0.0)},
        "light": {"type": "SUN", "energy": 4.8, "loc": (0.2, -1.5, 3.2), "rot": (65, 0, 5), "ambient": 0.5}
    }
]

PLANE_DIMENSIONS = {
    "choco_pie": (0.24, 0.24 / 1.1657),
    "kelloggs": (0.14, 0.14 / 0.5695)
}

PLANE_TARGET_POINTS = {
    "choco_pie": [
        {"name": "choco_lotte_logo", "uv": [0.187, 0.850], "desc": "Center of 'LOTTE' logo"},
        {"name": "choco_letter_C", "uv": [0.270, 0.620], "desc": "Capital letter 'C' in 'Choco'"},
        {"name": "choco_letter_P", "uv": [0.240, 0.380], "desc": "Capital letter 'P' in 'Pie'"},
        {"name": "choco_marshmallow", "uv": [0.630, 0.280], "desc": "White marshmallow filling center"},
        {"name": "choco_badge_12packs", "uv": [0.730, 0.130], "desc": "Gold '12 PACKS' badge text"}
    ],
    "kelloggs": [
        {"name": "kelloggs_letter_K", "uv": [0.450, 0.800], "desc": "Capital 'K' in red Kellogg's logo"},
        {"name": "kelloggs_veg_symbol", "uv": [0.900, 0.880], "desc": "Green vegetarian indicator square dot"},
        {"name": "kelloggs_corn_C", "uv": [0.740, 0.520], "desc": "Bold black letter 'C' in CORN"},
        {"name": "kelloggs_milk_splash", "uv": [0.480, 0.380], "desc": "Upper central splash of white milk"},
        {"name": "kelloggs_power_badge", "uv": [0.820, 0.320], "desc": "Top center of red POWER BREAKFAST box"}
    ]
}

def create_image_plane(name, img_type, transform_cfg):
    w, h = PLANE_DIMENSIONS[img_type]
    hw, hh = w / 2.0, h / 2.0
    verts = [(-hw, 0.0, -hh), (hw, 0.0, -hh), (hw, 0.0, hh), (-hw, 0.0, hh)]
    faces = [(0, 1, 2, 3)]
    
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    mat = bpy.data.materials.new(f"{name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    tex_path = os.path.join(SHARED_DIR, f"{img_type}.png")
    tex_image = bpy.data.images.load(tex_path, check_existing=True)
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = tex_image
    tex_node.interpolation = "Smart"
    
    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_node.inputs["Roughness"].default_value = 0.35
    bsdf_node.inputs["Specular IOR Level"].default_value = 0.5
    
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    mat.node_tree.links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])
    
    mesh.materials.append(mat)
    
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for loop_idx, uv in zip(mesh.polygons[0].loop_indices, uv_coords):
        uv_layer.data[loop_idx].uv = uv
        
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = transform_cfg["loc"]
    rot = transform_cfg["rot"]
    obj.rotation_euler = (math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2]))
    return obj

def setup_scene(cfg):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    
    scene = bpy.context.scene
    available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    
    world = bpy.data.worlds.new(f"World_{cfg['id']}")
    world.use_nodes = True
    bg_strength = cfg["light"].get("ambient", 0.5)
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.72, 0.75, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = bg_strength
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, -0.15))
    table = bpy.context.active_object
    table.name = "TableFloor"
    t_mat = bpy.data.materials.new("TableMat")
    t_mat.use_nodes = True
    t_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.22, 0.24, 0.26, 1.0)
    t_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.4
    table.data.materials.append(t_mat)

    l_cfg = cfg["light"]
    l_type = l_cfg["type"]
    l_data = bpy.data.lights.new(name="MainLight", type=l_type)
    l_data.energy = l_cfg["energy"]
    if l_type == "POINT":
        l_data.shadow_soft_size = 0.05
    l_obj = bpy.data.objects.new("MainLight", l_data)
    scene.collection.objects.link(l_obj)
    l_obj.location = l_cfg["loc"]
    rot_deg = l_cfg["rot"]
    l_obj.rotation_euler = (math.radians(rot_deg[0]), math.radians(rot_deg[1]), math.radians(rot_deg[2]))

    choco_obj = create_image_plane("ChocoPiePlane", "choco_pie", cfg["choco_plane"])
    kelloggs_obj = create_image_plane("KelloggsPlane", "kelloggs", cfg["kelloggs_plane"])

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    target_anchor = bpy.context.active_object
    target_anchor.name = "SceneCenterAnchor"

    bpy.context.view_layer.update()

    def create_camera(name, cam_cfg):
        c_data = bpy.data.cameras.new(name)
        fx_px = 950.0
        c_data.lens = fx_px * 36.0 / 1280.0
        c_data.sensor_width = 36.0
        c_data.sensor_fit = "HORIZONTAL"
        c_data.clip_start = 0.05
        c_data.clip_end = 50.0
        c_obj = bpy.data.objects.new(name, c_data)
        scene.collection.objects.link(c_obj)
        c_obj.location = cam_cfg["loc"]
        
        track = c_obj.constraints.new(type='TRACK_TO')
        track.target = target_anchor
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        if cam_cfg.get("roll", 0.0) != 0.0:
            c_obj.rotation_euler.rotate_axis('Z', math.radians(cam_cfg["roll"]))
            
        return c_obj

    cam0 = create_camera("Cam0", cfg["cam0"])
    cam1 = create_camera("Cam1", cfg["cam1"])
    
    bpy.context.view_layer.update()
    return choco_obj, kelloggs_obj, cam0, cam1

def project_point_to_camera_px(scene, cam_obj, world_point):
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, world_point)
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    u_px = co_2d.x * width
    v_px = (1.0 - co_2d.y) * height
    is_visible = (0.0 <= co_2d.x <= 1.0) and (0.0 <= co_2d.y <= 1.0) and (co_2d.z > 0)
    return round(float(u_px), 2), round(float(v_px), 2), bool(is_visible)

def run_generation():
    # Only render scenes 006 through 015 (the 10 new experiments)
    for cfg in SCENE_CONFIGS[5:]:
        s_id = cfg["id"]
        scene_dir = os.path.join(BASE_DIR, s_id)
        inp_dir = os.path.join(scene_dir, "inputs")
        art_dir = os.path.join(scene_dir, "artifacts")
        out_dir = os.path.join(scene_dir, "outputs")
        os.makedirs(inp_dir, exist_ok=True)
        os.makedirs(art_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        
        choco_obj, kelloggs_obj, cam0, cam1 = setup_scene(cfg)
        scene = bpy.context.scene
        
        blend_path = os.path.join(art_dir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        
        scene.camera = cam0
        cam0_path = os.path.join(inp_dir, "cam0_render.png")
        scene.render.filepath = cam0_path
        bpy.ops.render.render(write_still=True)
        
        scene.camera = cam1
        cam1_path = os.path.join(inp_dir, "cam1_render.png")
        scene.render.filepath = cam1_path
        bpy.ops.render.render(write_still=True)
        
        all_gt_points = []
        for plane_name, plane_obj in [("choco_pie", choco_obj), ("kelloggs", kelloggs_obj)]:
            w, h = PLANE_DIMENSIONS[plane_name]
            targets = PLANE_TARGET_POINTS[plane_name]
            
            for pt in targets:
                u_norm, v_norm = pt["uv"]
                local_pt = mathutils.Vector(((u_norm - 0.5) * w, 0.0, (v_norm - 0.5) * h))
                world_pt = plane_obj.matrix_world @ local_pt
                
                u0, v0, vis0 = project_point_to_camera_px(scene, cam0, world_pt)
                u1, v1, vis1 = project_point_to_camera_px(scene, cam1, world_pt)
                
                all_gt_points.append({
                    "name": pt["name"],
                    "plane": plane_name,
                    "description": pt["desc"],
                    "uv_norm": [round(u_norm, 4), round(v_norm, 4)],
                    "world_pos_m": [round(float(world_pt.x), 4), round(float(world_pt.y), 4), round(float(world_pt.z), 4)],
                    "cam0_px": [u0, v0],
                    "cam1_px": [u1, v1],
                    "visible_cam0": vis0,
                    "visible_cam1": vis1
                })
                
        gt_data = {
            "scene_id": s_id,
            "description": cfg["description"],
            "image_planes": ["choco_pie", "kelloggs"],
            "image_resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "focal_length_px": 950.0,
            "points": all_gt_points,
            "cam0_world_matrix": [list(row) for row in cam0.matrix_world],
            "cam1_world_matrix": [list(row) for row in cam1.matrix_world],
            "choco_world_matrix": [list(row) for row in choco_obj.matrix_world],
            "kelloggs_world_matrix": [list(row) for row in kelloggs_obj.matrix_world]
        }
        
        gt_path = os.path.join(inp_dir, "ground_truth.json")
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)
            
        spec_path = os.path.join(inp_dir, "scene_spec.json")
        with open(spec_path, "w") as f:
            json.dump(cfg, f, indent=2)
            
        # Copy build_scene.py into artifacts
        with open(os.path.join(art_dir, "build_scene.py"), "w") as f:
            f.write("# Generated from generate_planes_scenes.py for scene " + s_id + "\n")

        print(f"Scene {s_id} rendered with both image planes and ground truth exported.")

if __name__ == "__main__":
    run_generation()
