import bpy
import bmesh
import math
from mathutils import Vector, Matrix
import os

print("--- Starting ASL Communicator Production CAD Generation (Optimized) ---")

# Reset
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.001  # 1 BU = 1 mm
scene.unit_settings.length_unit = 'MILLIMETERS'

output_dir = r"d:\School\CAPSTONE\G12\ASL\cad"
os.makedirs(output_dir, exist_ok=True)

# Collections
col_components = bpy.data.collections.new("1. Hardware Mockups")
col_enclosure = bpy.data.collections.new("2. 3D Printable Enclosure")
scene.collection.children.link(col_components)
scene.collection.children.link(col_enclosure)

# Materials
def get_mat(name, color, roughness=0.35, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.diffuse_color = color
        if hasattr(mat, "roughness"):
            mat.roughness = roughness
        if hasattr(mat, "metallic"):
            mat.metallic = metallic
    return mat

mat_front_shell = get_mat("Mat_Front_MatteBlack", (0.12, 0.13, 0.15, 1.0), roughness=0.3)
mat_rear_shell  = get_mat("Mat_Rear_DarkGrey",   (0.20, 0.22, 0.25, 1.0), roughness=0.35)
mat_pcb_green   = get_mat("Mat_PCB_Pi_Green",    (0.04, 0.38, 0.12, 1.0), roughness=0.5)
mat_pcb_orange  = get_mat("Mat_PCB_ESP_Orange",  (0.85, 0.42, 0.05, 1.0), roughness=0.5)
mat_metal       = get_mat("Mat_Metal_Ports",     (0.80, 0.82, 0.85, 1.0), roughness=0.2, metallic=0.9)
mat_screen      = get_mat("Mat_Screen_Glass",    (0.02, 0.03, 0.05, 1.0), roughness=0.1)
mat_speaker     = get_mat("Mat_Speaker_Cone",    (0.15, 0.15, 0.15, 1.0), roughness=0.6)
mat_powerbank   = get_mat("Mat_PowerBank_Blue",  (0.08, 0.35, 0.65, 1.0), roughness=0.4)

def add_to_col(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)

def create_box(name, size, loc, mat=None, col=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if col:
        add_to_col(obj, col)
    return obj

def create_cyl(name, radius, depth, loc, rot=(0,0,0), mat=None, col=None, vertices=24):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=loc, rotation=rot, vertices=vertices)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if col:
        add_to_col(obj, col)
    return obj

def apply_bool_fast(target, cutter, operation='DIFFERENCE'):
    mod = target.modifiers.new(name="BoolOp", type='BOOLEAN')
    mod.operation = operation
    mod.solver = 'MANIFOLD'
    mod.object = cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier="BoolOp")
    bpy.data.objects.remove(cutter, do_unlink=True)

# Dimensions
ENCL_W = 82.0      # Overall Width
ENCL_H = 162.0     # Overall Height
ENCL_DEPTH = 32.0  # 16mm Front + 16mm Rear
WALL = 2.4         # Sturdy FDM wall
CORNER_R = 8.0     # Phone fillet

# 1. HARDWARE MOCKUPS
print("1/5 Creating hardware mockups...")
pi_z = -5.0
pi_pcb = create_box("Pi2B_PCB", (85.0, 56.0, 1.6), (0, -2.0, pi_z), mat_pcb_green, col_components)
usb_mock = create_box("Pi2B_USB_Ports", (17.5, 31.0, 16.0), (38.0, 3.5, pi_z + 8.8), mat_metal, col_components)
eth_mock = create_box("Pi2B_Ethernet", (21.5, 16.0, 13.5), (38.0, -19.5, pi_z + 7.5), mat_metal, col_components)
hdmi_mock = create_box("Pi2B_HDMI", (15.5, 11.5, 6.5), (-10.5, -27.5, pi_z + 4.0), mat_metal, col_components)
audio_mock = create_cyl("Pi2B_Audio_Jack", 3.5, 14.0, (11.0, -28.0, pi_z + 4.0), rot=(math.radians(90),0,0), mat=mat_metal, col=col_components)
pwr_mock = create_box("Pi2B_MicroUSB_Pwr", (8.0, 6.0, 3.2), (-32.0, -28.0, pi_z + 2.4), mat_metal, col_components)
sd_mock = create_box("Pi2B_MicroSD", (12.0, 15.0, 1.2), (-40.0, -2.0, pi_z - 1.4), mat_metal, col_components)

esp_z = 9.5
esp_pcb = create_box("ESP32_CYD_PCB", (50.0, 86.0, 1.6), (0, 0, esp_z), mat_pcb_orange, col_components)
esp_glass = create_box("ESP32_Screen_Glass", (43.2, 57.6, 2.5), (0, 0, esp_z + 2.0), mat_screen, col_components)

spk_top = create_cyl("Speaker_40mm_Top", 20.0, 4.5, (0, 57.0, esp_z + 1.0), mat=mat_speaker, col=col_components)
spk_bot = create_cyl("Speaker_40mm_Bottom", 20.0, 4.5, (0, -57.0, esp_z + 1.0), mat=mat_speaker, col=col_components)

cam_z = -12.0
cam_pcb = create_box("Camera_PCB", (25.0, 24.0, 1.2), (0, 50.0, cam_z), mat_pcb_green, col_components)
cam_lens = create_cyl("Camera_Lens", 4.0, 6.0, (0, 50.0, cam_z - 3.5), mat=mat_metal, col=col_components)

pb_mock = create_box("PowerBank_External", (65.0, 95.0, 14.0), (0, -20.0, -25.5), mat_powerbank, col_components)

# 2. FRONT SHELL
print("2/5 Generating Front Shell with speaker grilles and screen bezel...")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 8.0))
front_shell = bpy.context.active_object
front_shell.name = "ASL_Front_Shell"
front_shell.scale = (ENCL_W, ENCL_H, 16.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mod_bev = front_shell.modifiers.new("CornerBevel", 'BEVEL')
mod_bev.width = CORNER_R
mod_bev.segments = 8
mod_bev.limit_method = 'ANGLE'
mod_bev.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CornerBevel")

# Hollow cavity
front_cavity = create_box("Front_Cavity", (ENCL_W - 2*WALL, ENCL_H - 2*WALL, 16.0 - WALL), (0, 0, 8.0 - WALL/2))
mod_cav_bev = front_cavity.modifiers.new("CavBevel", 'BEVEL')
mod_cav_bev.width = max(1.0, CORNER_R - WALL)
mod_cav_bev.segments = 8
mod_cav_bev.limit_method = 'ANGLE'
mod_cav_bev.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CavBevel")
apply_bool_fast(front_shell, front_cavity)

# Screen window opening & recessed ledge
screen_window = create_box("Cut_ScreenWindow", (44.5, 59.0, 10.0), (0, 0, 16.0))
apply_bool_fast(front_shell, screen_window)

screen_recess = create_box("Cut_ScreenRecess", (51.0, 87.0, 3.5), (0, 0, 14.5))
apply_bool_fast(front_shell, screen_recess)

# Combined acoustic speaker grilles
def build_combined_grille(cy_list):
    bm = bmesh.new()
    for cy in cy_list:
        rings = [(5.5, 8), (10.5, 12), (15.5, 16)]
        for r, cnt in rings:
            for i in range(cnt):
                ang = (2 * math.pi / cnt) * i
                hx = r * math.cos(ang)
                hy = cy + r * math.sin(ang)
                # create cylinder in bmesh
                trans = Matrix.Translation((hx, hy, 16.0))
                bmesh.ops.create_cone(
                    bm,
                    cap_ends=True,
                    cap_tris=False,
                    segments=12,
                    radius1=1.3,
                    radius2=1.3,
                    depth=10.0,
                    matrix=trans
                )
    me = bpy.data.meshes.new("GrilleCutterMesh")
    bm.to_mesh(me)
    bm.free()
    grille_obj = bpy.data.objects.new("CombinedGrilleCutter", me)
    scene.collection.objects.link(grille_obj)
    return grille_obj

grille_cutter = build_combined_grille([57.0, -57.0])
apply_bool_fast(front_shell, grille_cutter)

# Internal speaker retention rings
for sy in [57.0, -57.0]:
    ring_outer = create_cyl(f"SpkRing_Out_{sy}", 21.5, 5.0, (0, sy, 11.5))
    ring_inner = create_cyl(f"SpkRing_In_{sy}", 20.1, 7.0, (0, sy, 11.5))
    apply_bool_fast(ring_outer, ring_inner)
    mod_j = front_shell.modifiers.new("JoinRing", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = ring_outer
    bpy.context.view_layer.objects.active = front_shell
    bpy.ops.object.modifier_apply(modifier="JoinRing")
    bpy.data.objects.remove(ring_outer, do_unlink=True)

# Front corner screw bosses
screw_corners = [
    (ENCL_W/2 - 6.0,  ENCL_H/2 - 6.0),
    (-ENCL_W/2 + 6.0,  ENCL_H/2 - 6.0),
    (ENCL_W/2 - 6.0, -ENCL_H/2 + 6.0),
    (-ENCL_W/2 + 6.0, -ENCL_H/2 + 6.0),
]
for idx, (bx, by) in enumerate(screw_corners):
    boss = create_cyl(f"Boss_F_{idx}", 4.0, 13.0, (bx, by, 7.5))
    hole = create_cyl(f"BossHole_F_{idx}", 1.5, 16.0, (bx, by, 7.5))
    apply_bool_fast(boss, hole)
    mod_j = front_shell.modifiers.new(f"JoinBoss_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = boss
    bpy.context.view_layer.objects.active = front_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinBoss_{idx}")
    bpy.data.objects.remove(boss, do_unlink=True)

front_shell.data.materials.append(mat_front_shell)
add_to_col(front_shell, col_enclosure)

# 3. REAR SHELL
print("3/5 Generating Rear Shell with camera mount, Pi standoffs & strap loops...")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -8.0))
rear_shell = bpy.context.active_object
rear_shell.name = "ASL_Rear_Shell"
rear_shell.scale = (ENCL_W, ENCL_H, 16.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mod_bev_r = rear_shell.modifiers.new("CornerBevel_R", 'BEVEL')
mod_bev_r.width = CORNER_R
mod_bev_r.segments = 8
mod_bev_r.limit_method = 'ANGLE'
mod_bev_r.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CornerBevel_R")

rear_cavity = create_box("Rear_Cavity", (ENCL_W - 2*WALL, ENCL_H - 2*WALL, 16.0 - WALL), (0, 0, -8.0 + WALL/2))
mod_cav_r = rear_cavity.modifiers.new("CavBevel_R", 'BEVEL')
mod_cav_r.width = max(1.0, CORNER_R - WALL)
mod_cav_r.segments = 8
mod_cav_r.limit_method = 'ANGLE'
mod_cav_r.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CavBevel_R")
apply_bool_fast(rear_shell, rear_cavity)

# Camera lens aperture & ribbon slot
cam_aperture = create_cyl("Cut_CamAperture", 5.5, 10.0, (0, 50.0, -16.0))
apply_bool_fast(rear_shell, cam_aperture)

cam_ribbon_slot = create_box("Cut_RibbonSlot", (18.0, 2.5, 10.0), (0, 35.0, -14.0))
apply_bool_fast(rear_shell, cam_ribbon_slot)

# Pi 2B Mounting Standoffs (4x M2.5)
pi_holes = [
    (-29.0, -2.0 - 24.5),
    ( 29.0, -2.0 - 24.5),
    (-29.0, -2.0 + 24.5),
    ( 29.0, -2.0 + 24.5),
]
for idx, (hx, hy) in enumerate(pi_holes):
    standoff = create_cyl(f"PiStandoff_{idx}", 3.2, 8.0, (hx, hy, -10.0))
    pilot = create_cyl(f"PiPilot_{idx}", 1.25, 10.0, (hx, hy, -10.0))
    apply_bool_fast(standoff, pilot)
    mod_j = rear_shell.modifiers.new(f"JoinPiBoss_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = standoff
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinPiBoss_{idx}")
    bpy.data.objects.remove(standoff, do_unlink=True)

# Port Cutouts
cut_usb_eth = create_box("Cut_USB_ETH", (15.0, 54.0, 18.0), (ENCL_W/2, -8.0, -4.5))
apply_bool_fast(rear_shell, cut_usb_eth)

cut_side_ports = create_box("Cut_SidePorts", (60.0, 15.0, 11.0), (-8.0, -ENCL_H/2 + 54.0, -2.5))
apply_bool_fast(rear_shell, cut_side_ports)

cut_sd = create_box("Cut_SD_Slot", (15.0, 5.0, 10.0), (-ENCL_W/2, -2.0, -6.5))
apply_bool_fast(rear_shell, cut_sd)

# Ventilation grill slots
bm_vents = bmesh.new()
for i in range(7):
    y_vent = 10.0 + i * 6.0
    trans = Matrix.Translation((-ENCL_W/2, y_vent, -5.0))
    bmesh.ops.create_cube(bm_vents, size=1.0, matrix=trans @ Matrix.Diagonal((12.0, 2.5, 8.0, 1.0)))
me_vents = bpy.data.meshes.new("VentCutterMesh")
bm_vents.to_mesh(me_vents)
bm_vents.free()
vents_cutter = bpy.data.objects.new("VentCutter", me_vents)
scene.collection.objects.link(vents_cutter)
apply_bool_fast(rear_shell, vents_cutter)

# External Power Bank Strap Brackets
for by in [-4.0, -38.0]:
    bracket = create_box(f"StrapBracket_{by}", (32.0, 8.0, 6.0), (0, by, -17.5))
    strap_pass = create_box(f"StrapPass_{by}", (23.0, 4.0, 8.0), (0, by, -17.5))
    apply_bool_fast(bracket, strap_pass)
    mod_j = rear_shell.modifiers.new(f"JoinBracket_{by}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = bracket
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinBracket_{by}")
    bpy.data.objects.remove(bracket, do_unlink=True)

# Rear corner screw bosses
for idx, (bx, by) in enumerate(screw_corners):
    boss_r = create_cyl(f"Boss_R_{idx}", 4.0, 13.0, (bx, by, -7.5))
    hole_r = create_cyl(f"BossHole_R_{idx}", 1.5, 16.0, (bx, by, -7.5))
    apply_bool_fast(boss_r, hole_r)
    mod_j = rear_shell.modifiers.new(f"JoinBoss_R_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = boss_r
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinBoss_R_{idx}")
    bpy.data.objects.remove(boss_r, do_unlink=True)

# Angled Tabletop Bevel Foot
foot_cut = create_box("FootBevel", (ENCL_W + 10.0, 14.0, 14.0), (0, -ENCL_H/2, -14.0))
foot_cut.rotation_euler = (math.radians(-15), 0, 0)
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
apply_bool_fast(rear_shell, foot_cut)

rear_shell.data.materials.append(mat_rear_shell)
add_to_col(rear_shell, col_enclosure)

# 4. EXPORT FILES
print("4/5 Exporting project and STL files...")
blend_file = os.path.join(output_dir, "asl_communicator_model.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_file)

stl_front = os.path.join(output_dir, "asl_front_shell.stl")
stl_rear = os.path.join(output_dir, "asl_rear_shell.stl")

bpy.ops.object.select_all(action='DESELECT')
front_shell.select_set(True)
bpy.context.view_layer.objects.active = front_shell
bpy.ops.wm.stl_export(filepath=stl_front, export_selected_objects=True)

bpy.ops.object.select_all(action='DESELECT')
rear_shell.select_set(True)
bpy.context.view_layer.objects.active = rear_shell
bpy.ops.wm.stl_export(filepath=stl_rear, export_selected_objects=True)

# 5. RENDERS
print("5/5 Rendering multi-angle previews...")
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'MATCAP'
scene.display.shading.color_type = 'MATERIAL'
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.render.resolution_x = 1280
scene.render.resolution_y = 720

cam_data = bpy.data.cameras.new(name="RenderCam")
cam_obj = bpy.data.objects.new(name="RenderCam", object_data=cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

views = [
    ("render_front_partner_view.png", (130, -180, 150), (math.radians(58), 0, math.radians(35))),
    ("render_rear_signer_view.png", (-140, 170, -140), (math.radians(125), 0, math.radians(-140))),
    ("render_side_ports_view.png", (210, 0, 20), (math.radians(85), 0, math.radians(90)))
]

for name, loc, rot in views:
    cam_obj.location = loc
    cam_obj.rotation_euler = rot
    scene.render.filepath = os.path.join(output_dir, name)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {name}")

print("=== DONE! ===")
