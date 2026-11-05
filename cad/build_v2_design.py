import bpy
import bmesh
import math
from mathutils import Vector, Matrix
import os

print("=== Starting ASL Communicator V2 - High-Detail Industrial CAD with Folding Handle ===")

# Reset
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.001  # 1 BU = 1 mm
scene.unit_settings.length_unit = 'MILLIMETERS'

output_dir = r"d:\School\CAPSTONE\G12\ASL\cad"
os.makedirs(output_dir, exist_ok=True)

# Collections
col_mockups = bpy.data.collections.new("1. Hardware Mockups")
col_front = bpy.data.collections.new("2. Front Shell")
col_rear = bpy.data.collections.new("3. Rear Shell")
col_handle = bpy.data.collections.new("4. Folding Handle")
scene.collection.children.link(col_mockups)
scene.collection.children.link(col_front)
scene.collection.children.link(col_rear)
scene.collection.children.link(col_handle)

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

mat_shell_dark = get_mat("Mat_Polymer_DarkGrey", (0.12, 0.13, 0.15, 1.0), roughness=0.3)
mat_accent     = get_mat("Mat_Accent_Teal",      (0.05, 0.55, 0.52, 1.0), roughness=0.25)
mat_handle     = get_mat("Mat_Rubber_Grip",      (0.08, 0.08, 0.09, 1.0), roughness=0.6)
mat_pcb_green  = get_mat("Mat_PCB_Pi_Green",     (0.04, 0.38, 0.12, 1.0), roughness=0.5)
mat_pcb_orange = get_mat("Mat_PCB_ESP_Orange",   (0.85, 0.42, 0.05, 1.0), roughness=0.5)
mat_metal      = get_mat("Mat_Metal_Silver",     (0.80, 0.82, 0.85, 1.0), roughness=0.2, metallic=0.9)
mat_screen     = get_mat("Mat_Screen_Glass",     (0.02, 0.03, 0.05, 1.0), roughness=0.1)
mat_speaker    = get_mat("Mat_Speaker_Cone",     (0.15, 0.15, 0.15, 1.0), roughness=0.7)

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

def apply_bool(target, cutter, operation='DIFFERENCE', solver='MANIFOLD'):
    mod = target.modifiers.new(name="BoolOp", type='BOOLEAN')
    mod.operation = operation
    mod.solver = solver
    mod.object = cutter
    bpy.context.view_layer.objects.active = target
    try:
        bpy.ops.object.modifier_apply(modifier="BoolOp")
    except Exception:
        mod.solver = 'EXACT'
        bpy.ops.object.modifier_apply(modifier="BoolOp")
    bpy.data.objects.remove(cutter, do_unlink=True)

# Main Enclosure Dimensions
ENCL_W = 84.0      # Width
ENCL_H = 164.0     # Height
ENCL_DEPTH = 34.0  # 17mm Front + 17mm Rear
WALL = 2.5         # Solid wall thickness
CORNER_R = 9.0     # Smooth phone fillet

# ==============================================================================
# 1. HARDWARE MOCKUPS
# ==============================================================================
print("1/5 Generating hardware mockups...")
pi_z = -5.0
create_box("Pi2B_PCB", (85.0, 56.0, 1.6), (0, -2.0, pi_z), mat_pcb_green, col_mockups)
create_box("Pi2B_USB_Ports", (17.5, 31.0, 16.0), (38.0, 3.5, pi_z + 8.8), mat_metal, col_mockups)
create_box("Pi2B_Ethernet", (21.5, 16.0, 13.5), (38.0, -19.5, pi_z + 7.5), mat_metal, col_mockups)
create_box("Pi2B_HDMI", (15.5, 11.5, 6.5), (-10.5, -27.5, pi_z + 4.0), mat_metal, col_mockups)
create_cyl("Pi2B_Audio_Jack", 3.5, 14.0, (11.0, -28.0, pi_z + 4.0), rot=(math.radians(90),0,0), mat=mat_metal, col=col_mockups)
create_box("Pi2B_MicroUSB_Pwr", (8.0, 6.0, 3.2), (-32.0, -28.0, pi_z + 2.4), mat_metal, col_mockups)
create_box("Pi2B_MicroSD", (12.0, 15.0, 1.2), (-40.0, -2.0, pi_z - 1.4), mat_metal, col_mockups)

esp_z = 10.0
create_box("ESP32_CYD_PCB", (50.0, 86.0, 1.6), (0, 0, esp_z), mat_pcb_orange, col_mockups)
create_box("ESP32_Screen_Glass", (43.2, 57.6, 2.5), (0, 0, esp_z + 2.0), mat_screen, col_mockups)

create_cyl("Speaker_40mm_Top", 20.0, 4.5, (0, 58.0, esp_z + 1.0), mat=mat_speaker, col=col_mockups)
create_cyl("Speaker_40mm_Bottom", 20.0, 4.5, (0, -58.0, esp_z + 1.0), mat=mat_speaker, col=col_mockups)

cam_z = -12.5
create_box("Camera_PCB", (25.0, 24.0, 1.2), (0, 52.0, cam_z), mat_pcb_green, col_mockups)
create_cyl("Camera_Lens", 4.0, 6.0, (0, 52.0, cam_z - 3.5), mat=mat_metal, col=col_mockups)

# ==============================================================================
# 2. FRONT SHELL (Partner / Listener Facing)
# ==============================================================================
print("2/5 Generating High-Detail Front Shell...")

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 8.5))
front_shell = bpy.context.active_object
front_shell.name = "ASL_Front_Shell"
front_shell.scale = (ENCL_W, ENCL_H, 17.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Rounded corners
mod_bev = front_shell.modifiers.new("CornerBevel", 'BEVEL')
mod_bev.width = CORNER_R
mod_bev.segments = 8
mod_bev.limit_method = 'ANGLE'
mod_bev.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CornerBevel")

# Cavity
cavity_f = create_box("Cavity_F", (ENCL_W - 2*WALL, ENCL_H - 2*WALL, 17.0 - WALL), (0, 0, 8.5 - WALL/2))
mod_cav_bev = cavity_f.modifiers.new("CavBevel", 'BEVEL')
mod_cav_bev.width = max(1.0, CORNER_R - WALL)
mod_cav_bev.segments = 8
mod_cav_bev.limit_method = 'ANGLE'
mod_cav_bev.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CavBevel")
apply_bool(front_shell, cavity_f)

# Screen Bezel Opening with angled chamfer hood
screen_cut = create_box("ScreenCut", (44.5, 59.0, 10.0), (0, 0, 17.0))
apply_bool(front_shell, screen_cut)

screen_shelf = create_box("ScreenShelf", (51.0, 87.0, 3.5), (0, 0, 15.5))
apply_bool(front_shell, screen_shelf)

# LED Status Light Hole
led_hole = create_cyl("LED_Status_Hole", 1.5, 8.0, (26.0, 32.0, 17.0))
apply_bool(front_shell, led_hole)

# Modern Hexagonal Acoustic Honeycomb Grille Function
def build_hex_grille(cy_list):
    bm = bmesh.new()
    spacing = 3.6
    for cy in cy_list:
        for row in range(-5, 6):
            for col in range(-5, 6):
                dist = math.sqrt((col * spacing)**2 + (row * spacing * 0.866)**2)
                if dist <= 18.5 and dist >= 1.0:
                    hx = col * spacing + (spacing/2 if row % 2 else 0)
                    hy = cy + row * spacing * 0.866
                    trans = Matrix.Translation((hx, hy, 17.0))
                    # Hexagonal prism
                    bmesh.ops.create_cone(
                        bm,
                        cap_ends=True,
                        cap_tris=False,
                        segments=6,
                        radius1=1.2,
                        radius2=1.2,
                        depth=8.0,
                        matrix=trans
                    )
    me = bpy.data.meshes.new("HexGrilleMesh")
    bm.to_mesh(me)
    bm.free()
    grille_obj = bpy.data.objects.new("HexGrilleCutter", me)
    scene.collection.objects.link(grille_obj)
    return grille_obj

hex_grille = build_hex_grille([58.0, -58.0])
apply_bool(front_shell, hex_grille)

# Side finger grips / knurled ribs
bm_grips = bmesh.new()
for side in [-1, 1]:
    for i in range(12):
        gy = -30.0 + i * 5.0
        gx = side * (ENCL_W / 2)
        trans = Matrix.Translation((gx, gy, 8.5))
        bmesh.ops.create_cube(bm_grips, size=1.0, matrix=trans @ Matrix.Diagonal((3.0, 2.2, 10.0, 1.0)))
me_grips = bpy.data.meshes.new("GripCutterMesh")
bm_grips.to_mesh(me_grips)
bm_grips.free()
grip_cutter = bpy.data.objects.new("GripCutter", me_grips)
scene.collection.objects.link(grip_cutter)
apply_bool(front_shell, grip_cutter)

# Internal speaker retention rings
for sy in [58.0, -58.0]:
    ring_out = create_cyl(f"SpkRing_Out_{sy}", 21.5, 5.0, (0, sy, 12.0))
    ring_in = create_cyl(f"SpkRing_In_{sy}", 20.1, 7.0, (0, sy, 12.0))
    apply_bool(ring_out, ring_in)
    mod_j = front_shell.modifiers.new("JoinRing", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = ring_out
    bpy.context.view_layer.objects.active = front_shell
    bpy.ops.object.modifier_apply(modifier="JoinRing")
    bpy.data.objects.remove(ring_out, do_unlink=True)

# 4 Corner M3 Screw Bosses
screw_corners = [
    ( ENCL_W/2 - 6.5,  ENCL_H/2 - 6.5),
    (-ENCL_W/2 + 6.5,  ENCL_H/2 - 6.5),
    ( ENCL_W/2 - 6.5, -ENCL_H/2 + 6.5),
    (-ENCL_W/2 + 6.5, -ENCL_H/2 + 6.5),
]
for idx, (bx, by) in enumerate(screw_corners):
    boss = create_cyl(f"Boss_F_{idx}", 4.2, 14.0, (bx, by, 8.0))
    hole = create_cyl(f"BossHole_F_{idx}", 1.5, 18.0, (bx, by, 8.0))
    apply_bool(boss, hole)
    mod_j = front_shell.modifiers.new(f"JoinBoss_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = boss
    bpy.context.view_layer.objects.active = front_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinBoss_{idx}")
    bpy.data.objects.remove(boss, do_unlink=True)

front_shell.data.materials.append(mat_shell_dark)
add_to_col(front_shell, col_front)

# ==============================================================================
# 3. REAR SHELL (Signer Facing with Handle Bay & Hinge)
# ==============================================================================
print("3/5 Generating High-Detail Rear Shell with Handle Bay...")

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -8.5))
rear_shell = bpy.context.active_object
rear_shell.name = "ASL_Rear_Shell"
rear_shell.scale = (ENCL_W, ENCL_H, 17.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mod_bev_r = rear_shell.modifiers.new("CornerBevel_R", 'BEVEL')
mod_bev_r.width = CORNER_R
mod_bev_r.segments = 8
mod_bev_r.limit_method = 'ANGLE'
mod_bev_r.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CornerBevel_R")

# Cavity
cavity_r = create_box("Cavity_R", (ENCL_W - 2*WALL, ENCL_H - 2*WALL, 17.0 - WALL), (0, 0, -8.5 + WALL/2))
mod_cav_r = cavity_r.modifiers.new("CavBevel_R", 'BEVEL')
mod_cav_r.width = max(1.0, CORNER_R - WALL)
mod_cav_r.segments = 8
mod_cav_r.limit_method = 'ANGLE'
mod_cav_r.angle_limit = math.radians(60)
bpy.ops.object.modifier_apply(modifier="CavBevel_R")
apply_bool(rear_shell, cavity_r)

# Camera Module Lens Aperture & Protective Bezel Ring
cam_cut = create_cyl("Cut_CamLens", 5.5, 10.0, (0, 52.0, -17.0))
apply_bool(rear_shell, cam_cut)

cam_ribbon = create_box("Cut_CamRibbon", (18.0, 2.5, 10.0), (0, 36.0, -15.0))
apply_bool(rear_shell, cam_ribbon)

# Camera protective hood ring on outside
cam_hood_out = create_cyl("CamHood_Out", 8.0, 2.5, (0, 52.0, -18.25))
cam_hood_in = create_cyl("CamHood_In", 5.6, 5.0, (0, 52.0, -18.25))
apply_bool(cam_hood_out, cam_hood_in)
mod_j = rear_shell.modifiers.new("JoinCamHood", 'BOOLEAN')
mod_j.operation = 'UNION'
mod_j.solver = 'MANIFOLD'
mod_j.object = cam_hood_out
bpy.context.view_layer.objects.active = rear_shell
bpy.ops.object.modifier_apply(modifier="JoinCamHood")
bpy.data.objects.remove(cam_hood_out, do_unlink=True)

# ------------------------------------------------------------------------------
# HANDLE STOWAGE BAY & HINGE BRACKETS
# ------------------------------------------------------------------------------
# Handle Recessed Bay: 26mm W x 105mm L x 14mm D along lower center of rear shell
BAY_W = 26.0
BAY_L = 105.0
BAY_D = 13.5
BAY_Y = -28.0

bay_cutter = create_box("Handle_Stowage_Bay", (BAY_W, BAY_L, BAY_D), (0, BAY_Y, -17.0 + BAY_D/2))
mod_bay_bev = bay_cutter.modifiers.new("BayBevel", 'BEVEL')
mod_bay_bev.width = 3.5
mod_bay_bev.segments = 6
bpy.ops.object.modifier_apply(modifier="BayBevel")
apply_bool(rear_shell, bay_cutter)

# Thumb nail pull-notch at top of bay
thumb_notch = create_cyl("ThumbNotch", 8.0, 8.0, (0, BAY_Y + BAY_L/2 + 2.0, -17.0))
apply_bool(rear_shell, thumb_notch)

# Integrated Dual-Shear Hinge Knuckles at the bottom of the bay (Y = -ENCL_H/2)
HINGE_Y = -ENCL_H/2 + 4.0
HINGE_Z = -13.0
for sign in [-1, 1]:
    knuckle_x = sign * 11.0
    knuckle = create_cyl(f"HingeKnuckle_{sign}", 5.5, 5.0, (knuckle_x, HINGE_Y, HINGE_Z), rot=(0, math.radians(90), 0))
    pin_hole = create_cyl(f"HingePinHole_{sign}", 1.6, 12.0, (knuckle_x, HINGE_Y, HINGE_Z), rot=(0, math.radians(90), 0))
    apply_bool(knuckle, pin_hole)
    mod_j = rear_shell.modifiers.new(f"JoinKnuckle_{sign}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = knuckle
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinKnuckle_{sign}")
    bpy.data.objects.remove(knuckle, do_unlink=True)

# Pi 2B Mounting Standoffs (4x M2.5)
pi_holes = [
    (-29.0, -2.0 - 24.5),
    ( 29.0, -2.0 - 24.5),
    (-29.0, -2.0 + 24.5),
    ( 29.0, -2.0 + 24.5),
]
for idx, (hx, hy) in enumerate(pi_holes):
    standoff = create_cyl(f"PiStandoff_{idx}", 3.2, 8.0, (hx, hy, -10.5))
    pilot = create_cyl(f"PiPilot_{idx}", 1.25, 10.0, (hx, hy, -10.5))
    apply_bool(standoff, pilot)
    mod_j = rear_shell.modifiers.new(f"JoinPiBoss_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = standoff
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinPiBoss_{idx}")
    bpy.data.objects.remove(standoff, do_unlink=True)

# Side Ports Cutouts (+X side: USB & Ethernet)
cut_usb_eth = create_box("Cut_USB_ETH", (15.0, 54.0, 18.0), (ENCL_W/2, -8.0, -5.0))
apply_bool(rear_shell, cut_usb_eth)

# Side Ports Cutouts (-Y side: HDMI, MicroUSB, Audio)
cut_side_ports = create_box("Cut_SidePorts", (60.0, 15.0, 11.0), (-8.0, -ENCL_H/2 + 54.0, -2.5))
apply_bool(rear_shell, cut_side_ports)

# MicroSD Slot
cut_sd = create_box("Cut_SD_Slot", (15.0, 5.0, 10.0), (-ENCL_W/2, -2.0, -7.0))
apply_bool(rear_shell, cut_sd)

# Passive Cooling Vents on side
bm_vents = bmesh.new()
for i in range(8):
    y_vent = 8.0 + i * 5.5
    trans = Matrix.Translation((-ENCL_W/2, y_vent, -5.5))
    bmesh.ops.create_cube(bm_vents, size=1.0, matrix=trans @ Matrix.Diagonal((12.0, 2.2, 8.0, 1.0)))
me_vents = bpy.data.meshes.new("VentMesh")
bm_vents.to_mesh(me_vents)
bm_vents.free()
vent_obj = bpy.data.objects.new("VentCutter", me_vents)
scene.collection.objects.link(vent_obj)
apply_bool(rear_shell, vent_obj)

# Rear Corner M3 Bosses
for idx, (bx, by) in enumerate(screw_corners):
    boss_r = create_cyl(f"Boss_R_{idx}", 4.2, 14.0, (bx, by, -8.0))
    hole_r = create_cyl(f"BossHole_R_{idx}", 1.5, 18.0, (bx, by, -8.0))
    apply_bool(boss_r, hole_r)
    mod_j = rear_shell.modifiers.new(f"JoinBoss_R_{idx}", 'BOOLEAN')
    mod_j.operation = 'UNION'
    mod_j.solver = 'MANIFOLD'
    mod_j.object = boss_r
    bpy.context.view_layer.objects.active = rear_shell
    bpy.ops.object.modifier_apply(modifier=f"JoinBoss_R_{idx}")
    bpy.data.objects.remove(boss_r, do_unlink=True)

rear_shell.data.materials.append(mat_shell_dark)
add_to_col(rear_shell, col_rear)

# ==============================================================================
# 4. ERGONOMIC FOLDING HANDLE (Standalone Component)
# ==============================================================================
print("4/5 Modeling Ergonomic Folding Handle...")

HDL_L = 98.0
HDL_W = 23.0
HDL_T = 12.0

# Base handle body
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, BAY_Y - 2.0, -17.0 + HDL_T/2))
handle = bpy.context.active_object
handle.name = "ASL_Ergonomic_Handle"
handle.scale = (HDL_W, HDL_L, HDL_T)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mod_hdl_bev = handle.modifiers.new("HandleBevel", 'BEVEL')
mod_hdl_bev.width = 4.0
mod_hdl_bev.segments = 8
bpy.ops.object.modifier_apply(modifier="HandleBevel")

# 3-finger sculpted ergonomic scallops
for idx, fy in enumerate([-18.0, 2.0, 22.0]):
    scallop = create_cyl(f"Scallop_{idx}", 11.0, 30.0, (0, BAY_Y + fy, -17.0 + HDL_T), rot=(0, math.radians(90), 0))
    apply_bool(handle, scallop)

# Lanyard / Wrist-strap loop hole at top of handle
lanyard_hole = create_cyl("LanyardHole", 2.5, 30.0, (0, BAY_Y + HDL_L/2 - 8.0, -17.0 + HDL_T/2), rot=(0, math.radians(90), 0))
apply_bool(handle, lanyard_hole)

# Handle Hinge Knuckle (Fits precisely between the rear shell hinge brackets)
hdl_hinge = create_cyl("HandleHingeKnuckle", 5.5, 14.0, (0, HINGE_Y, HINGE_Z), rot=(0, math.radians(90), 0))
hdl_pin = create_cyl("HandlePinHole", 1.6, 25.0, (0, HINGE_Y, HINGE_Z), rot=(0, math.radians(90), 0))
apply_bool(hdl_hinge, hdl_pin)

mod_j = handle.modifiers.new("JoinHandleHinge", 'BOOLEAN')
mod_j.operation = 'UNION'
mod_j.solver = 'MANIFOLD'
mod_j.object = hdl_hinge
bpy.context.view_layer.objects.active = handle
bpy.ops.object.modifier_apply(modifier="JoinHandleHinge")
bpy.data.objects.remove(hdl_hinge, do_unlink=True)

handle.data.materials.append(mat_handle)
add_to_col(handle, col_handle)

# ==============================================================================
# 5. EXPORT PRODUCTION STLs & BLENDER PROJECT
# ==============================================================================
print("5/5 Exporting STLs and saving Blender project...")

blend_file = os.path.join(output_dir, "asl_communicator_model.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_file)
print(f"Blender project saved: {blend_file}")

stl_front  = os.path.join(output_dir, "asl_front_shell.stl")
stl_rear   = os.path.join(output_dir, "asl_rear_shell.stl")
stl_handle = os.path.join(output_dir, "asl_ergonomic_handle.stl")

# Front
bpy.ops.object.select_all(action='DESELECT')
front_shell.select_set(True)
bpy.context.view_layer.objects.active = front_shell
bpy.ops.wm.stl_export(filepath=stl_front, export_selected_objects=True)
print(f"Exported: {stl_front}")

# Rear
bpy.ops.object.select_all(action='DESELECT')
rear_shell.select_set(True)
bpy.context.view_layer.objects.active = rear_shell
bpy.ops.wm.stl_export(filepath=stl_rear, export_selected_objects=True)
print(f"Exported: {stl_rear}")

# Handle
bpy.ops.object.select_all(action='DESELECT')
handle.select_set(True)
bpy.context.view_layer.objects.active = handle
bpy.ops.wm.stl_export(filepath=stl_handle, export_selected_objects=True)
print(f"Exported: {stl_handle}")

# ==============================================================================
# 6. MULTI-ANGLE CRISP RENDERS (WORKBENCH)
# ==============================================================================
print("Generating multi-angle preview renders...")

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

# Render 1: Front Partner View (Hex Grilles, Screen Bezel, Knurled Side Grips)
cam_obj.location = (130, -180, 150)
cam_obj.rotation_euler = (math.radians(58), 0, math.radians(35))
scene.render.filepath = os.path.join(output_dir, "render_v2_front_partner.png")
bpy.ops.render.render(write_still=True)
print("Rendered: render_v2_front_partner.png")

# Render 2: Rear Signer View with Handle Folded Flush in Bay
cam_obj.location = (-140, 170, -140)
cam_obj.rotation_euler = (math.radians(125), 0, math.radians(-140))
scene.render.filepath = os.path.join(output_dir, "render_v2_rear_folded_handle.png")
bpy.ops.render.render(write_still=True)
print("Rendered: render_v2_rear_folded_handle.png")

# Render 3: Deployed Handle State (Rotate handle 90 degrees out as a pistol grip!)
handle.rotation_euler = (math.radians(-90), 0, 0)
handle.location = (0, HINGE_Y - 2.0, HINGE_Z - HDL_L/2 + 10.0)
cam_obj.location = (-170, -140, -60)
cam_obj.rotation_euler = (math.radians(105), 0, math.radians(-125))
scene.render.filepath = os.path.join(output_dir, "render_v2_handle_deployed.png")
bpy.ops.render.render(write_still=True)
print("Rendered: render_v2_handle_deployed.png")

# Render 4: Side Profile & Ports
cam_obj.location = (210, 0, 20)
cam_obj.rotation_euler = (math.radians(85), 0, math.radians(90))
scene.render.filepath = os.path.join(output_dir, "render_v2_side_ports.png")
bpy.ops.render.render(write_still=True)
print("Rendered: render_v2_side_ports.png")

# Restore handle to folded state for saving
handle.rotation_euler = (0, 0, 0)
handle.location = (0, 0, 0)
bpy.ops.wm.save_as_mainfile(filepath=blend_file)

print("=== ALL V2 CAD ENGINEERING COMPLETED SUCCESSFULLY ===")
