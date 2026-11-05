import bpy
import bmesh
import math
from mathutils import Vector, Matrix
import os

# Clear existing objects in the default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Set scene units to millimeters
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.001  # 1 Blender Unit = 1 mm
scene.unit_settings.length_unit = 'MILLIMETERS'

# Collections
def get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col

col_components = get_or_create_collection("1. Hardware Mockups (Reference)")
col_enclosure = get_or_create_collection("2. 3D Printable Enclosure")

# Materials
def create_mat(name, color, roughness=0.4, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Roughness'].default_value = roughness
            bsdf.inputs['Metallic'].default_value = metallic
    return mat

mat_shell_front = create_mat("Mat_Shell_Front", (0.12, 0.13, 0.15, 1.0), roughness=0.3)
mat_shell_rear = create_mat("Mat_Shell_Rear", (0.18, 0.20, 0.23, 1.0), roughness=0.35)
mat_pcb_green = create_mat("Mat_PCB_Green", (0.05, 0.35, 0.12, 1.0), roughness=0.5)
mat_pcb_orange = create_mat("Mat_PCB_Orange", (0.85, 0.40, 0.05, 1.0), roughness=0.5)
mat_metal = create_mat("Mat_Metal", (0.8, 0.82, 0.85, 1.0), roughness=0.25, metallic=0.9)
mat_glass = create_mat("Mat_Screen_Glass", (0.02, 0.04, 0.06, 1.0), roughness=0.1)
mat_speaker = create_mat("Mat_Speaker_Cone", (0.1, 0.1, 0.1, 1.0), roughness=0.7)
mat_powerbank = create_mat("Mat_PowerBank", (0.05, 0.4, 0.7, 1.0), roughness=0.4)

# Helper to create a box
def make_box(name, size, loc, mat=None, collection=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if collection:
        # Move to collection
        for c in obj.users_collection:
            c.objects.unlink(obj)
        collection.objects.link(obj)
    return obj

# Helper to create a cylinder
def make_cylinder(name, radius, depth, loc, rot=(0,0,0), mat=None, collection=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if collection:
        for c in obj.users_collection:
            c.objects.unlink(obj)
        collection.objects.link(obj)
    return obj

print("Building Hardware Mockups...")

# ==========================================
# 1. HARDWARE MOCKUPS
# ==========================================

# Pi 2B Mockup
# PCB: 85 x 56 x 1.6 mm. Placed inside behind the screen
pi_pcb = make_box("Pi2B_PCB", (85.0, 56.0, 1.6), (0, 0, -5), mat=mat_pcb_green, collection=col_components)
# USB & Ethernet ports
usb_ports = make_box("Pi2B_USB_Ports", (17.0, 30.0, 16.0), (38.0, 5.0, 3.5), mat=mat_metal, collection=col_components)
eth_port = make_box("Pi2B_Ethernet", (21.0, 16.0, 13.5), (38.0, -18.0, 2.5), mat=mat_metal, collection=col_components)
hdmi_port = make_box("Pi2B_HDMI", (15.0, 11.0, 6.5), (-10.5, -25.5, -1.5), mat=mat_metal, collection=col_components)
audio_jack = make_cylinder("Pi2B_Audio_Jack", 3.5, 14.0, (11.0, -26.0, -1.0), rot=(math.radians(90), 0, 0), mat=mat_metal, collection=col_components)
pwr_port = make_box("Pi2B_MicroUSB_Pwr", (8.0, 6.0, 3.2), (-32.0, -26.0, -2.5), mat=mat_metal, collection=col_components)

# ESP32 CYD 2.8" Screen Mockup
# Facing positive Z (towards the listener)
esp_pcb = make_box("ESP32_Screen_PCB", (50.0, 86.0, 1.6), (0, 0, 10.0), mat=mat_pcb_orange, collection=col_components)
esp_lcd = make_box("ESP32_LCD_Glass", (43.2, 57.6, 2.5), (0, 0, 12.0), mat=mat_glass, collection=col_components)

# Two 40mm Speakers (one at top, one at bottom on front face)
spk_top = make_cylinder("Speaker_Top", 20.0, 4.5, (0, 56.0, 10.5), rot=(0, 0, 0), mat=mat_speaker, collection=col_components)
spk_bot = make_cylinder("Speaker_Bottom", 20.0, 4.5, (0, -56.0, 10.5), rot=(0, 0, 0), mat=mat_speaker, collection=col_components)

# Camera Rev 1.3 Mockup (facing negative Z, towards the signer)
cam_pcb = make_box("Camera_PCB", (25.0, 24.0, 1.2), (0, 45.0, -12.5), mat=mat_pcb_green, collection=col_components)
cam_lens = make_cylinder("Camera_Lens", 4.0, 6.0, (0, 45.0, -15.5), rot=(0, 0, 0), mat=mat_glass, collection=col_components)

# Representative External Power Bank (strapped to back)
pb_mockup = make_box("PowerBank_External", (65.0, 95.0, 14.0), (0, -18.0, -26.0), mat=mat_powerbank, collection=col_components)

print("Building 3D Printable Enclosure Shells...")

# ==========================================
# 2. ENCLOSURE SHELLS (PARAMETRIC DESIGN)
# ==========================================
# Overall Enclosure Dimensions
ENCL_W = 80.0     # Width (X)
ENCL_H = 158.0    # Height (Y)
ENCL_D = 30.0     # Depth (Z total: Front + Rear)
WALL_THICK = 2.4  # Wall thickness for rigidity & clean 3D printing
CORNER_R = 7.0    # Rounded phone-style corners

# Function to create rounded box enclosure half
def create_shell_body(name, width, height, depth, wall, is_front=True):
    # Create base cube
    z_center = depth / 2.0 if is_front else -depth / 2.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, z_center))
    shell = bpy.context.active_object
    shell.name = name
    shell.scale = (width, height, depth)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Bevel vertical edges for phone-like rounded feel
    mod_bevel = shell.modifiers.new(name="CornerBevel", type='BEVEL')
    mod_bevel.width = CORNER_R
    mod_bevel.segments = 8
    mod_bevel.limit_method = 'ANGLE'
    mod_bevel.angle_limit = math.radians(60)
    bpy.ops.object.modifier_apply(modifier="CornerBevel")
    
    # Hollow out the inner cavity
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, z_center))
    cutter = bpy.context.active_object
    cutter.name = "InnerCutter"
    cutter.scale = (width - 2 * wall, height - 2 * wall, depth)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    mod_cut_bevel = cutter.modifiers.new(name="InnerBevel", type='BEVEL')
    mod_cut_bevel.width = max(1.0, CORNER_R - wall)
    mod_cut_bevel.segments = 8
    mod_cut_bevel.limit_method = 'ANGLE'
    mod_cut_bevel.angle_limit = math.radians(60)
    bpy.ops.object.modifier_apply(modifier="InnerBevel")
    
    # Boolean difference
    mod_bool = shell.modifiers.new(name="Cavity", type='BOOLEAN')
    mod_bool.operation = 'DIFFERENCE'
    mod_bool.object = cutter
    bpy.ops.object.modifier_apply(modifier="Cavity")
    bpy.data.objects.remove(cutter, do_unlink=True)
    
    return shell

# 2.1 FRONT SHELL (Face towards Listener / Hearing Person)
front_shell = create_shell_body("ASL_Front_Shell", ENCL_W, ENCL_H, 15.0, WALL_THICK, is_front=True)
front_shell.data.materials.append(mat_shell_front)
col_enclosure.objects.link(front_shell)

# Cut screen bezel opening in front shell
screen_cutter = make_box("Cutter_Screen", (45.0, 60.0, 20.0), (0, 0, 15.0))
mod_sc = front_shell.modifiers.new(name="ScreenWindow", type='BOOLEAN')
mod_sc.operation = 'DIFFERENCE'
mod_sc.object = screen_cutter
bpy.ops.object.modifier_apply(modifier="ScreenWindow")
bpy.data.objects.remove(screen_cutter, do_unlink=True)

# Acoustic speaker grilles (Top & Bottom)
def add_speaker_grille(target_obj, center_x, center_y, z_pos):
    # Radial acoustic slot pattern
    for r in [6.0, 11.0, 16.0]:
        count = int(r * 1.3)
        for i in range(count):
            angle = (2 * math.pi / count) * i
            px = center_x + r * math.cos(angle)
            py = center_y + r * math.sin(angle)
            slot = make_cylinder(f"GrilleHole_{r}_{i}", 1.6, 12.0, (px, py, z_pos))
            mod_g = target_obj.modifiers.new(name=f"Grille_{r}_{i}", type='BOOLEAN')
            mod_g.operation = 'DIFFERENCE'
            mod_g.object = slot
            bpy.ops.object.modifier_apply(modifier=f"Grille_{r}_{i}")
            bpy.data.objects.remove(slot, do_unlink=True)

print("Generating acoustic speaker grilles on Front Shell...")
add_speaker_grille(front_shell, 0, 56.0, 15.0)
add_speaker_grille(front_shell, 0, -56.0, 15.0)

# 2.2 REAR SHELL (Face towards Signer / Deaf User)
rear_shell = create_shell_body("ASL_Rear_Shell", ENCL_W, ENCL_H, 15.0, WALL_THICK, is_front=False)
rear_shell.data.materials.append(mat_shell_rear)
col_enclosure.objects.link(rear_shell)

# Camera Aperture Cutout on Rear Shell (facing signer)
cam_aperture = make_cylinder("Cutter_CamLens", 5.0, 20.0, (0, 45.0, -15.0))
mod_cam = rear_shell.modifiers.new(name="CameraWindow", type='BOOLEAN')
mod_cam.operation = 'DIFFERENCE'
mod_cam.object = cam_aperture
bpy.ops.object.modifier_apply(modifier="CameraWindow")
bpy.data.objects.remove(cam_aperture, do_unlink=True)

# Cutouts for Raspberry Pi 2B Ports on Side of Rear Shell:
# USB & Ethernet Cutout (Right edge: X ~ 40mm)
usb_eth_cut = make_box("Cutter_USB_ETH", (15.0, 54.0, 18.0), (40.0, -3.0, -5.0))
mod_ue = rear_shell.modifiers.new(name="Cut_USB_ETH", type='BOOLEAN')
mod_ue.operation = 'DIFFERENCE'
mod_ue.object = usb_eth_cut
bpy.ops.object.modifier_apply(modifier="Cut_USB_ETH")
bpy.data.objects.remove(usb_eth_cut, do_unlink=True)

# HDMI & Micro-USB Power Cutout (Bottom / Left edge)
pwr_hdmi_cut = make_box("Cutter_Pwr_HDMI", (50.0, 15.0, 10.0), (-16.0, -29.0, -4.0))
mod_ph = rear_shell.modifiers.new(name="Cut_PWR_HDMI", type='BOOLEAN')
mod_ph.operation = 'DIFFERENCE'
mod_ph.object = pwr_hdmi_cut
bpy.ops.object.modifier_apply(modifier="Cut_PWR_HDMI")
bpy.data.objects.remove(pwr_hdmi_cut, do_unlink=True)

# MicroSD card access slot (on bottom edge)
sd_cut = make_box("Cutter_MicroSD", (16.0, 12.0, 6.0), (0, -78.0, -5.0))
mod_sd = rear_shell.modifiers.new(name="Cut_MicroSD", type='BOOLEAN')
mod_sd.operation = 'DIFFERENCE'
mod_sd.object = sd_cut
bpy.ops.object.modifier_apply(modifier="Cut_MicroSD")
bpy.data.objects.remove(sd_cut, do_unlink=True)

# Power Bank Strap Loops (Integrated brackets on the rear face)
print("Adding universal power bank strap loops on Rear Shell...")
strap_top = make_box("Strap_Bracket_Top", (32.0, 6.0, 5.0), (0, 0.0, -16.5), mat=mat_shell_rear, collection=col_enclosure)
strap_bot = make_box("Strap_Bracket_Bot", (32.0, 6.0, 5.0), (0, -36.0, -16.5), mat=mat_shell_rear, collection=col_enclosure)

# Cut through-slots for 20mm velcro strap
for b, y_pos in [(strap_top, 0.0), (strap_bot, -36.0)]:
    slot_cut = make_box(f"Cutter_Slot_{y_pos}", (24.0, 3.2, 8.0), (0, y_pos, -16.5))
    m = b.modifiers.new(name="SlotPass", type='BOOLEAN')
    m.operation = 'DIFFERENCE'
    m.object = slot_cut
    bpy.ops.object.modifier_apply(modifier="SlotPass")
    bpy.data.objects.remove(slot_cut, do_unlink=True)

# Corner screw standoffs (M3 screw bosses for assembling the two halves)
boss_locs = [
    (ENCL_W/2 - 6.0, ENCL_H/2 - 6.0),
    (-ENCL_W/2 + 6.0, ENCL_H/2 - 6.0),
    (ENCL_W/2 - 6.0, -ENCL_H/2 + 6.0),
    (-ENCL_W/2 + 6.0, -ENCL_H/2 + 6.0),
]
for idx, (bx, by) in enumerate(boss_locs):
    # Boss in rear shell
    boss = make_cylinder(f"ScrewBoss_Rear_{idx}", 3.5, 12.0, (bx, by, -6.0), mat=mat_shell_rear, collection=col_enclosure)
    # Pilot hole for M3 screw / brass heat-set insert
    hole = make_cylinder(f"Hole_{idx}", 1.6, 14.0, (bx, by, -6.0))
    m = boss.modifiers.new(name="PilotHole", type='BOOLEAN')
    m.operation = 'DIFFERENCE'
    m.object = hole
    bpy.ops.object.modifier_apply(modifier="PilotHole")
    bpy.data.objects.remove(hole, do_unlink=True)

# Set viewport display and camera
bpy.ops.object.select_all(action='DESELECT')

# Add camera for nice presentation render
bpy.ops.object.camera_add(location=(160, -180, 140), rotation=(math.radians(55), 0, math.radians(40)))
cam_view = bpy.context.active_object
scene.camera = cam_view

# Add warm studio 3-point lighting
light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
light_data.energy = 3.5
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (100, -100, 200)
light_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(45))

fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
fill_data.energy = 80.0
fill_data.size = 200.0
fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_data)
scene.collection.objects.link(fill_obj)
fill_obj.location = (-120, 100, 100)

# Save the blend file
cad_dir = r"d:\School\CAPSTONE\G12\ASL\cad"
blend_path = os.path.join(cad_dir, "asl_communicator_model.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Successfully saved Blender project to: {blend_path}")

# Export STLs
front_stl_path = os.path.join(cad_dir, "asl_front_shell.stl")
rear_stl_path = os.path.join(cad_dir, "asl_rear_shell.stl")

# Export front shell STL
bpy.ops.object.select_all(action='DESELECT')
front_shell.select_set(True)
bpy.context.view_layer.objects.active = front_shell
bpy.ops.wm.stl_export(filepath=front_stl_path, export_selected_objects=True)
print(f"Exported Front Shell STL: {front_stl_path}")

# Export rear shell STL
bpy.ops.object.select_all(action='DESELECT')
rear_shell.select_set(True)
bpy.context.view_layer.objects.active = rear_shell
bpy.ops.wm.stl_export(filepath=rear_stl_path, export_selected_objects=True)
print(f"Exported Rear Shell STL: {rear_stl_path}")

print("ALL CAD MODELING COMPLETED SUCCESSFULLY!")
