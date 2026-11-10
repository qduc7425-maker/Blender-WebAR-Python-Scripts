import bpy
import bmesh
import math

# ---------- Don dep canh ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ---------- Mau sac dung ----------
try:
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
except Exception:
    pass

# ---------- Vat lieu (doi mau o day) ----------
def make_mat(name, color, rough=0.55, alpha=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
        try:
            m.blend_method = 'BLEND'
        except Exception:
            pass
    return m

SKIN    = make_mat("Skin",    (0.96, 0.72, 0.50, 1))
HAIR    = make_mat("Hair",    (0.16, 0.08, 0.05, 1))
HAT     = make_mat("Hat",     (0.98, 0.58, 0.08, 1))
HATBAND = make_mat("HatBand", (1.00, 0.80, 0.20, 1))
SHIRT   = make_mat("Shirt",   (0.85, 0.90, 0.25, 1))
TRIM    = make_mat("Trim",    (0.25, 0.60, 0.28, 1))
PANTS   = make_mat("Pants",   (0.55, 0.22, 0.10, 1))
SHOE    = make_mat("Shoe",    (0.85, 0.55, 0.20, 1))
DARK    = make_mat("Dark",    (0.03, 0.02, 0.02, 1), 0.3)
MOUTH   = make_mat("Mouth",   (0.50, 0.05, 0.08, 1), 0.4)
TONGUE  = make_mat("Tongue",  (0.95, 0.42, 0.48, 1), 0.4)
TEETH   = make_mat("Teeth",   (1.00, 1.00, 1.00, 1), 0.3)
CHEEK   = make_mat("Cheek",   (0.78, 0.58, 0.80, 1))   # ma tim hong nhat (lanh)
METAL   = make_mat("Metal",   (0.70, 0.70, 0.76, 1), 0.25)
GLASS   = make_mat("Glass",   (0.70, 0.90, 1.00, 1), 0.05, alpha=0.4)
GROUND  = make_mat("Ground",  (0.88, 0.92, 0.85, 1), 0.9)
SWEAT   = make_mat("Sweat",   (0.70, 0.90, 1.00, 1), 0.1)      # giot mo hoi
CUFF    = make_mat("Cuff",    (0.62, 0.34, 0.28, 1))           # gau quan xan
FANBODY = make_mat("FanBody", (0.86, 0.68, 0.45, 1), 0.7)      # mat quat
FANRIB  = make_mat("FanRib",  (0.45, 0.27, 0.15, 1), 0.7)      # nan quat + vien
MOTION  = make_mat("Motion",  (0.30, 0.25, 0.22, 1), 0.6)      # net chuyen dong

# ---------- DA MAT: xanh tai nhat dan TU TRAN XUONG (nhu dang nong nhung mau xanh) ----------
# Chinh cac so nay neu muon xanh dam/nhat hon hoac lan xuong thap hon:
COLD_COLOR    = (0.64, 0.75, 0.93, 1)    # mau xanh
COLD_STRENGTH = 0.90                     # 0 = khong xanh, 1 = xanh het co (o tran)
COLD_TOP, COLD_BOTTOM = 0.70, 0.15       # do cao tren dau (0 = cam, 1 = dinh dau): xanh dam tu TOP, nhat dan toi BOTTOM
SKIN_BASE     = (0.96, 0.72, 0.50, 1)    # mau da goc

def make_gradient_skin(name, base, tint, strength, top, bottom):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes, links = m.node_tree.nodes, m.node_tree.links
    bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs["Roughness"].default_value = 0.55
    tc  = nodes.new('ShaderNodeTexCoord')        # toa do "Generated": z chay 0 (cam) -> 1 (dinh dau)
    sep = nodes.new('ShaderNodeSeparateXYZ')
    rng = nodes.new('ShaderNodeMapRange')
    try:
        mix = nodes.new('ShaderNodeMix')         # Blender 3.4+
        mix.data_type = 'RGBA'
    except Exception:
        mix = nodes.new('ShaderNodeMixRGB')      # Blender cu hon
    cols = [i for i in mix.inputs if i.type == 'RGBA']
    out  = [o for o in mix.outputs if o.type == 'RGBA'][0]
    links.new(sep.inputs[0], tc.outputs[0])
    links.new(rng.inputs[0], sep.outputs[2])
    rng.inputs[1].default_value = bottom
    rng.inputs[2].default_value = top
    rng.inputs[3].default_value = 0.0
    rng.inputs[4].default_value = strength
    links.new(mix.inputs[0], rng.outputs[0])
    cols[0].default_value = base
    cols[1].default_value = tint
    links.new(bsdf.inputs["Base Color"], out)
    return m

def lerp_col(a, b, f):
    return tuple(a[i] * (1 - f) + b[i] * f for i in range(4))

FACESKIN = make_gradient_skin("FaceSkin", SKIN_BASE, COLD_COLOR, COLD_STRENGTH, COLD_TOP, COLD_BOTTOM)
EARSKIN  = make_mat("EarSkin",  lerp_col(SKIN_BASE, COLD_COLOR, 0.50), 0.55)   # tai: xanh vua
NOSESKIN = make_mat("NoseSkin", lerp_col(SKIN_BASE, COLD_COLOR, 0.30), 0.55)   # mui: xanh nhe
PALE     = make_mat("Pale", (0.88, 0.86, 0.95, 1), 0.55)                       # ban tay tai nhot
SCARF    = make_mat("Scarf", (0.95, 0.45, 0.15, 1), 0.8)                       # khan quang cam

# ---------- Ham tien ich ----------
R = math.radians

def parent_to(o, p):
    bpy.context.view_layer.update()
    o.parent = p
    o.matrix_parent_inverse = p.matrix_world.inverted()

def empty(name, loc, parent=None):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=loc)
    e = bpy.context.active_object
    e.name = name
    e.empty_display_size = 0.05
    if parent:
        parent_to(e, parent)
    return e

def part(kind, name, loc, scale, mat, parent=None, rot=(0, 0, 0)):
    if kind == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, location=loc)
    elif kind == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=2, location=loc)
    elif kind == 'torus':
        bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.15,
                                         major_segments=48, minor_segments=12, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    o.rotation_euler = rot
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    if parent:
        parent_to(o, parent)
    return o

def dome(name, loc, scale, mat, parent=None, cut=0.0):
    """Nua tren cua hinh cau: bo phan co z < cut (tinh theo cau don vi)."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, location=loc)
    o = bpy.context.active_object
    bm = bmesh.new()
    bm.from_mesh(o.data)
    low = [v for v in bm.verts if v.co.z < cut - 1e-6]
    bmesh.ops.delete(bm, geom=low, context='VERTS')
    bm.to_mesh(o.data)
    bm.free()
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    if parent:
        parent_to(o, parent)
    return o

# --- Ve tren mat dau: bam theo be mat ellipsoid cua dau ---
HEAD_C = (0.0, 0.0, 1.50)
HEAD_R = (0.38, 0.35, 0.36)

def surf_y(x, z):
    v = 1 - (x / HEAD_R[0]) ** 2 - ((z - HEAD_C[2]) / HEAD_R[2]) ** 2
    return -HEAD_R[1] * math.sqrt(max(v, 0.0))

def face_arc(name, cx, cz, w, h, mat, thick, parent, n=13):
    """Net cong (dung cho mat cuoi nham, long may)."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = thick
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    sp = cu.splines.new('POLY')
    sp.points.add(n - 1)
    for i in range(n):
        t = i / (n - 1)
        x = cx + w * (t - 0.5)
        z = cz + h * math.sin(math.pi * t)
        sp.points[i].co = (x, surf_y(x, z) - 0.004, z, 1)
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    parent_to(o, parent)
    return o

def face_curve(name, pts, mat, thick, parent, off=0.006):
    """Duong cong bat ky (danh sach (x, z)) bam theo mat dau."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = thick
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for i, (x, z) in enumerate(pts):
        sp.points[i].co = (x, surf_y(x, z) - off, z, 1)
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    parent_to(o, parent)
    return o

def ellipse(cx, cz, rx, rz, n=28):
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cz + rz * math.sin(2 * math.pi * i / n)) for i in range(n)]

def face_shape(name, outline, mat, off, parent):
    """To phang bam theo mat dau. outline: danh sach (x, z) nguoc chieu kim dong ho."""
    n = len(outline)
    cx = sum(p[0] for p in outline) / n
    cz = sum(p[1] for p in outline) / n
    pts = [(cx, cz)]
    pts += [(cx + (x - cx) * 0.5, cz + (z - cz) * 0.5) for x, z in outline]
    pts += list(outline)
    verts = [(x, surf_y(x, z) - off, z) for x, z in pts]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((0, 1 + i, 1 + j))
        faces.append((1 + i, 1 + n + i, 1 + n + j, 1 + j))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    parent_to(o, parent)
    return o

# ---------- Dung nhan vat (nhin ve phia -Y), cao khoang 2.0 ----------
root = empty("Character", (0, 0, 0))

# Than + ham
part('sphere',   "Body",   (0, 0, 0.90), (0.25, 0.18, 0.30), SHIRT, root)
part('cylinder', "Collar", (0, 0, 1.16), (0.15, 0.11, 0.03), TRIM, root)
part('cylinder', "Hip",    (0, 0, 0.64), (0.22, 0.16, 0.08), PANTS, root)

# Chan
hip_L = empty("Hip_L", (-0.10, 0, 0.62), root)
hip_R = empty("Hip_R", (0.10, 0, 0.62), root)
for hp, tag, x in ((hip_L, "L", -0.10), (hip_R, "R", 0.10)):
    part('cylinder', f"Leg_{tag}",  (x, 0, 0.37), (0.09, 0.09, 0.26), PANTS, hp)
    part('sphere',   f"Shoe_{tag}", (x, -0.05, 0.08), (0.12, 0.20, 0.08), SHOE, hp)

# Gau quan xan len (nhu anh)
for hp, x in ((hip_L, -0.10), (hip_R, 0.10)):
    part('cylinder', f"Cuff_{hp.name}", (x, 0, 0.17), (0.103, 0.103, 0.032), CUFF, hp)

# Dau
neck = empty("Neck", (0, 0, 1.18), root)
part('sphere', "Head", HEAD_C, HEAD_R, FACESKIN, neck)
part('sphere', "Hair", (0, 0.05, 1.52), (0.395, 0.34, 0.36), HAIR, neck)
for tag, x in (("L", -1), ("R", 1)):
    part('sphere', f"Ear_{tag}",   (x * 0.375, 0, 1.50), (0.05, 0.04, 0.07), EARSKIN, neck)
    part('sphere', f"Cheek_{tag}", (x * 0.22, -0.262, 1.43), (0.075, 0.030, 0.050), CHEEK, neck)
    # Long may lo lang: dau trong (gan mui) cao, dau ngoai thap
    b_in, b_out = x * 0.14 - x * 0.05, x * 0.14 + x * 0.05
    brow = []
    for k in range(9):
        t = k / 8
        brow.append((b_in + (b_out - b_in) * t, 1.640 - 0.028 * t + 0.007 * math.sin(math.pi * t)))
    face_curve(f"Brow_{tag}", brow, HAIR, 0.0075, neck, off=0.018)
part('sphere', "Nose", (0, -0.335, 1.46), (0.035, 0.035, 0.030), NOSESKIN, neck)

# ---------- MAT: tron to, nhin xuong ben trai, co vien den ----------
EZ = 1.52                        # do cao tam mat
BLINK = {}                       # truc de chop mat (thu phong theo chieu doc)
for tag, s in (("L", -1), ("R", 1)):
    ex = s * 0.14
    bl = empty(f"Blink_{tag}", (ex, surf_y(ex, EZ), EZ), neck)
    BLINK[tag] = bl
    face_shape(f"EyeWhite_{tag}", ellipse(ex, EZ, 0.062, 0.062), TEETH, 0.004, bl)                   # long trang
    face_shape(f"Iris_{tag}",     ellipse(ex - 0.013, EZ - 0.013, 0.043, 0.047), HAIR, 0.008, bl)    # long den, lech xuong-trai
    face_shape(f"Pupil_{tag}",    ellipse(ex - 0.014, EZ - 0.014, 0.025, 0.028), DARK, 0.010, bl)
    face_shape(f"Glint_{tag}",    ellipse(ex + 0.002, EZ - 0.006, 0.010, 0.011), TEETH, 0.012, bl)   # diem sang
    ring = [(ex + 0.064 * math.cos(2 * math.pi * k / 28), EZ + 0.064 * math.sin(2 * math.pi * k / 28))
            for k in range(29)]
    face_curve(f"EyeRim_{tag}", ring, DARK, 0.0045, bl, off=0.016)                                   # vien mat

# ---------- MIENG: net luon song nho (run ray vi lanh) ----------
mouth_pts = [(-0.048 + 0.096 * k / 24, 1.385 + 0.0075 * math.sin(2 * math.pi * 2.5 * k / 24))
             for k in range(25)]
face_curve("Mouth", mouth_pts, DARK, 0.0045, neck, off=0.008)

# Toc mai: mu toc phu dinh dau + cac loc toc ru xuong tran (thay phan non che tran)
dome("Hair_Cap", HEAD_C, (0.392, 0.362, 0.372), HAIR, neck, cut=0.5)

def head_point(phi_deg, z, out=0.014):
    """Diem tren be mat dau (phia truoc), nho ra ngoai 'out'."""
    k = math.sqrt(max(1 - ((z - HEAD_C[2]) / HEAD_R[2]) ** 2, 0))
    p = math.radians(phi_deg)
    rx = HEAD_R[0] * k + out
    ry = HEAD_R[1] * k + out
    return (rx * math.sin(p), -ry * math.cos(p), z)

# (Da BO toc mai theo yeu cau. Con lai toc o hai ben tai va toc phia sau.)

# Toc mai hai ben canh tai
for tag, s in (("L", -1), ("R", 1)):
    part('sphere', f"Sideburn_{tag}", (s * 0.358, -0.11, 1.56), (0.030, 0.050, 0.090), HAIR, neck)

# Non: chi con nua tren, dat cao hon de khong che tran
part('sphere',   "Hat_Top",  (0, 0, 2.015), (0.04, 0.04, 0.035), HATBAND, neck)
dome("Hat_Dome", (0, 0, 1.765), (0.405, 0.375, 0.25), HAT, neck, cut=0.0)
part('cylinder', "Hat_Band", (0, 0, 1.775), (0.41, 0.38, 0.04), HATBAND, neck)
part('cylinder', "Hat_Brim", (0, -0.04, 1.76), (0.50, 0.50, 0.025), HAT, neck)

# Tay co KHUYU: vai (Shoulder) -> khuyu (Elbow) -> cang tay + ban tay
SX, SZ = 0.18, 1.08
sh_L = empty("Shoulder_L", (-SX, 0, SZ), root)
sh_R = empty("Shoulder_R", (SX, 0, SZ), root)
ARM = {}
for sh, tag, x in ((sh_L, "L", -SX), (sh_R, "R", SX)):
    elb = empty(f"Elbow_{tag}", (x, 0, SZ - 0.28), sh)
    part('sphere',   f"Joint_{tag}",      (x, 0, SZ), (0.10, 0.10, 0.10), SHIRT, sh)
    part('cylinder', f"UpperArm_{tag}",   (x, 0, SZ - 0.14), (0.085, 0.085, 0.14), SHIRT, sh)
    part('sphere',   f"ElbowBall_{tag}",  (x, 0, SZ - 0.28), (0.078, 0.078, 0.078), SHIRT, elb)
    part('cylinder', f"Forearm_{tag}",    (x, 0, SZ - 0.39), (0.070, 0.070, 0.11), SHIRT, elb)
    hand = part('sphere', f"Hand_{tag}",  (x, 0, SZ - 0.52), (0.075, 0.075, 0.075), PALE, elb)
    ARM[tag] = (sh, elb, hand)
elb_L, elb_R = ARM["L"][1], ARM["R"][1]

# ---------- DANG LANH: hai tay bat cheo om nguoi, khan quang cam, run ray ----------
# (goc tinh bang do: vai x,y,z va khuyu x - da tinh de ban tay dat len canh tay ben doi dien)
# Tay PHAI man hinh vat sang trai, o phia TRUOC; tay TRAI man hinh vat sang phai, o phia SAU.
POSE_A = dict(sh=(-30, -70, -72), el=-76)
POSE_B = dict(sh=(-40,  54,  62), el=-88)
def set_pose(sh, el, p):
    sh.rotation_euler = (R(p['sh'][0]), R(p['sh'][1]), R(p['sh'][2]))
    el.rotation_euler = (R(p['el']), 0, 0)
set_pose(sh_R, elb_R, POSE_A)
set_pose(sh_L, elb_L, POSE_B)
bpy.context.view_layer.update()

def line3d(name, pts, mat, thick, parent):
    """Duong cong 3D co do day (pts: danh sach (x, y, z) toa do the gioi)."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = thick
    cu.bevel_resolution = 3
    cu.use_fill_caps = True
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for i, (x, y, z) in enumerate(pts):
        sp.points[i].co = (x, y, z, 1)
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    parent_to(o, parent)
    return o

# Khan quang: hai vong day quanh co + duoi khan ru xuong nguc co tua
part('sphere', "Scarf_Top", (0, 0, 1.215), (0.255, 0.190, 0.085), SCARF, root)
part('sphere', "Scarf_Low", (0, 0, 1.125), (0.275, 0.205, 0.075), SCARF, root)
TILT = 14
part('sphere', "Scarf_Tail", (0.09, -0.205, 0.90), (0.075, 0.026, 0.27), SCARF, root, rot=(0, R(TILT), 0))
bx = 0.09 - 0.27 * math.sin(R(TILT))
bz = 0.90 - 0.27 * math.cos(R(TILT))
for k in range(-2, 3):                                    # tua khan o mep duoi duoi khan
    part('cylinder', f"Scarf_Fringe_{k + 2}",
         (bx + k * 0.026 * math.cos(R(TILT)), -0.205, bz - 0.012 - k * 0.026 * math.sin(R(TILT))),
         (0.009, 0.009, 0.028), SCARF, root, rot=(0, R(TILT), 0))

# Net run ray (luon song) hai ben nguoi va hai chan
def wavy(name, x0, z0, h, amp, cycles, parent):
    pts = [(x0 + amp * math.sin(2 * math.pi * cycles * k / 24), -0.12, z0 + h * (k / 24 - 0.5))
           for k in range(25)]
    return line3d(name, pts, MOTION, 0.0045, parent)
SHV_L = (-0.40, -0.12, 0.80)
SHV_R = (0.40, -0.12, 0.80)
shiver_L = empty("Shiver_L", SHV_L, root)
shiver_R = empty("Shiver_R", SHV_R, root)
wavy("Shiver_L1", -0.42, 1.02, 0.34, 0.018, 2.0, shiver_L)
wavy("Shiver_L2", -0.48, 0.97, 0.14, 0.014, 1.5, shiver_L)
wavy("Shiver_L3", -0.30, 0.42, 0.30, 0.015, 2.0, shiver_L)
wavy("Shiver_R1",  0.42, 1.04, 0.36, 0.018, 2.0, shiver_R)
wavy("Shiver_R2",  0.47, 1.07, 0.14, 0.014, 1.5, shiver_R)
wavy("Shiver_R3",  0.30, 0.42, 0.30, 0.015, 2.0, shiver_R)
wavy("Shiver_R4",  0.36, 0.40, 0.12, 0.012, 1.5, shiver_R)

# ---------- San, den, camera ----------
bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, 0))
bpy.context.active_object.data.materials.append(GROUND)

bpy.ops.object.light_add(type='SUN', location=(4, -4, 6))
sun = bpy.context.active_object
sun.rotation_euler = (R(50), 0, R(30))
sun.data.energy = 3

bpy.ops.object.camera_add(location=(0, -6.0, 1.1))
cam = bpy.context.active_object
cam.rotation_euler = (R(90), 0, 0)
bpy.context.scene.camera = cam

bpy.context.scene.world.use_nodes = True
bg = next(n for n in bpy.context.scene.world.node_tree.nodes if n.type == 'BACKGROUND')
bg.inputs["Color"].default_value = (0.78, 0.90, 1.0, 1)

# ---------- Animation: run NHE, lap 4 giay ----------
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 96

# Run nhe: dao dong rat nho moi 4 frame (khoang 6 lan/giay), kem cui dau nhe
# Muon run manh hon: tang SHAKE (vd 2.0); nhe hon nua: giam (vd 0.5)
SHAKE = 1.0
for i, f in enumerate(range(1, 98, 4)):
    sg = 1 if i % 2 == 0 else -1
    root.location = (0.002 * SHAKE * sg, 0, -0.0015 * SHAKE * sg)
    root.keyframe_insert("location", frame=f)
    neck.rotation_euler = (R(4), 0, R(0.6 * SHAKE * sg))
    neck.keyframe_insert("rotation_euler", frame=f)
    for sv, base, ph in ((shiver_L, SHV_L, 1), (shiver_R, SHV_R, -1)):
        sv.location = (base[0] + 0.005 * SHAKE * sg * ph, base[1], base[2])
        sv.keyframe_insert("location", frame=f)
        sv.scale = (1.0, 1.0, 1.0 + 0.03 * SHAKE * sg * ph)
        sv.keyframe_insert("scale", frame=f)

# Chop mat nhanh (so)
for tag in ("L", "R"):
    for f, k in ((1, 1.0), (60, 1.0), (63, 0.08), (66, 1.0), (97, 1.0)):
        BLINK[tag].scale = (1.0, 1.0, k)
        BLINK[tag].keyframe_insert("scale", frame=f)

scene.frame_set(1)
print("Xong! Bam phim cach (Space) de xem animation.")
