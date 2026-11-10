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
CHEEK   = make_mat("Cheek",   (1.00, 0.50, 0.45, 1))
METAL   = make_mat("Metal",   (0.70, 0.70, 0.76, 1), 0.25)
GLASS   = make_mat("Glass",   (0.70, 0.90, 1.00, 1), 0.05, alpha=0.4)
GROUND  = make_mat("Ground",  (0.88, 0.92, 0.85, 1), 0.9)

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

# Dau
neck = empty("Neck", (0, 0, 1.18), root)
part('sphere', "Head", HEAD_C, HEAD_R, SKIN, neck)
part('sphere', "Hair", (0, 0.05, 1.52), (0.395, 0.34, 0.36), HAIR, neck)
for tag, x in (("L", -1), ("R", 1)):
    part('sphere', f"Ear_{tag}",   (x * 0.375, 0, 1.50), (0.05, 0.04, 0.07), SKIN, neck)
    part('sphere', f"Cheek_{tag}", (x * 0.22, -0.265, 1.43), (0.06, 0.025, 0.04), CHEEK, neck)
    face_arc(f"Brow_{tag}", x * 0.14, 1.625, 0.10, 0.015, HAIR, 0.007, neck)
part('sphere', "Nose", (0, -0.335, 1.46), (0.035, 0.035, 0.030), SKIN, neck)

# ---------- MAT: mo to tron, nhin LEN TREN ----------
EZ = 1.525                       # do cao tam mat
BLINK = {}                       # truc de chop mat (thu phong theo chieu doc)
for tag, s in (("L", -1), ("R", 1)):
    ex = s * 0.14
    bl = empty(f"Blink_{tag}", (ex, surf_y(ex, EZ), EZ), neck)
    BLINK[tag] = bl
    face_shape(f"EyeWhite_{tag}", ellipse(ex, EZ, 0.058, 0.066), TEETH, 0.004, bl)          # long trang
    face_shape(f"Iris_{tag}",     ellipse(ex + 0.006, EZ + 0.024, 0.036, 0.041), HAIR, 0.008, bl)   # long den, lech len
    face_shape(f"Pupil_{tag}",    ellipse(ex + 0.007, EZ + 0.028, 0.021, 0.025), DARK, 0.012, bl)   # con ngươi
    face_shape(f"Glint_{tag}",    ellipse(ex + 0.017, EZ + 0.040, 0.009, 0.010), TEETH, 0.016, bl)  # diem sang
    # vien mi tren
    lash = [(ex + 0.061 * math.cos(math.radians(a)), EZ + 0.069 * math.sin(math.radians(a)))
            for a in range(8, 173, 12)]
    face_curve(f"Lash_{tag}", lash, DARK, 0.005, bl, off=0.014)

# ---------- MIENG: net cong XUONG (hai dau thap, giua cao), mep PHAI cao hon mep TRAI ----------
# "Phai" = ben phai cua nguoi xem man hinh. Muon doi ben: dat MOUTH_FLIP = True
MOUTH_FLIP = False
mouth_pts = []
for i in range(15):
    t = i / 14
    x = -0.060 + 0.130 * t
    z = 1.375 + 0.020 * t + 0.020 * math.sin(math.pi * t)   # giua cao, dau phai cao hon dau trai
    mouth_pts.append((-x if MOUTH_FLIP else x, z))
face_curve("Mouth", mouth_pts, DARK, 0.0085, neck, off=0.008)

# Toc mai: mu toc phu dinh dau + cac loc toc ru xuong tran (thay phan non che tran)
dome("Hair_Cap", HEAD_C, (0.392, 0.362, 0.372), HAIR, neck, cut=0.5)

def head_point(phi_deg, z, out=0.014):
    """Diem tren be mat dau (phia truoc), nho ra ngoai 'out'."""
    k = math.sqrt(max(1 - ((z - HEAD_C[2]) / HEAD_R[2]) ** 2, 0))
    p = math.radians(phi_deg)
    rx = HEAD_R[0] * k + out
    ry = HEAD_R[1] * k + out
    return (rx * math.sin(p), -ry * math.cos(p), z)

# (goc, do cao tam, nua chieu dai, nua chieu rong) - chieu dai khac nhau cho tu nhien
bangs = [(-80, 1.640, 0.060, 0.060), (-55, 1.665, 0.050, 0.070), (-28, 1.675, 0.045, 0.070),
         (0, 1.655, 0.055, 0.075),
         (28, 1.670, 0.050, 0.070), (55, 1.665, 0.050, 0.070), (80, 1.640, 0.060, 0.060)]
for i, (phi, zc, hl, hw) in enumerate(bangs):
    part('sphere', f"Bang_{i}", head_point(phi, zc), (hw, 0.030, hl), HAIR, neck,
         rot=(0, -R(phi) * 0.25, R(phi)))

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
    part('cylinder', f"Sleeve_{tag}",     (x, 0, SZ - 0.07), (0.088, 0.088, 0.09), SHIRT, sh)
    part('cylinder', f"SleeveTrim_{tag}", (x, 0, SZ - 0.16), (0.091, 0.091, 0.012), TRIM, sh)
    part('cylinder', f"UpperArm_{tag}",   (x, 0, SZ - 0.22), (0.052, 0.052, 0.06), SKIN, sh)
    part('sphere',   f"ElbowBall_{tag}",  (x, 0, SZ - 0.28), (0.056, 0.056, 0.056), SKIN, elb)
    part('cylinder', f"Forearm_{tag}",    (x, 0, SZ - 0.39), (0.050, 0.050, 0.11), SKIN, elb)
    hand = part('sphere', f"Hand_{tag}",  (x, 0, SZ - 0.52), (0.075, 0.075, 0.075), SKIN, elb)
    ARM[tag] = (sh, elb, hand)
elb_L, elb_R = ARM["L"][1], ARM["R"][1]

# ---------- DANG: tay trai chong cam (suy nghi), tay phai cam kinh lup ha thap ----------
# (goc tinh bang do: vai x,y,z va khuyu x - da tinh de ban tay cham cam)
POSE_L = dict(sh=(-55, 20, 25), el=-120)
POSE_R = dict(sh=(0, -20, 10),  el=-65)
def set_pose(sh, el, p):
    sh.rotation_euler = (R(p['sh'][0]), R(p['sh'][1]), R(p['sh'][2]))
    el.rotation_euler = (R(p['el']), 0, 0)
set_pose(sh_L, elb_L, POSE_L)
set_pose(sh_R, elb_R, POSE_R)
bpy.context.view_layer.update()

# Kinh lup: dat theo vi tri ban tay phai sau khi da tao dang, gan vao khuyu phai
hp = ARM["R"][2].matrix_world.translation
mx, my, mz = hp.x, hp.y, hp.z
part('cylinder', "Mag_Handle", (mx, my, mz + 0.10), (0.02, 0.02, 0.11), METAL, elb_R)
part('torus',    "Mag_Ring",   (mx, my, mz + 0.35), (0.15, 0.15, 0.15), METAL, elb_R, rot=(R(90), 0, 0))
part('cylinder', "Mag_Glass",  (mx, my, mz + 0.35), (0.14, 0.14, 0.01), GLASS, elb_R, rot=(R(90), 0, 0))

# Dau CHAM HOI "?" mau do cam tren dau (dung nhu anh 2D)
MARK_MAT = make_mat("Mark", (0.95, 0.25, 0.10, 1), 0.4)
mark = empty("Mark", (0.52, -0.05, 1.98), root)
MX, MY, MCZ, MR = 0.52, -0.05, 2.17, 0.085        # tam + ban kinh phan mong cua dau hoi
hook = []
for a in range(170, -76, -15):                       # tu ben trai, vong len tren, xuong ben phai
    hook.append((MX + MR * math.cos(math.radians(a)), MY, MCZ + MR * math.sin(math.radians(a)), 1))
hook += [(MX + 0.005, MY, MCZ - 0.130, 1), (MX, MY, MCZ - 0.170, 1)]   # than keo xuong giua
cu = bpy.data.curves.new("Mark_Hook", 'CURVE')
cu.dimensions = '3D'
cu.bevel_depth = 0.032
cu.bevel_resolution = 4
cu.use_fill_caps = True
sp = cu.splines.new('POLY')
sp.points.add(len(hook) - 1)
for i, p in enumerate(hook):
    sp.points[i].co = p
hook_obj = bpy.data.objects.new("Mark_Hook", cu)
bpy.context.collection.objects.link(hook_obj)
hook_obj.data.materials.append(MARK_MAT)
parent_to(hook_obj, mark)
part('sphere', "Mark_Dot", (MX, MY, MCZ - 0.250), (0.045, 0.045, 0.045), MARK_MAT, mark)

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

# ---------- Animation: dang suy nghi, tho nhe + nghieng dau, lap 4 giay ----------
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 96

# Moi moc: (frame, root_z, dau[roll,xoay], khuyu_trai, khuyu_phai, vai_phai_y, mark_z, mark_scale)
KEYS = [
    (1,  0.000, (-7, -3), -120, -65, -20, 0.00, 1.00),
    (25, 0.015, (-4,  0), -124, -70, -24, 0.04, 1.12),
    (49, 0.000, (-7,  3), -120, -65, -20, 0.00, 1.00),
    (73, 0.015, (-4,  0), -116, -60, -17, 0.04, 1.12),
    (97, 0.000, (-7, -3), -120, -65, -20, 0.00, 1.00),
]
for f, z, (roll, turn), eL, eR, shRy, mz_, ms in KEYS:
    root.location = (0, 0, z)
    root.keyframe_insert("location", index=2, frame=f)
    neck.rotation_euler = (0, R(roll), R(turn))
    neck.keyframe_insert("rotation_euler", frame=f)
    elb_L.rotation_euler = (R(eL), 0, 0)
    elb_L.keyframe_insert("rotation_euler", index=0, frame=f)
    elb_R.rotation_euler = (R(eR), 0, 0)
    elb_R.keyframe_insert("rotation_euler", index=0, frame=f)
    sh_R.rotation_euler = (R(POSE_R['sh'][0]), R(shRy), R(POSE_R['sh'][2]))
    sh_R.keyframe_insert("rotation_euler", frame=f)
    mark.location = (0.52, -0.05, 1.98 + mz_)
    mark.keyframe_insert("location", frame=f)
    mark.scale = (ms, ms, ms)
    mark.keyframe_insert("scale", frame=f)

# Chop mat nhe (thu phong chieu doc roi mo lai quanh frame 68-76)
for tag in ("L", "R"):
    for f, k in ((1, 1.0), (66, 1.0), (70, 0.08), (74, 1.0), (97, 1.0)):
        BLINK[tag].scale = (1.0, 1.0, k)
        BLINK[tag].keyframe_insert("scale", frame=f)

# Tay trai (chong cam) giu nguyen goc vai
sh_L.rotation_euler = (R(POSE_L['sh'][0]), R(POSE_L['sh'][1]), R(POSE_L['sh'][2]))
sh_L.keyframe_insert("rotation_euler", frame=1)
sh_L.keyframe_insert("rotation_euler", frame=97)

scene.frame_set(1)
print("Xong! Bam phim cach (Space) de xem animation.")
