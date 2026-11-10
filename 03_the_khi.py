import bpy
import bmesh
import math
import numpy as np          # Blender co san numpy

# =====================================================================
#  MO HINH 3 (Trang 7) - THE KHI: 14 hat trong ong bom, bay hon loan va va vao thanh binh.
#  Nut "Nen lai": day pit-tong -> the tich giam, hat sat lai nhau hon, va cham day hon.
#
#  Cach dung: Run Script -> bam N trong khung nhin 3D -> tab "The khi"
#             -> 3 nut: "Nen lai", "Gian ra", "Dat lai"
# =====================================================================

# ---------- Thong so (doi o day) ----------
N        = 14                       # so hat (12-15)
R_P      = 0.12                     # ban kinh hat
COLOR    = (1.00, 0.42, 0.12, 1)    # mau hat (cam do: khi nong, nang luong cao)
V0       = 3.0                      # toc do TB cua hat (don vi/giay)
SEED     = 5                        # doi so nay -> chuyen dong khac
EXPORT_GLB = ""                     # vd "C:/tmp/the_khi.glb" de xuat file cho AR (de trong = khong xuat)

# Ong bom nam ngang theo truc X
RW   = 0.90                         # ban kinh long ong
L    = 4.0                          # chieu dai long ong luc chua nen
P_MIN = 1.3                         # vi tri pit-tong khi nen het co (the tich con ~33%)

FPS, SUB = 24, 4
IDLE_HALF = 60                                   # doan quan sat: 60 frame tien + 60 frame lui = vong lap 120
IDLE_A_END     = 2 * IDLE_HALF                   # 1..120      : quan sat khi chua nen (lap)
COMPRESS_START = IDLE_A_END + 1                  # 121
COMPRESS_END   = COMPRESS_START + 72             # 193         : pit-tong tien vao
IDLE_B_START   = COMPRESS_END                    # 193..312    : quan sat khi da nen (lap)
IDLE_B_END     = IDLE_B_START + 2 * IDLE_HALF - 1
RELEASE_START  = IDLE_B_END + 1                  # 313
RELEASE_END    = RELEASE_START + 72              # 385         : pit-tong lui ra
IDLE_A2_START  = RELEASE_END                     # 385..504    : quan sat sau khi gian ra (lap)
IDLE_A2_END    = IDLE_A2_START + 2 * IDLE_HALF - 1
LAST_FRAME     = IDLE_A2_END

# ====================== PHAN 1: MO PHONG (khong dung bpy) ======================
def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)

def piston_x(f):
    """Vi tri mat pit-tong (truc X) tai frame f."""
    if f <= COMPRESS_START:
        return L
    if f <= COMPRESS_END:
        return L + (P_MIN - L) * _smooth((f - COMPRESS_START) / (COMPRESS_END - COMPRESS_START))
    if f <= RELEASE_START:
        return P_MIN
    if f <= RELEASE_END:
        return P_MIN + (L - P_MIN) * _smooth((f - RELEASE_START) / (RELEASE_END - RELEASE_START))
    return L

def init_state(rng):
    pts = []
    lim = RW - R_P
    while len(pts) < N:
        c = np.array([rng.uniform(R_P, L - R_P), *rng.uniform(-lim, lim, 2)])
        if math.hypot(c[1], c[2]) > lim:
            continue
        if all(np.linalg.norm(c - q) > 3 * R_P for q in pts):
            pts.append(c)
    pos = np.array(pts)
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    vel = d * (V0 * rng.uniform(0.6, 1.4, N))[:, None]
    return dict(pos=pos, vel=vel, hits=np.zeros(1))

def copy_state(S):
    return {k: v.copy() for k, v in S.items()}

def step(S, ft):
    dt = 1.0 / (FPS * SUB)
    pos, vel = S['pos'], S['vel']
    p_old, p_new = piston_x(ft), piston_x(ft + 1.0 / SUB)
    vp = (p_new - p_old) / dt
    pos += vel * dt
    # thanh ong (hinh tru)
    lim = RW - R_P
    rho = np.hypot(pos[:, 1], pos[:, 2]) + 1e-12
    m = rho > lim
    if m.any():
        n = np.zeros_like(pos)
        n[:, 1], n[:, 2] = pos[:, 1] / rho, pos[:, 2] / rho
        pos[m, 1] *= (lim / rho[m])
        pos[m, 2] *= (lim / rho[m])
        vn = (vel * n).sum(1)
        r = m & (vn > 0)
        vel[r] -= 2 * vn[r, None] * n[r]
        S['hits'] += r.sum()
    # day ong (ben trai)
    m = pos[:, 0] < R_P
    pos[m, 0] = R_P
    r = m & (vel[:, 0] < 0)
    vel[r, 0] = -vel[r, 0]
    S['hits'] += r.sum()
    # pit-tong (ben phai, co the dang chuyen dong)
    m = pos[:, 0] > p_new - R_P
    pos[m, 0] = p_new - R_P
    r = m & (vel[:, 0] > vp)
    vel[r, 0] = 2 * vp - vel[r, 0]            # phan xa dan hoi so voi mat dang chuyen dong -> nen thi nong len
    S['hits'] += r.sum()
    # hat va cham hat (dan hoi)
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((d * d).sum(-1)) + np.eye(N) * 9
    ii, jj = np.where(np.triu(dist < 2 * R_P, 1))
    for i, j in zip(ii, jj):
        nrm = (pos[i] - pos[j]) / dist[i, j]
        ov = 2 * R_P - dist[i, j]
        pos[i] += nrm * ov / 2
        pos[j] -= nrm * ov / 2
        vr = np.dot(vel[i] - vel[j], nrm)
        if vr < 0:
            vel[i] -= vr * nrm
            vel[j] += vr * nrm

def run_frames(S, f_start, n):
    """Mo phong n frame tu f_start; tra ve n+1 trang thai (vi tri)."""
    out = [S['pos'].copy()]
    for k in range(n):
        for s in range(SUB):
            step(S, f_start + k + s / SUB)
        out.append(S['pos'].copy())
    return out

def loop_segment(S, f0, frames):
    """Quan sat lap: tien IDLE_HALF frame roi lui lai (chuyen dong dan hoi dao nguoc duoc nen van tu nhien)."""
    st = run_frames(S, f0, IDLE_HALF)
    for i in range(IDLE_HALF + 1):
        frames[f0 + i] = st[i]
    for i in range(1, IDLE_HALF):
        frames[f0 + IDLE_HALF + i] = st[IDLE_HALF - i]

def build_frames(seed=SEED):
    rng = np.random.default_rng(seed)
    S = init_state(rng)
    run_frames(S, -60, 60)                      # khoi dong cho hat phan tan ngau nhien
    S0 = copy_state(S)
    frames = {}
    loop_segment(copy_state(S0), 1, frames)                                  # (1) chua nen
    Sc = copy_state(S0)                                                      # (2) nen lai
    st = run_frames(Sc, COMPRESS_START, COMPRESS_END - COMPRESS_START)
    for i, p in enumerate(st):
        frames[COMPRESS_START + i] = p
    SB = copy_state(Sc)
    loop_segment(copy_state(SB), IDLE_B_START, frames)                       # (3) dang nen
    Sr = copy_state(SB)                                                      # (4) gian ra
    st = run_frames(Sr, RELEASE_START, RELEASE_END - RELEASE_START)
    for i, p in enumerate(st):
        frames[RELEASE_START + i] = p
    loop_segment(copy_state(Sr), IDLE_A2_START, frames)                      # (5) sau khi gian ra
    return frames

# ====================== PHAN 2: DUNG CANH TRONG BLENDER ======================
print("Dang mo phong chuyen dong cua %d hat..." % N)
frames = build_frames(SEED)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
scene = bpy.context.scene
try:
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
except Exception:
    pass

def make_mat(name, color, rough=0.4, alpha=None, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED"),
                          ("show_transparent_back", False)):
            try:
                setattr(m, attr, val)
            except Exception:
                pass
    return m

MAT_GAS    = make_mat("Gas", COLOR, 0.25)
MAT_GLASS  = make_mat("Barrel", (0.85, 0.93, 1.0, 1), 0.05, alpha=0.18)
MAT_RUBBER = make_mat("Rubber", (0.12, 0.12, 0.15, 1), 0.7)
MAT_METAL  = make_mat("Metal", (0.70, 0.72, 0.78, 1), 0.3, metal=0.6)
MAT_TICK   = make_mat("Tick", (0.10, 0.10, 0.12, 1), 0.6)
MAT_GROUND = make_mat("Ground", (0.93, 0.94, 0.96, 1), 0.9)

def parent_to(o, p):
    bpy.context.view_layer.update()
    o.parent = p
    o.matrix_parent_inverse = p.matrix_world.inverted()

root = bpy.data.objects.new("GasModel", None)
bpy.context.collection.objects.link(root)

# ---------- Hat khi ----------
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=R_P, location=(0, 0, 0))
proto = bpy.context.active_object
mesh = proto.data
for poly in mesh.polygons:
    poly.use_smooth = True
mesh.materials.append(MAT_GAS)
atoms = []
for i in range(N):
    o = bpy.data.objects.new(f"Atom_{i:02d}", mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    atoms.append(o)
bpy.data.objects.remove(proto)

# ---------- Ong bom: than ong (kinh), dau ong, mui, vach chia ----------
def add_cyl(name, loc, radius, depth, mat, rot=(0, math.pi / 2, 0), parent=root):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    for poly in o.data.polygons:
        poly.use_smooth = True
    o.data.materials.append(mat)
    if parent is not None:
        parent_to(o, parent)
    return o

barrel = add_cyl("Barrel", (L / 2 + 0.05, 0, 0), RW + 0.02, L + 0.1, MAT_GLASS)
bm = bmesh.new()
bm.from_mesh(barrel.data)
bm.normal_update()
caps = [f for f in bm.faces if abs(f.normal.z) > 0.9]
bmesh.ops.delete(bm, geom=caps, context='FACES')           # bo 2 nap -> ong thong
bm.to_mesh(barrel.data)
bm.free()
sol = barrel.modifiers.new("Solidify", 'SOLIDIFY')
sol.thickness = 0.04
sol.offset = 1.0
add_cyl("EndCap", (-0.05, 0, 0), RW + 0.06, 0.10, MAT_METAL)                 # day ong
add_cyl("Nozzle", (-0.45, 0, 0), 0.16, 0.70, MAT_METAL)                      # dau mui
add_cyl("Flange", (L + 0.12, 0, 0), RW + 0.35, 0.08, MAT_METAL)              # canh cam tay o mieng ong
for k in range(1, 8):                                                        # vach chia the tich
    x = k * 0.5
    bpy.ops.mesh.primitive_torus_add(major_radius=RW + 0.06, minor_radius=0.012,
                                     major_segments=64, minor_segments=8,
                                     location=(x, 0, 0), rotation=(0, math.pi / 2, 0))
    t = bpy.context.active_object
    t.name = f"Tick_{k}"
    t.data.materials.append(MAT_TICK)
    parent_to(t, root)

# ---------- Pit-tong (chuyen dong) ----------
piston = bpy.data.objects.new("Piston", None)
bpy.context.collection.objects.link(piston)
piston.location = (L, 0, 0)
piston.parent = root
parent_to_piston = lambda o: parent_to(o, piston)
head = add_cyl("Piston_Head", (L + 0.10, 0, 0), RW, 0.20, MAT_RUBBER, parent=None)
rod  = add_cyl("Piston_Rod",  (L + 2.45, 0, 0), 0.12, 4.5, MAT_METAL, parent=None)
grip = add_cyl("Piston_Grip", (L + 4.75, 0, 0), 0.65, 0.14, MAT_METAL, parent=None)
for o in (head, rod, grip):
    parent_to_piston(o)

# ---------- San, anh sang, camera ----------
bpy.ops.mesh.primitive_plane_add(size=22, location=(4, 0, -1.25))
bpy.context.active_object.data.materials.append(MAT_GROUND)
bpy.ops.object.light_add(type='SUN', location=(4, -6, 8))
sun = bpy.context.active_object
sun.rotation_euler = (math.radians(55), 0, math.radians(20))
sun.data.energy = 3
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(3.9, 0, 0))
target = bpy.context.active_object
target.name = "CamTarget"
target.empty_display_size = 0.2
bpy.ops.object.camera_add(location=(3.9, -12.5, 0.8))
cam = bpy.context.active_object
cam.data.lens = 35
con = cam.constraints.new('TRACK_TO')
con.target = target
con.track_axis = 'TRACK_NEGATIVE_Z'
con.up_axis = 'UP_Y'
scene.camera = cam
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
bg.inputs["Color"].default_value = (0.93, 0.95, 1.0, 1)

# ---------- Ghi keyframe ----------
scene.render.fps = FPS
prefs = bpy.context.preferences.edit
try:
    old_interp = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = 'LINEAR'
except Exception:
    old_interp = None
for f in range(1, LAST_FRAME + 1):
    P = frames[f]
    for i, o in enumerate(atoms):
        o.location = (float(P[i][0]), float(P[i][1]), float(P[i][2]))
        o.keyframe_insert("location", frame=f)
    piston.location = (float(piston_x(f)), 0.0, 0.0)
    piston.keyframe_insert("location", frame=f)
if old_interp is not None:
    try:
        prefs.keyframe_new_interpolation_type = old_interp
    except Exception:
        pass

# ====================== NUT BAM ======================
STATE = {"mode": "free"}

def _play(context):
    try:
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    except Exception:
        pass

class GAS_OT_compress(bpy.types.Operator):
    bl_idname = "gas.compress"
    bl_label = "Nén lại"
    bl_description = "Đẩy pit-tông vào: thể tích khí giảm, các hạt gần nhau hơn"
    def execute(self, context):
        STATE["mode"] = "compress"
        sc = context.scene
        sc.frame_start, sc.frame_end = COMPRESS_START, IDLE_B_END
        sc.frame_current = COMPRESS_START
        _play(context)
        return {'FINISHED'}

class GAS_OT_release(bpy.types.Operator):
    bl_idname = "gas.release"
    bl_label = "Giãn ra"
    bl_description = "Kéo pit-tông ra: khí giãn ra lấp đầy ống bơm"
    def execute(self, context):
        STATE["mode"] = "release"
        sc = context.scene
        sc.frame_start, sc.frame_end = RELEASE_START, IDLE_A2_END
        sc.frame_current = RELEASE_START
        _play(context)
        return {'FINISHED'}

class GAS_OT_reset(bpy.types.Operator):
    bl_idname = "gas.reset"
    bl_label = "Đặt lại"
    bl_description = "Quay về trạng thái ban đầu"
    def execute(self, context):
        STATE["mode"] = "free"
        sc = context.scene
        sc.frame_start, sc.frame_end = 1, IDLE_A_END
        sc.frame_current = 1
        _play(context)
        return {'FINISHED'}

class GAS_PT_panel(bpy.types.Panel):
    bl_label = "Mô hình thể khí"
    bl_idname = "GAS_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Thể khí"
    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text=f"{N} hạt • bay hỗn loạn, va vào thành bình")
        col.label(text="Thể tích khí: %.0f%%" % context.scene.get("gas_vol_pct", 100.0))
        col.separator()
        col.operator("gas.compress", icon='TRIA_LEFT')
        col.operator("gas.release", icon='TRIA_RIGHT')
        col.operator("gas.reset", icon='LOOP_BACK')

def gas_frame_handler(scene):
    f = scene.frame_current
    scene["gas_vol_pct"] = piston_x(f) / L * 100.0
    if STATE["mode"] == "compress" and f >= IDLE_B_START and scene.frame_start != IDLE_B_START:
        scene.frame_start = IDLE_B_START                      # nen xong: giu nguyen o trang thai nen (lap)
    elif STATE["mode"] == "release" and f >= IDLE_A2_START and scene.frame_start != IDLE_A2_START:
        scene.frame_start = IDLE_A2_START

for nm in ("GAS_OT_compress", "GAS_OT_release", "GAS_OT_reset", "GAS_PT_panel"):
    old = getattr(bpy.types, nm, None)
    if old is not None:
        try:
            bpy.utils.unregister_class(old)
        except Exception:
            pass
for cls in (GAS_OT_compress, GAS_OT_release, GAS_OT_reset, GAS_PT_panel):
    bpy.utils.register_class(cls)
for h in list(bpy.app.handlers.frame_change_post):
    if getattr(h, "__name__", "") == "gas_frame_handler":
        bpy.app.handlers.frame_change_post.remove(h)
bpy.app.handlers.frame_change_post.append(gas_frame_handler)

# ---------- Che do xem co mau + nhin qua camera ----------
try:
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            sp = area.spaces.active
            sp.shading.type = 'MATERIAL'
            sp.region_3d.view_perspective = 'CAMERA'
except Exception:
    pass

scene.frame_start, scene.frame_end = 1, IDLE_A_END
scene.frame_set(1)

if EXPORT_GLB:
    scene.frame_start, scene.frame_end = 1, LAST_FRAME
    bpy.ops.export_scene.gltf(filepath=EXPORT_GLB, export_format='GLB', export_animations=True, export_apply=True)
    scene.frame_start, scene.frame_end = 1, IDLE_A_END

print("Xong! Bam N trong khung nhin 3D -> tab 'Thể khí' -> nut 'Nén lại'.")
print("Cac doan animation (frame): chua nen %d-%d (lap) | nen %d-%d | dang nen %d-%d (lap) | gian ra %d-%d | sau gian %d-%d (lap)"
      % (1, IDLE_A_END, COMPRESS_START, COMPRESS_END, IDLE_B_START, IDLE_B_END,
         RELEASE_START, RELEASE_END, IDLE_A2_START, IDLE_A2_END))
