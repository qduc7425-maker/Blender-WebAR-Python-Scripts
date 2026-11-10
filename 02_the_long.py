import bpy
import bmesh
import math
import numpy as np          # Blender co san numpy

# =====================================================================
#  MO HINH THE LONG (Trang 6): 70 hat trong coc (nam im truoc khi do);
#  nut "Do sang hop khac" -> chat long doi hinh theo vat chua, THE TICH GIU NGUYEN
#
#  Cach dung: Run Script -> bam N trong khung nhin 3D -> tab "The long"
#             -> 2 nut: "Do sang hop khac" va "Dat lai (quan sat)"
# =====================================================================

# ---------- Thong so (doi o day) ----------
N        = 70                     # so hat (khong doi trong luc do)
R_P      = 0.17                   # ban kinh hat dung de mo phong
R_DRAW   = 0.165                  # ban kinh hat ve ra (hoi nho hon de khong cham nhau)
COLOR    = (0.15, 0.55, 0.95, 1)  # mau hat (xanh nuoc)
SEED     = 3                      # doi so nay -> kieu chuyen dong khac
EXPORT_GLB = ""                   # vd "C:/tmp/the_long.glb" de xuat file cho AR (de trong = khong xuat)

# Coc (hinh non cut, mieng rong o tren)
CUP_R1, CUP_R2, CUP_H = 0.85, 1.05, 1.9
# Hop (hinh chu nhat, ben phai coc): [x0,x1] theo X, +-BOX_Y theo Y, cao BOX_H
BOX_X0, BOX_X1, BOX_Y, BOX_H = 1.6, 3.6, 1.1, 0.9
BOX_XC = (BOX_X0 + BOX_X1) / 2

# Vat ly (khong can doi)
FPS, SUB, ITERS = 24, 6, 5        # khung/giay, buoc con moi khung, so lan giai va cham
G        = 6.0                    # trong luc (nho hon that -> cham, uyen chuyen)
DAMP     = 3.0                    # nhot (cang lon cang "sanh")
AGIT     = 6.0                    # rung nhiet ngau nhien
SW_W, SW_T, SW_K = 3.0, 3.0, 4.0  # dong chay dao chieu theo do cao: lam hat truot qua nhau

# Dong thoi gian (frame)
IDLE_HALF  = 60                                  # 60 frame tien + 60 frame lui = vong lap 120 (dung cho doan trong hop)
IDLE1_END  = 2 * IDLE_HALF                       # 1..120: quan sat trong coc (hat nam im)
POUR_START = IDLE1_END + 1                       # 121: bat dau do
POUR_END   = POUR_START + 340                    # 385: do xong
BOX_IDLE_START = POUR_END                        # 385..504: quan sat trong hop
BOX_IDLE_END   = BOX_IDLE_START + 2 * IDLE_HALF - 1
WARM = 120                                       # so frame "khoi dong" de hat mat trat tu
SETTLE = 90                                      # so frame de hat LANG XUONG va nam im trong coc

# ====================== PHAN 1: MO PHONG (khong dung bpy) ======================
PRIM = np.array([CUP_R2, 0.0, CUP_H])            # mep phai cua coc (truc xoay khi do)
P0 = np.array([CUP_R2, 0.0, CUP_H])              # vi tri nghi
P1 = P0 + np.array([0.0, 0.0, 1.1])              # nhac coc len
P2 = np.array([BOX_XC - 0.3, 0.0, CUP_H + 1.1])        # dua coc ra phia tren hop
CUP_KEYS = [                                     # (frame, truc xoay, goc nghieng do)
    (POUR_START,       P0, 0),
    (POUR_START + 28,  P1, 0),
    (POUR_START + 64,  P2, 0),
    (POUR_START + 128, P2, 100),
    (POUR_START + 248, P2, 120),
    (POUR_START + 284, P2, 0),
    (POUR_START + 316, P1, 0),
    (POUR_START + 340, P0, 0),
]

def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)

def cup_pose(f):
    """Tra ve (truc xoay P, goc theta rad) cua coc tai frame f (co the le)."""
    k = CUP_KEYS
    if f <= k[0][0]:
        return k[0][1], math.radians(k[0][2])
    for (fa, pa, ta), (fb, pb, tb) in zip(k[:-1], k[1:]):
        if f <= fb:
            u = _smooth((f - fa) / (fb - fa))
            return pa + (pb - pa) * u, math.radians(ta + (tb - ta) * u)
    return k[-1][1], math.radians(k[-1][2])

def rot_y(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def cup_loc_rot(f):
    """Vi tri goc coc (loc) va ma tran quay Rm: world = loc + Rm @ local."""
    P, th = cup_pose(f)
    Rm = rot_y(th)
    return P - Rm @ PRIM, Rm, th

K_WALL = (CUP_R2 - CUP_R1) / CUP_H
WALL_N = math.sqrt(1 + K_WALL ** 2)

def init_state(rng):
    """Xep hat thanh luoi long trong coc (se bi lam mat trat tu o buoc khoi dong)."""
    s = 2 * R_P
    pts = []
    iz = 0
    while len(pts) < N and iz < 60:
        z = R_P + iz * s
        rmax = CUP_R1 + K_WALL * z - R_P * WALL_N - 0.02
        layer = []
        m = int(rmax / s) + 1
        for ix in range(-m, m + 1):
            for iy in range(-m, m + 1):
                x, y = ix * s, iy * s
                if math.hypot(x, y) <= rmax:
                    layer.append((x, y, z))
        rng.shuffle(layer)
        pts += layer
        iz += 1
    pos = np.array(pts[:N], dtype=float)
    pos += rng.uniform(-0.03, 0.03, pos.shape)
    return dict(pos=pos, vel=np.zeros_like(pos),
                fcup=np.ones(N, bool), fbox=np.zeros(N, bool))

def copy_state(S):
    return {k: v.copy() for k, v in S.items()}

def solve_pairs(pos):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((d * d).sum(-1)) + np.eye(len(pos)) * 1e3
    over = np.clip(2 * R_P - dist, 0, None)
    if not over.any():
        return False
    corr = ((over / dist)[..., None] * d).sum(1)
    cnt = (over > 0).sum(1)
    pos += 0.5 * corr * (1.4 / np.maximum(cnt, 1))[:, None]
    return True

def cup_constraint(S, loc, Rm, th):
    idx = np.where(S['fcup'])[0]
    if idx.size == 0:
        return
    pl = (S['pos'][idx] - loc) @ Rm                      # sang toa do coc
    x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
    rho = np.hypot(x, y) + 1e-9
    zc = np.clip(z, 0, CUP_H)
    d = (CUP_R1 + K_WALL * zc - rho) / WALL_N            # khoang cach toi thanh coc
    push = (z <= CUP_H) & (d < R_P)
    amt = np.where(push, R_P - d, 0.0)
    pl[:, 0] += amt * (-x / rho) / WALL_N
    pl[:, 1] += amt * (-y / rho) / WALL_N
    pl[:, 2] += amt * K_WALL / WALL_N
    if th < 0.5:                                         # coc con dung: them thanh chan phia tren mieng
        hi = z > CUP_H
        lim = CUP_R2 - R_P
        sc = np.where(hi & (rho > lim), lim / rho, 1.0)
        pl[:, 0] *= sc
        pl[:, 1] *= sc
    pl[:, 2] = np.maximum(pl[:, 2], R_P)                 # day coc
    S['pos'][idx] = loc + pl @ Rm.T

def box_constraint(S):
    idx = np.where(S['fbox'])[0]
    if idx.size == 0:
        return
    p = S['pos'][idx]
    p[:, 0] = np.clip(p[:, 0], BOX_X0 + R_P, BOX_X1 - R_P)
    p[:, 1] = np.clip(p[:, 1], -BOX_Y + R_P, BOX_Y - R_P)
    S['pos'][idx] = p

def step(S, ft, rng, env_cup, env_box, agit=1.0):
    DT = 1.0 / (FPS * SUB)
    pos, vel = S['pos'], S['vel']
    vel[:, 2] -= G * DT
    if agit > 1e-3:
        vel += rng.normal(0, AGIT * agit * DT, vel.shape) * np.array([1, 1, 0.5])
    t = ft / FPS
    for mask, cx, env in ((S['fcup'], 0.0, env_cup), (S['fbox'], BOX_XC, env_box)):
        if env > 1e-3 and mask.any():
            rel = pos[mask][:, :2] - np.array([cx, 0.0])
            rho = np.hypot(rel[:, 0], rel[:, 1]) + 1e-9
            om = SW_W * env * np.sin(2 * math.pi * t / SW_T + 2.5 * pos[mask][:, 2])
            mag = om * np.minimum(rho, 0.8)
            target = np.stack([-rel[:, 1] / rho * mag, rel[:, 0] / rho * mag], 1)
            vel[np.where(mask)[0], :2] += SW_K * (target - vel[mask][:, :2]) * DT
    vel *= math.exp(-DAMP * DT)
    sp = np.linalg.norm(vel, axis=1)
    big = sp > 8.0
    vel[big] *= (8.0 / sp[big])[:, None]
    old = pos.copy()
    pos += vel * DT
    loc, Rm, th = cup_loc_rot(ft + 1.0 / SUB)
    # hat roi vao hop: tu luc do ghi nho la "trong hop"
    inbox = (~S['fcup']) & (~S['fbox']) & (pos[:, 2] < BOX_H) & \
            (pos[:, 0] > BOX_X0) & (pos[:, 0] < BOX_X1) & (np.abs(pos[:, 1]) < BOX_Y)
    S['fbox'] |= inbox
    for _ in range(ITERS):
        solve_pairs(pos)
        cup_constraint(S, loc, Rm, th)
        box_constraint(S)
        pos[:, 2] = np.maximum(pos[:, 2], R_P)           # san
    # hat ra khoi mieng coc -> khong con bi coc rang buoc
    zl = ((pos - loc) @ Rm)[:, 2]
    S['fcup'] &= ~((zl > CUP_H + 0.02) & (th > 0.5))   # chi tinh la da do ra khi coc da nghieng
    vel[:] = (pos - old) / DT

def run_frames(S, f_start, n, env_cup, env_box, rng, agit=lambda f: 1.0):
    """Mo phong n frame tu frame f_start; tra ve danh sach n+1 trang thai (vi tri)."""
    out = [S['pos'].copy()]
    for k in range(n):
        f = f_start + k
        for s in range(SUB):
            ft = f + s / SUB
            step(S, ft, rng, env_cup(ft), env_box(ft), agit(ft))
        out.append(S['pos'].copy())
    return out

def ramp(a, b):
    """0 truoc a, len 1 tai b."""
    return lambda f: _smooth((f - a) / (b - a))

def build_frames(seed=SEED):
    """Tra ve dict {frame: mang vi tri (N,3)} cho toan bo dong thoi gian 1..BOX_IDLE_END."""
    rng = np.random.default_rng(seed)
    S = init_state(rng)
    one, zero = (lambda f: 1.0), (lambda f: 0.0)
    run_frames(S, -WARM - SETTLE, WARM, one, zero, rng)                # khoi dong: hat mat trat tu
    run_frames(S, -SETTLE, SETTLE, zero, zero, rng, agit=zero)         # het khuay, het rung: hat lang xuong, nam im
    S['vel'][:] = 0.0
    S0 = copy_state(S)
    frames = {}

    # (1) quan sat trong coc: cac hat NAM IM (khong rung, khong truot)
    for f in range(1, IDLE1_END + 1):
        frames[f] = S0['pos'].copy()

    # (2) bam "Do sang hop khac": hat bat dau chay (bat dau tu trang thai nam im S0)
    S2 = copy_state(S0)
    n_pour = POUR_END - POUR_START
    # rung nhe luc dang chay, roi TAT dan o cuoi de hat lang xuong va nam im trong hop
    st = run_frames(S2, POUR_START, n_pour, zero, zero, rng,
                    agit=lambda f: _smooth((f - POUR_START) / 24.0)
                                   * (1.0 - _smooth((f - (POUR_START + 250)) / 40.0)))
    for i in range(n_pour + 1):
        frames[POUR_START + i] = st[i]
    S_after_pour = copy_state(S2)

    # (3) quan sat trong hop: cac hat NAM IM
    for f in range(BOX_IDLE_START, BOX_IDLE_END + 1):
        frames[f] = S_after_pour['pos'].copy()
    return frames, S_after_pour


# ====================== PHAN 2: DUNG CANH TRONG BLENDER ======================
print("Dang mo phong chuyen dong cua %d hat (mat vai giay)..." % N)
frames, _ = build_frames(SEED)
LAST_FRAME = BOX_IDLE_END

# ---------- Don dep canh ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
scene = bpy.context.scene
try:
    scene.view_settings.view_transform = 'Standard'     # mau sac dung (khong bi nhat)
    scene.view_settings.look = 'None'
except Exception:
    pass

# ---------- Vat lieu ----------
def make_mat(name, color, rough=0.4, alpha=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')   # tim theo loai, khong theo ten
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED"),
                          ("show_transparent_back", False)):
            try:
                setattr(m, attr, val)
            except Exception:
                pass
    return m

LIQ     = make_mat("Liquid",   COLOR, 0.2)
GLASS   = make_mat("CupGlass", (0.80, 0.92, 1.00, 1), 0.05, alpha=0.20)
BOXGLS  = make_mat("BoxGlass", (0.95, 0.86, 0.72, 1), 0.05, alpha=0.22)
GROUND  = make_mat("Ground",   (0.93, 0.94, 0.96, 1), 0.9)

# ---------- 70 hat (dung chung mot mesh) ----------
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=R_DRAW, location=(0, 0, 0))
proto = bpy.context.active_object
mesh = proto.data
for poly in mesh.polygons:
    poly.use_smooth = True
mesh.materials.append(LIQ)

root = bpy.data.objects.new("LiquidModel", None)         # vat cha: phong to / dat / xuat AR ca khoi
bpy.context.collection.objects.link(root)
atoms = []
for i in range(N):
    o = bpy.data.objects.new(f"Atom_{i:02d}", mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    atoms.append(o)
bpy.data.objects.remove(proto)

# ---------- Coc (hinh non cut, ho mieng) ----------
cup = bpy.data.objects.new("Cup", None)                  # truc coc: chuyen dong cua coc duoc ghi o day
bpy.context.collection.objects.link(cup)
cup.parent = root
bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=CUP_R1, radius2=CUP_R2, depth=CUP_H,
                                location=(0, 0, CUP_H / 2))
cup_mesh_obj = bpy.context.active_object
cup_mesh_obj.name = "Cup_Glass"
bm = bmesh.new()
bm.from_mesh(cup_mesh_obj.data)
bm.normal_update()
top_faces = [f for f in bm.faces if f.normal.z > 0.9 and f.calc_center_median().z > CUP_H * 0.9]
bmesh.ops.delete(bm, geom=top_faces, context='FACES')    # bo nap tren -> coc ho mieng
bm.to_mesh(cup_mesh_obj.data)
bm.free()
for poly in cup_mesh_obj.data.polygons:
    poly.use_smooth = True
sol = cup_mesh_obj.modifiers.new("Solidify", 'SOLIDIFY')  # tao do day thanh coc, mat trong khop voi thanh mo phong
sol.thickness = 0.04
sol.offset = 1.0
cup_mesh_obj.data.materials.append(GLASS)
cup_mesh_obj.parent = cup

# ---------- Hop (5 tam kinh) ----------
box = bpy.data.objects.new("Box", None)
bpy.context.collection.objects.link(box)
box.parent = root
def wall(name, c, h):
    bpy.ops.mesh.primitive_cube_add(size=2, location=c)
    o = bpy.context.active_object
    o.name = name
    o.scale = h
    o.data.materials.append(BOXGLS)
    o.parent = box
tk = 0.03
sx = (BOX_X1 - BOX_X0) / 2
wall("Box_Floor", (BOX_XC, 0, -tk), (sx + 2 * tk, BOX_Y + 2 * tk, tk))
wall("Box_Left",  (BOX_X0 - tk, 0, BOX_H / 2), (tk, BOX_Y + 2 * tk, BOX_H / 2))
wall("Box_Right", (BOX_X1 + tk, 0, BOX_H / 2), (tk, BOX_Y + 2 * tk, BOX_H / 2))
wall("Box_Front", (BOX_XC, -(BOX_Y + tk), BOX_H / 2), (sx + 2 * tk, tk, BOX_H / 2))
wall("Box_Back",  (BOX_XC, (BOX_Y + tk), BOX_H / 2), (sx + 2 * tk, tk, BOX_H / 2))

# ---------- San, anh sang, camera ----------
bpy.ops.mesh.primitive_plane_add(size=18, location=(1.6, 0, -0.07))
bpy.context.active_object.data.materials.append(GROUND)

bpy.ops.object.light_add(type='SUN', location=(5, -6, 10))
sun = bpy.context.active_object
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
sun.data.energy = 3

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(1.6, 0, 2.2))
target = bpy.context.active_object
target.name = "CamTarget"
target.empty_display_size = 0.2
bpy.ops.object.camera_add(location=(1.6, -11.5, 2.4))
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

# ---------- Ghi keyframe: 70 hat + coc ----------
scene.render.fps = FPS
prefs = bpy.context.preferences.edit
try:
    old_interp = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = 'LINEAR'     # tranh qua da giua cac key
except Exception:
    old_interp = None

for f in range(1, LAST_FRAME + 1):
    P = frames[f]
    for i, o in enumerate(atoms):
        o.location = (float(P[i][0]), float(P[i][1]), float(P[i][2]))
        o.keyframe_insert("location", frame=f)

def key_cup(f):
    loc, _, th = cup_loc_rot(f)
    cup.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    cup.rotation_euler = (0.0, float(th), 0.0)
    cup.keyframe_insert("location", frame=f)
    cup.keyframe_insert("rotation_euler", frame=f)

key_cup(1)
key_cup(IDLE1_END)
for f in range(POUR_START, POUR_END + 1):
    key_cup(f)
key_cup(LAST_FRAME)

if old_interp is not None:
    try:
        prefs.keyframe_new_interpolation_type = old_interp
    except Exception:
        pass

# ====================== NUT BAM (bang dieu khien trong khung nhin 3D) ======================
STATE = {"mode": "cup"}

def _play(context):
    try:
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    except Exception:
        pass

class LIQUID_OT_pour(bpy.types.Operator):
    bl_idname = "liquid.pour"
    bl_label = "Đổ sang hộp khác"
    bl_description = "Đổ chất lỏng từ cốc sang hộp: hình dạng đổi theo vật chứa, thể tích không đổi"
    def execute(self, context):
        STATE["mode"] = "pour"
        sc = context.scene
        sc.frame_start = POUR_START
        sc.frame_end = BOX_IDLE_END
        sc.frame_current = POUR_START
        _play(context)
        return {'FINISHED'}

class LIQUID_OT_reset(bpy.types.Operator):
    bl_idname = "liquid.reset"
    bl_label = "Đặt lại (quan sát trong cốc)"
    bl_description = "Quay về trạng thái ban đầu: các hạt nằm im trong cốc"
    def execute(self, context):
        STATE["mode"] = "cup"
        sc = context.scene
        sc.frame_start = 1
        sc.frame_end = IDLE1_END
        sc.frame_current = 1
        _play(context)
        return {'FINISHED'}

class LIQUID_PT_panel(bpy.types.Panel):
    bl_label = "Mô hình thể lỏng"
    bl_idname = "LIQUID_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Thể lỏng"
    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text=f"{N} hạt • thể tích không đổi")
        col.separator()
        col.operator("liquid.pour", icon='PLAY')
        col.operator("liquid.reset", icon='LOOP_BACK')

def liquid_frame_handler(scene):
    # Sau khi do xong, chi lap lai doan "quan sat trong hop" (khong do lai)
    if STATE["mode"] == "pour" and scene.frame_current >= BOX_IDLE_START \
            and scene.frame_start != BOX_IDLE_START:
        scene.frame_start = BOX_IDLE_START

for nm in ("LIQUID_OT_pour", "LIQUID_OT_reset", "LIQUID_PT_panel"):   # chay lai script khong bi trung
    old = getattr(bpy.types, nm, None)
    if old is not None:
        try:
            bpy.utils.unregister_class(old)
        except Exception:
            pass
for cls in (LIQUID_OT_pour, LIQUID_OT_reset, LIQUID_PT_panel):
    bpy.utils.register_class(cls)
for h in list(bpy.app.handlers.frame_change_post):
    if getattr(h, "__name__", "") == "liquid_frame_handler":
        bpy.app.handlers.frame_change_post.remove(h)
bpy.app.handlers.frame_change_post.append(liquid_frame_handler)

# ---------- Bat che do xem co mau + nhin qua camera ----------
try:
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            sp = area.spaces.active
            sp.shading.type = 'MATERIAL'
            sp.region_3d.view_perspective = 'CAMERA'
except Exception:
    pass

# ---------- Trang thai ban dau: quan sat trong coc (lap) ----------
scene.frame_start = 1
scene.frame_end = IDLE1_END
scene.frame_set(1)

# ---------- (Tuy chon) Xuat GLB cho AR: giu nguyen animation ----------
if EXPORT_GLB:
    scene.frame_start, scene.frame_end = 1, LAST_FRAME
    bpy.ops.export_scene.gltf(filepath=EXPORT_GLB, export_format='GLB', export_animations=True, export_apply=True)
    scene.frame_start, scene.frame_end = 1, IDLE1_END

print("Xong! Bam N trong khung nhin 3D -> tab 'Thể lỏng' -> nut 'Đổ sang hộp khác'.")
print("Cac doan animation (frame): quan sat coc %d-%d (lap) | do %d-%d | quan sat hop %d-%d (lap)"
      % (1, IDLE1_END, POUR_START, POUR_END, BOX_IDLE_START, BOX_IDLE_END))
