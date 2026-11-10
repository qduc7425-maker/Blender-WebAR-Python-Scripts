import bpy
import bmesh
import math
import numpy as np          # Blender co san numpy

FPS = 24
BLUE   = (0.10, 0.35, 0.95, 1)     # the ran / lanh
ORANGE = (1.00, 0.68, 0.30, 1)     # the long (cam nhat)
GAS    = (1.00, 0.42, 0.12, 1)     # the khi (nong)

def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)

def lerp_col(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

# ====================== MO PHONG (khong dung bpy) ======================
def new_state(pos, r, lo, hi):
    n = len(pos)
    return dict(pos=np.array(pos, float), vel=np.zeros((n, 3)), r=np.array(r, float),
                lo=np.array(lo, float), hi=np.array(hi, float), bouncy=np.zeros(n, bool))

def copy_state(S):
    return {k: v.copy() for k, v in S.items()}

def pbd_step(S, dt, acc, damp, iters=3, pair_mask=None, speed=None, extra=None):
    """Mot buoc mo phong: gia toc -> di chuyen -> day cac hat khoi nhau -> tuong. lo/hi: gioi han tam hat tung hat."""
    pos, vel, r = S['pos'], S['vel'], S['r']
    n = len(pos)
    vel += acc * dt
    vel *= np.exp(-np.asarray(damp, float) * dt).reshape(-1, 1) if np.ndim(damp) else math.exp(-damp * dt)
    vpre = vel.copy()
    old = pos.copy()
    pos += vel * dt
    hit_lo = np.zeros((n, 3), bool)
    hit_hi = np.zeros((n, 3), bool)
    eye = np.eye(n) * 1e9
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d * d).sum(-1)) + eye + 1e-12
        ov = r[:, None] + r[None, :] - dist
        m = ov > 0
        if pair_mask is not None:
            m &= pair_mask
        if m.any():
            corr = (np.where(m, ov, 0.0) * 0.5)[..., None] * (d / dist[..., None])
            pos += corr.sum(1)
        if extra is not None:
            extra(S)
        hit_lo |= pos < S['lo']
        hit_hi |= pos > S['hi']
        np.clip(pos, S['lo'], S['hi'], out=pos)
    vnew = (pos - old) / dt
    b = S['bouncy']
    if b.any():
        sel_lo = hit_lo & b[:, None]
        sel_hi = hit_hi & b[:, None]
        vnew[sel_lo] = np.abs(vpre[sel_lo])
        vnew[sel_hi] = -np.abs(vpre[sel_hi])
    S['vel'][:] = vnew
    if speed is not None:
        sp = np.linalg.norm(S['vel'], axis=1) + 1e-9
        k = np.where(speed > 0, 0.15 * (speed / sp - 1.0), 0.0)
        S['vel'] *= (1.0 + k)[:, None]

def mirror_extend(frames, f0, f1):
    """Doan f0..f1 da co -> them f1+1 .. 2*f1-f0-1 la doan chay nguoc -> vong lap khong giat. Tra ve frame cuoi."""
    n = f1 - f0
    for i in range(1, n):
        frames[f1 + i] = frames[f1 - i]
    return f1 + n - 1

def snap(S, col, scl=None):
    n = len(S['pos'])
    return dict(pos=S['pos'].copy(), col=np.array(col, float).copy(),
                scl=np.ones(n) if scl is None else np.array(scl, float).copy())

# ====================== DUNG CANH TRONG BLENDER ======================
def setup_scene(bg=(0.93, 0.95, 1.0, 1)):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    sc = bpy.context.scene
    try:
        sc.view_settings.view_transform = 'Standard'
        sc.view_settings.look = 'None'
    except Exception:
        pass
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    n = next(n for n in sc.world.node_tree.nodes if n.type == 'BACKGROUND')
    n.inputs["Color"].default_value = bg
    sc.render.fps = FPS
    return sc

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
    try:
        m.diffuse_color = color
    except Exception:
        pass
    return m

def bsdf_of(m):
    return next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')

def parent_to(o, p):
    bpy.context.view_layer.update()
    o.parent = p
    o.matrix_parent_inverse = p.matrix_world.inverted()

def add_particles(root, n, name="Atom", segs=24):
    """n hat cau dung chung 1 mesh ban kinh 1; moi hat 1 vat lieu rieng de doi mau duoc."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=segs // 2 + 2, radius=1.0, location=(0, 0, 0))
    proto = bpy.context.active_object
    mesh = proto.data
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.materials.append(make_mat("tmp", BLUE))
    objs, mats = [], []
    for i in range(n):
        o = bpy.data.objects.new(f"{name}_{i:02d}", mesh)
        bpy.context.collection.objects.link(o)
        o.parent = root
        m = make_mat(f"{name}Mat_{i:02d}", BLUE, 0.3)
        o.material_slots[0].link = 'OBJECT'
        o.material_slots[0].material = m
        objs.append(o)
        mats.append(m)
    bpy.data.objects.remove(proto)
    return objs, mats

def add_box(name, parent, c, h, mat):
    """Hop chu nhat: tam (c), nua kich thuoc (h)."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=c)
    o = bpy.context.active_object
    o.name = name
    o.scale = h
    o.data.materials.append(mat)
    if parent is not None:
        o.parent = parent
    return o

def glass_dish(name, parent, x0, x1, y0, y1, z0, z1, mat, tk=0.05):
    """Khay/coc thuy tinh hop chu nhat ho mieng: san + 4 thanh."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = (x1 - x0) / 2, (y1 - y0) / 2
    hz = (z1 - z0) / 2
    add_box(name + "_Floor", parent, (cx, cy, z0 - tk), (sx + 2 * tk, sy + 2 * tk, tk), mat)
    add_box(name + "_L", parent, (x0 - tk, cy, z0 + hz), (tk, sy + 2 * tk, hz), mat)
    add_box(name + "_R", parent, (x1 + tk, cy, z0 + hz), (tk, sy + 2 * tk, hz), mat)
    add_box(name + "_F", parent, (cx, y0 - tk, z0 + hz), (sx + 2 * tk, tk, hz), mat)
    add_box(name + "_B", parent, (cx, y1 + tk, z0 + hz), (sx + 2 * tk, tk, hz), mat)

def add_ground_light_camera(sc, gz, cam_loc, tgt, lens=35, gsize=40):
    bpy.ops.mesh.primitive_plane_add(size=gsize, location=(tgt[0], 0, gz))
    bpy.context.active_object.data.materials.append(make_mat("Ground", (0.93, 0.94, 0.96, 1), 0.9))
    bpy.ops.object.light_add(type='SUN', location=(5, -6, 12))
    sun = bpy.context.active_object
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sun.data.energy = 3
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=tgt)
    t = bpy.context.active_object
    t.name = "CamTarget"
    t.empty_display_size = 0.2
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    cam.data.lens = lens
    con = cam.constraints.new('TRACK_TO')
    con.target = t
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    sc.camera = cam
    return cam, t

def linear_keys_on():
    prefs = bpy.context.preferences.edit
    try:
        old = prefs.keyframe_new_interpolation_type
        prefs.keyframe_new_interpolation_type = 'LINEAR'
        return old
    except Exception:
        return None

def linear_keys_off(old):
    if old is not None:
        try:
            bpy.context.preferences.edit.keyframe_new_interpolation_type = old
        except Exception:
            pass

def bake_particles(objs, mats, radius, frames, f_first, f_last, root_scale=1.0):
    """Ghi keyframe: vi tri moi frame; mau va kich co chi ghi khi thay doi."""
    old = linear_keys_on()
    n = len(objs)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    last_col = [None] * n
    last_scl = [None] * n
    bs = [bsdf_of(m) for m in mats]
    for f in range(f_first, f_last + 1):
        F = frames[f]
        for i, o in enumerate(objs):
            P = F['pos'][i]
            o.location = (float(P[0]), float(P[1]), float(P[2]))
            o.keyframe_insert("location", frame=f)
            c = tuple(float(x) for x in F['col'][i])
            if last_col[i] is None or max(abs(c[k] - last_col[i][1][k]) for k in range(4)) > 0.004 or f == f_last:
                if last_col[i] is not None and last_col[i][0] < f - 1:
                    inp = bs[i].inputs["Base Color"]
                    inp.default_value = last_col[i][1]
                    inp.keyframe_insert("default_value", frame=f - 1)
                inp = bs[i].inputs["Base Color"]
                inp.default_value = c
                inp.keyframe_insert("default_value", frame=f)
                try:
                    mats[i].diffuse_color = c
                    mats[i].keyframe_insert("diffuse_color", frame=f)
                except Exception:
                    pass
                last_col[i] = (f, c)
            s = float(F['scl'][i] * r[i])
            if last_scl[i] is None or abs(s - last_scl[i][1]) > 1e-4 or f == f_last:
                if last_scl[i] is not None and last_scl[i][0] < f - 1:
                    o.scale = (last_scl[i][1],) * 3
                    o.keyframe_insert("scale", frame=f - 1)
                o.scale = (s, s, s)
                o.keyframe_insert("scale", frame=f)
                last_scl[i] = (f, s)
    linear_keys_off(old)

def key_value(obj, attr, frame, value):
    setattr(obj, attr, value)
    obj.keyframe_insert(attr, frame=frame)

# ====================== NUT BAM / BANG DIEU KHIEN ======================
STATE = {"btn": None, "busy": False}

def _play(context):
    try:
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    except Exception:
        pass

def register_ui(prefix, tab, title, buttons, info=None, slider=None, full=(1, 100), first=None):
    """buttons: [dict(id,label,desc,icon,start,end,loop)]; loop = frame bat dau doan lap cuoi (tuy chon).
       slider: dict(label,f0,f1) keo de 'tua' animation; info(frame)->list[str]."""
    key = prefix + "_t"
    P = prefix.upper()
    names = []

    def mk_op(b):
        def execute(self, context):
            sc = context.scene
            STATE["btn"] = b
            sc.frame_start, sc.frame_end, sc.frame_current = b["start"], b["end"], b["start"]
            _play(context)
            return {'FINISHED'}
        cls = type(f"{P}_OT_{b['id']}", (bpy.types.Operator,),
                   dict(bl_idname=f"{prefix}.{b['id']}", bl_label=b["label"],
                        bl_description=b.get("desc", ""), execute=execute))
        return cls

    classes = [mk_op(b) for b in buttons]

    def on_slider(self, context):
        if STATE["busy"]:
            return
        sc = context.scene
        try:
            if sc.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        except Exception:
            pass
        STATE["btn"] = None
        sc.frame_start, sc.frame_end = full
        f0, f1 = slider["f0"], slider["f1"]
        STATE["busy"] = True
        sc.frame_set(int(round(f0 + float(sc[key]) * (f1 - f0))))
        STATE["busy"] = False

    def draw(self, context):
        col = self.layout.column(align=True)
        if info is not None:
            for line in info(context.scene.frame_current):
                col.label(text=line)
            col.separator()
        if slider is not None:
            col.prop(context.scene, key, text=slider["label"], slider=True)
            col.separator()
        for b in buttons:
            col.operator(f"{prefix}.{b['id']}", icon=b.get("icon", 'PLAY'))
    classes.append(type(f"{P}_PT_panel", (bpy.types.Panel,),
                        dict(bl_label=title, bl_idname=f"{P}_PT_panel", bl_space_type='VIEW_3D',
                             bl_region_type='UI', bl_category=tab, draw=draw)))

    def handler(scene):
        f = scene.frame_current
        b = STATE.get("btn")
        if b and b.get("loop") and f >= b["loop"] and scene.frame_start != b["loop"]:
            scene.frame_start = b["loop"]
        if slider is not None and not STATE["busy"]:
            STATE["busy"] = True
            try:
                scene[key] = min(max((f - slider["f0"]) / float(slider["f1"] - slider["f0"]), 0.0), 1.0)
            except Exception:
                pass
            STATE["busy"] = False
    handler.__name__ = prefix + "_frame_handler"

    if slider is not None:
        setattr(bpy.types.Scene, key, bpy.props.FloatProperty(name=slider["label"], min=0.0, max=1.0,
                                                              default=0.0, update=on_slider))
    for c in classes:                                    # chay lai script khong bi trung
        old = getattr(bpy.types, c.__name__, None)
        if old is not None:
            try:
                bpy.utils.unregister_class(old)
            except Exception:
                pass
        bpy.utils.register_class(c)
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == handler.__name__:
            bpy.app.handlers.frame_change_post.remove(h)
    bpy.app.handlers.frame_change_post.append(handler)

def finish_view():
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                sp = area.spaces.active
                sp.shading.type = 'MATERIAL'
                sp.region_3d.view_perspective = 'CAMERA'
    except Exception:
        pass


# =====================================================================
#  MO HINH 6 (Trang 11) - SU BAY HOI: cac hat o lop BE MAT (cam nhat) nhan nhiet,
#  tang toc dot ngot, bat khoi dam chat long bay vut len thanh hat KHI, tan mac vao khong gian.
#  Cach dung: Run Script -> bam N trong khung nhin 3D -> tab "Bay hoi" -> nut "Chay"
# =====================================================================
N       = 60                         # so hat chat long luc dau
R_P     = 0.30
SEED    = 2
BX, BY  = 1.5, 1.5                   # nua kich thuoc long coc (hinh hop)
BH      = 3.4                        # chieu cao coc
ROOM_X, ROOM_Y, ROOM_Z = 5.5, 3.0, 9.0   # khong gian ben ngoai (gioi han hat khi)
N_EVAP  = 20                         # so hat bay hoi
EV_FIRST, EV_GAP, CHARGE = 20, 14, 8     # hat dau tien bat dau "tich nhiet" o frame 20, cach nhau 14 frame
EXPORT_GLB = ""                      # vd "C:/tmp/bay_hoi.glb" de xuat cho AR (de trong = khong xuat)

SUB = 4
SIM_END  = 420
LOOP_FROM = 301                      # 301..420 lap xuoi + nguoc (cac hat khi bay tan mac)
GAS_SPEED = 3.2

def build_frames(seed=SEED):
    rng = np.random.default_rng(seed)
    P0 = []
    while len(P0) < N:
        c = np.array([rng.uniform(-BX + R_P, BX - R_P), rng.uniform(-BY + R_P, BY - R_P), rng.uniform(R_P, 3.0)])
        if all(np.linalg.norm(c - q) > 2 * R_P * 1.05 for q in P0):
            P0.append(c)
    lo = np.tile([-BX + R_P, -BY + R_P, R_P], (N, 1))
    hi = np.tile([BX - R_P, BY - R_P, 60.0], (N, 1))
    S = new_state(P0, [R_P] * N, lo, hi)
    state = np.zeros(N, int)                    # 0 = long, 1 = dang tich nhiet, 2 = khi
    prog = np.zeros(N)                          # tien do doi mau (0..1)
    dt = 1.0 / (FPS * SUB)
    ns = np.zeros((N, 3))
    events = {EV_FIRST + EV_GAP * k: k for k in range(N_EVAP)}        # frame -> thu tu
    chosen = {}
    def set_bounds():
        for i in range(N):
            if state[i] == 2:
                x, y, z = S['pos'][i]
                inside = abs(x) <= BX - R_P + 1e-6 and abs(y) <= BY - R_P + 1e-6
                if z < BH + R_P and inside:                                   # con ben trong coc
                    S['lo'][i] = [-BX + R_P, -BY + R_P, R_P]
                    S['hi'][i] = [BX - R_P, BY - R_P, ROOM_Z]
                else:                                                         # da ra ngoai mieng coc
                    S['lo'][i] = [-ROOM_X, -ROOM_Y, BH + R_P if not inside else R_P]
                    S['hi'][i] = [ROOM_X, ROOM_Y, ROOM_Z]
            else:
                S['lo'][i] = [-BX + R_P, -BY + R_P, R_P]
                S['hi'][i] = [BX - R_P, BY - R_P, 60.0]
    def one_frame(f, do_events=True):
        for s in range(SUB):
            ns[:] = 0.8 * ns + 0.2 * rng.normal(size=(N, 3))
            acc = ns * 20.0
            liq = state == 0
            acc[:, 2] -= 11.0 * (state != 2)
            ch = state == 1
            acc[ch] = ns[ch] * 90.0
            acc[ch, 2] += 22.0                                              # tang toc huong len
            speed = np.where(state == 2, GAS_SPEED, 0.0)
            S['bouncy'][:] = state == 2
            damp = np.where(state == 2, 0.0, np.where(ch, 0.5, 2.0))
            set_bounds()
            pbd_step(S, dt, acc, damp, speed=speed)
        # chuyen trang thai
        for i in np.where(state == 1)[0]:
            prog[i] = min(1.0, prog[i] + 1.0 / (CHARGE * 1.0))
    frames = {}
    for _ in range(140):                                                     # khoi dong: chat long lang xuong
        one_frame(0)
    for f in range(1, SIM_END + 1):
        if f + CHARGE in events:                                              # chon hat o be mat de tich nhiet
            cand = [i for i in range(N) if state[i] == 0]
            top = max(cand, key=lambda i: S['pos'][i][2])
            state[top] = 1
            chosen[events[f + CHARGE]] = top
        if f in events:                                                      # BUNG RA: bay vut len
            pid = chosen[events[f]]
            state[pid] = 2
            prog[pid] = 1.0
            S['vel'][pid] = [rng.normal(0, 0.5), rng.normal(0, 0.5), 5.0]
        one_frame(f)
        col = np.array([lerp_col(ORANGE, GAS, prog[i]) for i in range(N)])
        frames[f] = snap(S, col)
        frames[f]['n_gas'] = int((state == 2).sum())
    last = mirror_extend(frames, LOOP_FROM, SIM_END)
    return frames, last

# ====================== PHAN 2: DUNG CANH TRONG BLENDER ======================
print("Dang mo phong...")
frames, LAST_FRAME = build_frames(SEED)
sc = setup_scene()
GLASS = make_mat("Beaker", (0.85, 0.93, 1.0, 1), 0.05, alpha=0.20)
root = bpy.data.objects.new("EvapModel", None)
bpy.context.collection.objects.link(root)
objs, mats = add_particles(root, N)
glass_dish("Beaker", root, -BX, BX, -BY, BY, 0.0, BH, GLASS)
SUNM = make_mat("Sun", (1.0, 0.85, 0.2, 1), 0.3)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, location=(-5.0, 2.0, 8.2))
bpy.context.active_object.data.materials.append(SUNM)
bpy.context.active_object.name = "Sun"
bpy.context.active_object.parent = root
for k in range(8):                                                           # tia nang
    a = k * math.pi / 4
    bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.12, depth=0.7,
                                    location=(-5.0 + 1.45 * math.cos(a), 2.0, 8.2 + 1.45 * math.sin(a)),
                                    rotation=(0, math.radians(90) - a, 0))
    bpy.context.active_object.data.materials.append(SUNM)
    bpy.context.active_object.parent = root
cam, tgt = add_ground_light_camera(sc, -0.1, (0.5, -17.0, 4.8), (0.3, 0, 3.8), 35)

bake_particles(objs, mats, R_P, frames, 1, LAST_FRAME)

def info(f):
    n = frames[min(max(f, 1), LAST_FRAME)]['n_gas']
    return ["Hạt cam = chất lỏng, hạt đỏ-cam = hơi", f"Đã bay hơi: {n}/{N_EVAP} hạt"]

register_ui("m6", "Bay hơi", "Sự bay hơi", [
    dict(id="run", label="Chạy", desc="Các hạt ở bề mặt lần lượt bứt phá bay lên",
         start=1, end=LAST_FRAME, loop=LOOP_FROM),
    dict(id="reset", label="Đặt lại", desc="Về trạng thái ban đầu", icon='LOOP_BACK', start=1, end=EV_FIRST - CHARGE - 2),
], info=info, full=(1, LAST_FRAME))
sc.frame_start, sc.frame_end = 1, EV_FIRST - CHARGE - 2
sc.frame_set(1)
finish_view()
if EXPORT_GLB:
    sc.frame_start, sc.frame_end = 1, LAST_FRAME
    bpy.ops.export_scene.gltf(filepath=EXPORT_GLB, export_format='GLB', export_animations=True, export_apply=True)
    sc.frame_start, sc.frame_end = 1, EV_FIRST - CHARGE - 2
print("Xong! Bam N -> tab 'Bay hơi' -> nut 'Chạy'.")
