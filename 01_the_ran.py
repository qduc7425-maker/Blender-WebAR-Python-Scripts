import bpy
import math
import random

# =====================================================
#  MO HINH THE RAN (Trang 5): 80 hat cau xep luoi, rung tai cho, mau XANH (lanh)
# =====================================================

# ---------- Thong so (doi o day) ----------
NX, NY, NZ = 5, 4, 4          # luoi 5 x 4 x 4 = 80 hat
SPACING    = 1.0              # khoang cach tam-tam giua 2 hat canh nhau
RADIUS     = 0.48             # ban kinh hat (gan SPACING/2 de xep khit)
AMP        = 0.025            # bien do rung (nho de hat khong xuyen vao nhau)
PERIOD     = 48               # chu ky lap (frame); 24 fps -> 2 giay
STEP       = 2                # moi STEP frame dat 1 key
COLOR      = (0.10, 0.35, 0.95, 1)   # xanh duong (nhiet do thap)
SEED       = 7                # doi so nay de kieu rung khac di
EXPORT_GLB = ""               # vi du "C:/tmp/the_ran.glb" de xuat file cho AR (de trong = khong xuat)

# ---------- Don dep canh ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Mau sac dung (khong bi nhat)
try:
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
except Exception:
    pass

# ---------- Vat lieu xanh ----------
mat = bpy.data.materials.new("Blue")
mat.use_nodes = True
bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')   # tim theo loai, khong theo ten
bsdf.inputs["Base Color"].default_value = COLOR
bsdf.inputs["Roughness"].default_value = 0.35

# ---------- Mot mau hat dung chung cho ca 80 hat ----------
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=RADIUS, location=(0, 0, 0))
proto = bpy.context.active_object
mesh = proto.data
for p in mesh.polygons:
    p.use_smooth = True
mesh.materials.append(mat)

# Truc cha: dung de dat / phong to / xuat AR ca khoi cung luc
root = bpy.data.objects.new("SolidModel", None)
bpy.context.collection.objects.link(root)

# ---------- Dung luoi hat (can giua goc toa do) ----------
rng = random.Random(SEED)
particles = []
for ix in range(NX):
    for iy in range(NY):
        for iz in range(NZ):
            rest = ((ix - (NX - 1) / 2) * SPACING,
                    (iy - (NY - 1) / 2) * SPACING,
                    (iz - (NZ - 1) / 2) * SPACING)
            o = bpy.data.objects.new(f"Atom_{ix}_{iy}_{iz}", mesh)   # dung chung mesh
            bpy.context.collection.objects.link(o)
            o.parent = root
            o.location = rest
            particles.append((o, rest))

# xoa hat mau ban dau (mesh van duoc cac hat tren dung)
bpy.data.objects.remove(proto)

# ---------- Rung tai cho: moi hat 1 pha + tan so rieng, lap hoan hao ----------
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = PERIOD       # frame PERIOD+1 trung frame 1 nen lap khong giat

for o, rest in particles:
    fx, fy, fz = (rng.choice((3, 4)) for _ in range(3))              # so chu ky nguyen / PERIOD
    px, py, pz = (rng.random() for _ in range(3))
    ax, ay, az = (AMP * rng.uniform(0.7, 1.0) for _ in range(3))
    for f in range(1, PERIOD + 2, STEP):
        t = (f - 1) / PERIOD
        o.location = (rest[0] + ax * math.sin(2 * math.pi * (fx * t + px)),
                      rest[1] + ay * math.sin(2 * math.pi * (fy * t + py)),
                      rest[2] + az * math.sin(2 * math.pi * (fz * t + pz)))
        o.keyframe_insert("location", frame=f)

# ---------- Anh sang, camera, nen ----------
bpy.ops.object.light_add(type='SUN', location=(6, -6, 10))
sun = bpy.context.active_object
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
sun.data.energy = 3

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
target = bpy.context.active_object
target.name = "CamTarget"
target.empty_display_size = 0.2

bpy.ops.object.camera_add(location=(7.5, -9.5, 5.5))
cam = bpy.context.active_object
con = cam.constraints.new('TRACK_TO')            # camera luon huong ve tam khoi
con.target = target
con.track_axis = 'TRACK_NEGATIVE_Z'
con.up_axis = 'UP_Y'
scene.camera = cam

scene.world.use_nodes = True
bg = next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
bg.inputs["Color"].default_value = (0.93, 0.95, 1.0, 1)

scene.frame_set(1)

# ---------- (Tuy chon) Xuat GLB cho AR: giu nguyen animation rung ----------
if EXPORT_GLB:
    bpy.ops.export_scene.gltf(filepath=EXPORT_GLB, export_format='GLB', export_animations=True)

print(f"Xong! {len(particles)} hat. Bam phim cach (Space) de xem animation.")
