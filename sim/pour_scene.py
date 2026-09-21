#!/usr/bin/env python3
"""Randomised pour scene: one A1X on the table, a bottle, a small glass, and
a laptop webcam at a random, unknown pose.

    sim/.venv/bin/python sim/pour_scene.py            # build seed 0, print the layout
    sim/.venv/bin/python sim/pour_scene.py 3 out.png  # build seed 3, render the laptop cam

`build(seed)` returns `(model, data, info)` and is deterministic for a seed.
The arm mount and the contact recipe come from `pickplace_scene.py`; the bottle
and glass from `bottle.py`; the rest of the scene is drawn per seed.

Everything a camera can see is randomised so a policy trained on these frames
cannot learn one particular table: the table's extents and where the arm sits
on it (the top stays at `TABLE_TOP`, because the mount stands on it), the table
and floor materials, the sky and haze, one to three lights, and zero to two
distractor primitives standing clear of the bottle-to-glass corridor.
`info["randomisation"]` records every draw. The bottle, glass, their placement
and the camera come from a separate RNG stream, so a given seed keeps the task
it had before the scene around it started moving.

The camera stands in for a laptop put down somewhere in front of the arm: its
pose is drawn from `CAM` (distance, bearing and height about the workspace)
and is *not* given to the planner. `info["cam"]` keeps the ground truth so the
calibration in `calib/` can be scored against it.

The default scene carries **no fiducials at all**, so every training frame is
marker free. `build(seed, calib_card=True)` adds the one marker the system
uses: a thin card pinched in the gripper, with an AprilTag 36h11 on each face,
which a human puts in before calibration and takes out afterwards. Its pose in
the gripper is drawn per seed and is *not* given to the solver; `info["calib"]`
keeps it as ground truth to score the recovered mount against.
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from a1x_control import TIP_AHEAD, Arm, rot                                  # noqa: E402
from bottle import add_bottle_meshes, bottle_xml, glass_xml, sample_bottle, sample_glass  # noqa: E402
from pickplace_scene import ARM_BASE, CONTACT, TABLE_TOP, Q_HOME, look_at    # noqa: E402

REGION = dict(x=(-0.18, 0.18), y=(0.03, 0.20))      # where bottle and glass may stand
MIN_APART = 0.11                                    # bottle to glass, centre to centre [m]
CAM = dict(W=640, H=480, fovy=(44.0, 54.0),         # laptop webcam, 4:3
           dist=(0.50, 0.85), bearing=(-1.6, 1.6),  # about the workspace centre, rad off +y
           height=(0.12, 0.38), jitter=0.04)        # lens height above the table, look-at jitter
WORKSPACE = np.array([0.0, 0.10, TABLE_TOP + 0.10])
GRASP_PITCH = 0.9                                   # approach pitch below level [rad]
POUR_TILT, POUR_ABOVE = 1.95, 0.03                 # rad past upright; mouth height over the rim [m]
TAGS_DIR = os.path.join(HERE, "assets", "tags")
TAG_MARGIN = 160 / 200              # marker side / printed plate side in assets/tags/*.png
GRIP_KP, GRIP_CMD = 1500.0, -0.03   # finger servo gain and the 'closed' command, see build()

# ------------------------------------------------------- calibration card
# The only fiducial in the system, and it is in the scene only while
# calibrating. A thin rigid rectangle is pinched at one end between the finger
# pads, so its normal is the closing axis and it sticks out past the fingertips
# along the approach axis. The protruding half carries one 36h11 tag per face,
# back to back, so whichever side the laptop ended up on, one face is readable.
#
# Card frame: x out along the approach, y across the card (gripper -z at the
# nominal pose), z the card normal, origin at the tag centre in the mid plane.
# `CARD["out"]` is how far the card protrudes past the fingertips; the pinched
# part carries no tag. A human puts the card in, so the nominal pose below is
# only a starting guess: `sample_card_pose` perturbs it per seed and the solver
# in `calib/` has to recover the truth from the images.
CARD = dict(L=0.130, W=0.090, T=0.002,      # card length, width, thickness [m]
            tag=0.070, out=0.090,           # marker side [m]; protrusion past the fingertips [m]
            pad=0.008,                      # collision margin around the card [m]
            gap=0.0006,                     # drawn card thinner than this, so the tags are proud
            dpos=0.012, drot=np.radians(10.0))   # per-session misplacement, uniform per axis
TIP_X = 0.045 + TIP_AHEAD                   # fingertip plane in the gripper_link frame [m]
CARD_R_NOM = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
CARD_T_NOM = np.array([TIP_X + CARD["out"] / 2, 0.0, 0.0])
# tag id -> pose in the card frame, exact from the card's geometry: same centre
# in the card plane, opposite normals, the thickness apart.
CARD_TAGS = {0: (np.array([0.0, 0.0, CARD["T"] / 2]), np.eye(3)),
             1: (np.array([0.0, 0.0, -CARD["T"] / 2]), np.diag([-1.0, 1.0, -1.0]))}
_ARM = {}

# The G1 wrist camera (hardware/g1_camera_mounts) is opt-in: it adds 104 g and
# 57 collision primitives to the wrist, so a scene with it and a scene without
# it are different robots. WRIST_CAMERA=left|right turns it on for every
# build() of the process, which is how gen_dataset reaches its workers;
# build(wrist=...) overrides that per call.
WRIST_HAND = os.environ.get("WRIST_CAMERA", "")
WRIST_CAM = "arm/wrist"
WRIST_JITTER = dict(pos_mm=2.0, rot_deg=1.5)             # re-seating play of the bracket and the lens
WRIST_EXPOSURE = dict(gain=(0.75, 1.25), gamma=(0.8, 1.25))   # a wrist camera swings through the light

# ------------------------------------------------------------ randomisation
# A policy trained on these frames must never learn one particular table, so
# every seed draws its own table, surfaces, sky, lights and clutter. Only the
# table *top* is fixed at TABLE_TOP, because ARM_BASE stands on it; the extents
# move, so the arm sits at a different distance from every edge. The draws come
# from a second RNG stream keyed on the seed, so the bottle, glass, placement
# and camera of a given seed are exactly what they were before.
TABLE_MARGIN = 0.06                    # REGION to table edge, at least [m]
TABLE_PAD = dict(x=(0.03, 0.34), y_front=(0.03, 0.34), y_back=(0.08, 0.30))
TABLE_H = 0.35                         # half height; top stays at TABLE_TOP
SURFACE_STYLES = ("colour", "checker", "gradient", "flat")
TEX_MARKS = ("none", "edge", "cross", "random")
LIGHTS = dict(n=(1, 4), total=(0.65, 1.15), height=(0.8, 2.2), spread=1.6,
              headlight_diffuse=(0.28, 0.55), headlight_ambient=(0.18, 0.34),
              headlight_specular=(0.0, 0.10))
DISTRACT = dict(n=(0, 3), clear=0.12, arm_clear=0.16, edge=0.04, view=0.55,
                r=(0.015, 0.045), h=(0.03, 0.12), mass=(0.05, 0.4))


def _wrist_module():
    mount = os.path.join(os.path.dirname(HERE), "hardware", "g1_camera_mounts", "sim")
    if mount not in sys.path:
        sys.path.append(mount)
    import wrist_camera
    return wrist_camera


def _arm_spec(wrist=""):
    """The arm, parsed once per process and per hand the camera is mounted on."""
    if wrist not in _ARM:
        spec = mujoco.MjSpec.from_file(os.path.join(HERE, "a1x.xml"))
        if wrist:
            _wrist_module().attach_wrist_camera(spec, hand=wrist)
        _ARM[wrist] = spec
    return _ARM[wrist]


def _add_tag_assets(spec, ids, plate):
    for i in ids:
        t = spec.add_texture()
        t.name, t.type = f"tag{i}", mujoco.mjtTexture.mjTEXTURE_2D
        t.file = os.path.join(TAGS_DIR, f"tag36h11_{i}.png")
        mat = spec.add_material()
        mat.name = f"tag{i}"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{i}"
        mat.texuniform = False
    # one square plate mesh with texture coordinates, reused by every tag
    m = spec.add_mesh()
    m.name = "tag_plate"
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    h = plate / 2
    m.uservert = np.array([-h, -h, 0, h, -h, 0, h, h, 0, -h, h, 0], np.float32)
    m.userface = np.array([0, 1, 2, 0, 2, 3], np.int32)
    m.usertexcoord = np.array([0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    m.userfacetexcoord = np.array([0, 1, 2, 0, 2, 3], np.int32)


def _add_tag(spec, body, tag_id, pos, R, prefix):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    g = body.add_geom()
    g.name, g.type, g.meshname = f"{prefix}tag{tag_id}", mujoco.mjtGeom.mjGEOM_MESH, "tag_plate"
    g.material = f"tag{tag_id}"
    g.pos, g.quat = pos, q
    g.contype = g.conaffinity = 0
    g.mass = 0.0
    g.group = 1
    s = body.add_site()
    s.name, s.pos, s.quat, s.size = f"{prefix}tag{tag_id}", pos, q, [0.003, 0, 0]
    s.group = 4


def card_nominal():
    """The card's nominal pose in the gripper_link frame, 4x4.

    This is all the solver is allowed to know: the card square to the hand,
    sticking `CARD['out']` past the fingertips with its normal along the
    closing axis, so the tag centre is half of that beyond the tips. What a
    hand actually does with it is `sample_card_pose`."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = CARD_R_NOM, CARD_T_NOM
    return T


def sample_card_pose(rng, extreme=False):
    """Where the card really ended up: the nominal pose times a per-session
    misplacement, uniform in +-`CARD['dpos']` on each card axis and
    +-`CARD['drot']` about each. `extreme` draws at the limits instead."""
    def draw(lim):
        if not extreme:
            return rng.uniform(-lim, lim, 3)
        return np.sign(rng.uniform(-1, 1, 3)) * lim * rng.uniform(0.95, 1.0, 3)

    d = np.eye(4)
    rv = draw(CARD["drot"])
    n = float(np.linalg.norm(rv))
    q, R = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, rv / (n + 1e-12), n)
    mujoco.mju_quat2Mat(R, q)
    d[:3, :3] = R.reshape(3, 3)
    d[:3, 3] = draw(CARD["dpos"])
    return card_nominal() @ d


def card_tag_poses():
    """{tag id: 4x4 pose of the tag frame in the card frame}, exact geometry."""
    out = {}
    for tid, (p, R) in CARD_TAGS.items():
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, p
        out[tid] = T
    return out


def _add_card(spec, body, T_card2frame):
    """The pinched card: a visible box, a padded invisible collider so the wave
    keeps real clearance, and one tag plate per face."""
    _add_tag_assets(spec, list(CARD_TAGS), CARD["tag"] / TAG_MARGIN)
    Rc, tc = np.asarray(T_card2frame)[:3, :3], np.asarray(T_card2frame)[:3, 3]
    qc = np.zeros(4); mujoco.mju_mat2Quat(qc, Rc.ravel())
    # the box centre sits back from the tag centre: only `out` of the card
    # protrudes past the fingertips, the rest is between the pads.
    centre = tc + Rc @ [CARD["out"] / 2 - CARD["L"] / 2, 0.0, 0.0]
    half = np.array([CARD["L"] / 2, CARD["W"] / 2, CARD["T"] / 2])
    # The tag plates sit on the card's two faces, so the drawn card is made a
    # shade thinner than the real one: coplanar surfaces z-fight, and the card
    # wins often enough that the detector sees a white rectangle.
    for name, grow, rgba, group, con in (("calib_card", -CARD["gap"], [0.97, 0.97, 0.97, 1], 1, 0),
                                         ("calib_card_pad", CARD["pad"], [0, 0, 0, 0], 3, 1)):
        g = body.add_geom()
        g.name, g.type = f"arm/{name}", mujoco.mjtGeom.mjGEOM_BOX
        g.size = half + [0.0, 0.0, grow] if grow < 0 else half + grow
        g.pos, g.quat, g.rgba, g.group, g.mass = centre, qc, rgba, group, 0.0
        g.contype = g.conaffinity = con
    for tid, T in card_tag_poses().items():
        _add_tag(spec, body, tid, tc + Rc @ T[:3, 3], Rc @ T[:3, :3], "arm/")


def _split_finger_pads(spec):
    """Give each finger plate and tip several collision boxes.

    MuJoCo's convex collider returns one contact point per box-cylinder pair,
    so a plate pressed flat on a bottle is a single point and the bottle can
    pivot freely about the line between the two fingers: a full bottle swings
    out of the hand the moment it tilts. Splitting each pad into a grid of
    boxes gives a real contact patch, which is what a rubber pad does."""
    for f in (1, 2):
        body = spec.body(f"arm/gripper_finger_link{f}")
        for g in list(body.geoms):
            if g.type != mujoco.mjtGeom.mjGEOM_BOX or g.contype == 0:
                continue
            sx, sy, sz = g.size
            nx, nz = (3, 2) if sx > 0.02 else (1, 2) if sx < 0.006 else (1, 1)
            if nx * nz == 1:
                continue
            g.contype = g.conaffinity = 0
            for i in range(nx):
                for k in range(nz):
                    h = body.add_geom()
                    h.type, h.classname = mujoco.mjtGeom.mjGEOM_BOX, g.classname
                    h.size = [sx / nx, sy, sz / nz]
                    h.pos = g.pos + [(2 * i + 1 - nx) * sx / nx, 0, (2 * k + 1 - nz) * sz / nz]
                    h.group = g.group


def _f(v):
    return " ".join(f"{x:.4f}" for x in np.asarray(v, float).ravel())


def _sample_table(rng):
    """Table extents and centre, with the top fixed at TABLE_TOP.

    The whole REGION keeps `TABLE_MARGIN` of table around it and the arm mount
    keeps `TABLE_PAD['y_back']` behind it; everything past that is drawn, so the
    arm stands anywhere from 0.1 m to 0.4 m from each edge."""
    x0 = REGION["x"][0] - TABLE_MARGIN - rng.uniform(*TABLE_PAD["x"])
    x1 = REGION["x"][1] + TABLE_MARGIN + rng.uniform(*TABLE_PAD["x"])
    y1 = REGION["y"][1] + TABLE_MARGIN + rng.uniform(*TABLE_PAD["y_front"])
    y0 = ARM_BASE[1] - rng.uniform(*TABLE_PAD["y_back"])
    half = np.array([(x1 - x0) / 2, (y1 - y0) / 2, TABLE_H])
    pos = np.array([(x0 + x1) / 2, (y0 + y1) / 2, TABLE_TOP - TABLE_H])
    return dict(pos=pos, half=half, x=(x0, x1), y=(y0, y1))


def _surface(rng, name):
    """Asset XML for one large surface: a flat colour, or a builtin MuJoCo
    texture (checker / gradient / flat) with random colours and repeat."""
    style = SURFACE_STYLES[rng.integers(len(SURFACE_STYLES))]
    refl = float(rng.uniform(0.0, 0.3))
    if style == "colour":
        rgba = np.concatenate([rng.uniform(0.05, 0.95, 3), [1.0]])
        return (f'<material name="{name}" rgba="{_f(rgba)}" reflectance="{refl:.3f}"/>',
                dict(style=style, rgba=rgba.round(3).tolist(), reflectance=round(refl, 3)))
    rgb1, rgb2 = rng.uniform(0.03, 0.95, 3), rng.uniform(0.03, 0.95, 3)
    markrgb = rng.uniform(0.0, 1.0, 3)
    mark = TEX_MARKS[rng.integers(len(TEX_MARKS))]
    rep = rng.uniform(1.0, 10.0, 2)
    xml = (f'<texture type="2d" name="{name}_tex" builtin="{style}" mark="{mark}" '
           f'rgb1="{_f(rgb1)}" rgb2="{_f(rgb2)}" markrgb="{_f(markrgb)}" width="300" height="300"/>'
           f'<material name="{name}" texture="{name}_tex" texuniform="true" '
           f'texrepeat="{rep[0]:.2f} {rep[1]:.2f}" reflectance="{refl:.3f}"/>')
    return xml, dict(style=style, mark=mark, rgb1=rgb1.round(3).tolist(), rgb2=rgb2.round(3).tolist(),
                     markrgb=markrgb.round(3).tolist(), texrepeat=rep.round(2).tolist(),
                     reflectance=round(refl, 3))


def _sky(rng):
    style = "gradient" if rng.random() < 0.7 else "flat"
    rgb1, rgb2 = rng.uniform(0.05, 0.95, 3), rng.uniform(0.0, 0.5, 3)
    haze = rng.uniform(0.05, 0.6, 3)
    xml = (f'<texture type="skybox" builtin="{style}" rgb1="{_f(rgb1)}" rgb2="{_f(rgb2)}" '
           f'width="512" height="3072"/>')
    return xml, haze, dict(style=style, rgb1=rgb1.round(3).tolist(), rgb2=rgb2.round(3).tolist(),
                           haze=haze.round(3).tolist())


def _sample_lights(rng):
    """One to three lights, directional or point, around and above the table.

    The total diffuse power is drawn first and split between them, and the
    headlight keeps a floor of light on whatever faces the camera, so a draw
    can change where the shadows fall without ever leaving the AprilTags too
    dark or too flat to decode (`calib/calibrate.py --sim` is the check)."""
    n = int(rng.integers(*LIGHTS["n"]))
    total = float(rng.uniform(*LIGHTS["total"]))
    xml, desc = [], []
    for i in range(n):
        pos = np.array([WORKSPACE[0] + rng.uniform(-LIGHTS["spread"], LIGHTS["spread"]),
                        WORKSPACE[1] + rng.uniform(-LIGHTS["spread"], LIGHTS["spread"]),
                        TABLE_TOP + rng.uniform(*LIGHTS["height"])])
        d = WORKSPACE - pos
        d /= np.linalg.norm(d)
        diffuse = total / n * rng.uniform(0.8, 1.2) * rng.uniform(0.85, 1.0, 3)
        spec = rng.uniform(0.0, 0.35) * np.ones(3)
        directional = bool(rng.random() < 0.5)
        shadow = bool(rng.random() < 0.7)
        xml.append(f'<light name="light{i}" pos="{_f(pos)}" dir="{_f(d)}" '
                   f'directional="{"true" if directional else "false"}" '
                   f'castshadow="{"true" if shadow else "false"}" '
                   f'diffuse="{_f(diffuse)}" specular="{_f(spec)}"/>')
        desc.append(dict(pos=pos.round(3).tolist(), directional=directional, castshadow=shadow,
                         diffuse=diffuse.round(3).tolist()))
    head = dict(diffuse=float(rng.uniform(*LIGHTS["headlight_diffuse"])),
                ambient=float(rng.uniform(*LIGHTS["headlight_ambient"])),
                specular=float(rng.uniform(*LIGHTS["headlight_specular"])))
    return "".join(xml), head, dict(n=n, total=round(total, 3), headlight=head, lights=desc)


def _seg_dist(p, a, b):
    """Distance from point `p` to the segment `a`-`b`, in the xy plane."""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _sample_distractors(rng, table, bp, gp):
    """Zero to two plain primitives standing on the table.

    They are free bodies with no role: perception never looks at them and the
    planner never touches them. They stay `DISTRACT['clear']` away from the
    whole bottle-to-glass corridor (which covers both objects) and clear of the
    arm mount, so they add pixels without taking work away from the planner."""
    n = int(rng.integers(*DISTRACT["n"]))
    out = []
    for i in range(n):
        for _ in range(200):
            kind = ("box", "cylinder", "sphere")[rng.integers(3)]
            r = float(rng.uniform(*DISTRACT["r"]))
            h = float(rng.uniform(*DISTRACT["h"]))
            if kind == "box":
                ry = float(rng.uniform(*DISTRACT["r"]))
                size, half_z, foot = f"{r:.4f} {ry:.4f} {h / 2:.4f}", h / 2, float(np.hypot(r, ry))
            elif kind == "cylinder":
                size, half_z, foot = f"{r:.4f} {h / 2:.4f}", h / 2, r
            else:
                size, half_z, foot = f"{r:.4f}", r, r
            xy = np.array([rng.uniform(table["x"][0] + DISTRACT["edge"] + foot,
                                       table["x"][1] - DISTRACT["edge"] - foot),
                           rng.uniform(table["y"][0] + DISTRACT["edge"] + foot,
                                       table["y"][1] - DISTRACT["edge"] - foot)])
            if _seg_dist(xy, bp, gp) < DISTRACT["clear"] + foot:
                continue
            if np.linalg.norm(xy - np.asarray(ARM_BASE[:2])) < DISTRACT["arm_clear"] + foot:
                continue
            if np.linalg.norm(xy - WORKSPACE[:2]) > DISTRACT["view"]:
                continue          # keep it in front of the webcam, not off in a corner
            if any(np.linalg.norm(xy - o["pos"][:2]) < foot + o["foot"] + 0.02 for o in out):
                continue
            out.append(dict(name=f"distract{i}", kind=kind, size=size, foot=foot,
                            pos=np.array([xy[0], xy[1], TABLE_TOP + half_z + 0.001]),
                            yaw=float(rng.uniform(-np.pi, np.pi)),
                            mass=float(rng.uniform(*DISTRACT["mass"])),
                            rgba=np.concatenate([rng.uniform(0.05, 0.95, 3), [1.0]])))
            break
    return out


def _distractor_xml(o):
    q = (np.cos(o["yaw"] / 2), 0.0, 0.0, np.sin(o["yaw"] / 2))
    contact = " ".join(f'{k}="{v}"' for k, v in CONTACT.items())
    return (f'<body name="{o["name"]}" pos="{_f(o["pos"])}" quat="{_f(q)}">'
            f'<freejoint name="{o["name"]}"/>'
            f'<geom name="{o["name"]}" type="{o["kind"]}" size="{o["size"]}" '
            f'mass="{o["mass"]:.4f}" rgba="{_f(o["rgba"])}" {contact}/></body>')


def _scene_xml(table, sky_xml, haze, floor_xml, table_mat_xml, lights_xml, head):
    """The pour scene's own base MJCF. `pickplace_scene.BASE_XML` is the fixed
    version of this and stays as it is; everything here is per seed."""
    return f"""
<mujoco model="a1x_pour">
  <option timestep="0.002" integrator="implicitfast" noslip_iterations="5"/>
  <visual>
    <headlight diffuse="{head['diffuse']:.3f} {head['diffuse']:.3f} {head['diffuse']:.3f}"
               ambient="{head['ambient']:.3f} {head['ambient']:.3f} {head['ambient']:.3f}"
               specular="{head['specular']:.3f} {head['specular']:.3f} {head['specular']:.3f}"/>
    <rgba haze="{_f(haze)} 1"/>
    <global azimuth="140" elevation="-20" offwidth="1600" offheight="1000"/>
  </visual>
  <asset>
    {sky_xml}
    {floor_xml}
    {table_mat_xml}
  </asset>
  <worldbody>
    {lights_xml}
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <geom name="table" type="box" pos="{_f(table['pos'])}" size="{_f(table['half'])}" material="table"/>
    <body name="arm_mount" pos="{ARM_BASE[0]} {ARM_BASE[1]} {ARM_BASE[2]}"/>
  </worldbody>
</mujoco>
"""


def sample_scene(seed):
    """Everything about a seed's look and furniture, drawn once per build."""
    rng = np.random.default_rng([int(seed), 0xB0771E])
    table = _sample_table(rng)
    sky_xml, haze, sky_d = _sky(rng)
    floor_xml, floor_d = _surface(rng, "floor")
    table_mat_xml, table_d = _surface(rng, "table")
    lights_xml, head, lights_d = _sample_lights(rng)
    base = _scene_xml(table, sky_xml, haze, floor_xml, table_mat_xml, lights_xml, head)
    desc = dict(table=dict(pos=table["pos"].round(4).tolist(), half=table["half"].round(4).tolist(),
                           x=[round(v, 4) for v in table["x"]], y=[round(v, 4) for v in table["y"]]),
                sky=sky_d, floor=floor_d, table_material=table_d, lighting=lights_d)
    return rng, table, base, desc


def cam_intrinsics(W, H, fovy):
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])


def _sample_cam(rng):
    dist = rng.uniform(*CAM["dist"])
    bearing = rng.uniform(*CAM["bearing"])
    h = rng.uniform(*CAM["height"])
    eye = np.array([WORKSPACE[0] + dist * np.sin(bearing), WORKSPACE[1] + dist * np.cos(bearing),
                    TABLE_TOP + h])
    target = WORKSPACE + rng.uniform(-CAM["jitter"], CAM["jitter"], 3)
    roll = rng.uniform(-0.05, 0.05)
    up = np.array([np.sin(roll) * np.cos(bearing), -np.sin(roll) * np.sin(bearing), np.cos(roll)])
    return dict(pos=eye, target=target, up=up, fovy=float(rng.uniform(*CAM["fovy"])),
                W=CAM["W"], H=CAM["H"])


def _cam_pose_world(model, data, cam_name):
    """(position, rotation) of a camera in OpenCV convention: z forward, y down."""
    c = model.camera(cam_name).id
    R_mj = data.cam_xmat[c].reshape(3, 3)
    return data.cam_xpos[c].copy(), R_mj @ np.diag([1.0, -1.0, -1.0])


def _reachable(arm, targets):
    seeds = [np.clip(np.concatenate([[a], elbow, [0.0, 0.0]]), arm.lo, arm.hi)
             for elbow in ((1.0, -1.6, 0.6), (1.9, -2.4, 0.5), (0.7, -1.0, 0.3))
             for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    return all(any(arm.ik(s, np.asarray(p, float), R)[1] for s in seeds) for p, R in targets)


def grasp_R(approach_xy, pitch=GRASP_PITCH):
    """TCP rotation for a grasp across an upright bottle: x approaches along
    `approach_xy` pitched `pitch` rad down, y closes horizontally, z tilts up
    and forward. A level approach is out of the wrist's reach at bottle
    heights; 40 to 60 degrees down reaches the whole table."""
    a = np.array([approach_xy[0], approach_xy[1]]); a /= np.linalg.norm(a)
    approach = np.array([a[0] * np.cos(pitch), a[1] * np.cos(pitch), -np.sin(pitch)])
    return rot(approach, [-a[1], a[0], 0.0])


def tilt_about(axis_xy, angle):
    """Rotation of `angle` about a horizontal axis with direction `axis_xy`."""
    ax = np.array([axis_xy[0], axis_xy[1], 0.0]); ax /= np.linalg.norm(ax)
    R = np.zeros(9); q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, ax, angle)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def pour_orientations(R_g, tilt=POUR_TILT, n=8):
    """Candidate TCP rotations for pouring: the grasp rotation tipped `tilt`
    rad about horizontal axes at `n` azimuths."""
    return [tilt_about((np.cos(a), np.sin(a)), tilt) @ R_g for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def build(seed=0, max_tries=60, calib_card=False, card_extreme=False, wrist=None):
    """Compile one randomised episode. Returns (model, data, info).

    `calib_card` pinches the calibration card in the gripper; with it the scene
    also carries `info["calib"]`. Without it the scene has no fiducial of any
    kind, which is what the planner, the dataset recorder and the policy
    evaluator build. `card_extreme` puts the card at the limits of the
    misplacement instead of anywhere inside them. `wrist` ("left", "right" or
    "") mounts the G1 wrist camera; None takes it from WRIST_CAMERA. The camera
    is `WRIST_CAM`, its pose is jittered per seed by the bracket's re-seating
    play, and `info["wrist"]` carries that seed's exposure draw. None of it
    touches the other draws of a seed."""
    wrist = WRIST_HAND if wrist is None else wrist
    rng = np.random.default_rng(seed)
    drng, table, base_xml, rand = sample_scene(seed)
    stats = {"grasp": 0, "pour": 0}
    for attempt in range(max_tries):
        b, g = sample_bottle(rng, arng=drng), sample_glass(rng)
        for _ in range(100):
            bp = np.array([rng.uniform(*REGION["x"]), rng.uniform(*REGION["y"])])
            gp = np.array([rng.uniform(*REGION["x"]), rng.uniform(*REGION["y"])])
            if np.linalg.norm(bp - gp) > MIN_APART:
                break
        else:
            continue
        b_yaw = rng.uniform(-np.pi, np.pi)
        cam = _sample_cam(rng)

        distractors = _sample_distractors(drng, table, bp, gp)
        extra = [bottle_xml(b, (bp[0], bp[1], TABLE_TOP + 0.001), b_yaw),
                 glass_xml(g, (gp[0], gp[1], TABLE_TOP + 0.001)),
                 *[_distractor_xml(o) for o in distractors],
                 f'<camera name="laptop_cam" mode="fixed" fovy="{cam["fovy"]:.3f}" '
                 f'pos="{cam["pos"][0]:.4f} {cam["pos"][1]:.4f} {cam["pos"][2]:.4f}" '
                 f'xyaxes="{look_at(cam["pos"], cam["target"], cam["up"])}"/>',
                 '<camera name="table_cam" mode="fixed" pos="0.95 -1.75 1.45" '
                 f'xyaxes="{look_at((0.95, -1.75, 1.45), (0.03, -0.1, 0.72))}"/>']
        spec = mujoco.MjSpec.from_string(
            base_xml.replace("</worldbody>", "".join(extra) + "</worldbody>"))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = max(1600, cam["W"]), max(1000, cam["H"])
        add_bottle_meshes(spec, b)
        spec.attach(_arm_spec(wrist).copy(), prefix="arm/", frame=spec.body("arm_mount").add_frame())
        _split_finger_pads(spec)
        T_card = None
        if calib_card:
            T_card = sample_card_pose(np.random.default_rng([int(seed), 0xCA1D]), card_extreme)
            _add_card(spec, spec.body("arm/gripper_link"), T_card)
        model = spec.compile()
        wrist_info = None
        if wrist:
            # on the compiled model, because the arm spec is shared by every seed
            wrng = np.random.default_rng([int(seed), 0xCA11])
            _wrist_module().jitter_camera(model, wrng, camera=WRIST_CAM, **WRIST_JITTER)
            wrist_info = dict(hand=wrist, camera=WRIST_CAM,
                              gain=float(wrng.uniform(*WRIST_EXPOSURE["gain"])),
                              gamma=float(wrng.uniform(*WRIST_EXPOSURE["gamma"])))
        # The URDF gives no gripper force, and a position servo commanded to
        # "closed" squeezes with kp times the travel left: 13 N on a 30 mm neck,
        # and a full bottle pivots out. A real gripper closes with a set force
        # whatever the opening, so the closed command sits 30 mm past the stop:
        # 50 N on a neck, 90 N on a body, capped by the actuator's forcerange.
        grip = model.actuator("arm/gripper_finger1").id
        model.actuator_gainprm[grip, 0] = GRIP_KP
        model.actuator_biasprm[grip, 1] = -GRIP_KP
        model.actuator_ctrlrange[grip] = [GRIP_CMD, 0.05]

        data = mujoco.MjData(model)
        arm = Arm(model, "arm/")
        data.qpos[arm.qadr] = Q_HOME
        data.ctrl[arm.acts] = Q_HOME
        data.ctrl[arm.grip] = 0.05
        mujoco.mj_forward(model, data)

        # Must be able to side-grasp the body from the base side and hold the
        # bottle tilted over the glass; otherwise draw again.
        base = np.array(ARM_BASE)
        R_g = grasp_R(bp - base[:2])
        grasp = np.array([bp[0], bp[1], TABLE_TOP + 0.5 * b["body_h"]])
        if not _reachable(arm, [(grasp - 0.10 * R_g[:, 0], R_g), (grasp, R_g)]):
            stats["grasp"] += 1
            continue
        # the mouth sits L above the TCP along gripper z; tilted about some
        # horizontal axis it must end up over the glass with the TCP reachable
        L = b["height"] - 0.5 * b["body_h"]
        mouth = np.array([gp[0], gp[1], TABLE_TOP + g["h"] + POUR_ABOVE])
        if not any(_reachable(arm, [(mouth - L * R_p[:, 2], R_p)]) for R_p in pour_orientations(R_g)):
            stats["pour"] += 1
            continue

        cam_pos, cam_R = _cam_pose_world(model, data, "laptop_cam")
        T_cam2world = np.eye(4); T_cam2world[:3, :3], T_cam2world[:3, 3] = cam_R, cam_pos
        T_base2world = np.eye(4); T_base2world[:3, 3] = base
        cam.update(K=cam_intrinsics(cam["W"], cam["H"], cam["fovy"]),
                   T_cam2world=T_cam2world, T_cam2base=np.linalg.inv(T_base2world) @ T_cam2world)
        info = dict(seed=int(seed), attempt=attempt, rejected=stats, bottle=b, glass=g,
                    bottle_pos=np.array([bp[0], bp[1], TABLE_TOP]), bottle_yaw=float(b_yaw),
                    glass_pos=np.array([gp[0], gp[1], TABLE_TOP]),
                    cam=cam, wrist=wrist_info,
                    arm_base=base, table_top=TABLE_TOP, q_home=Q_HOME.copy(),
                    randomisation=dict(
                        rand,
                        bottle=dict(opaque=bool(b["opaque"]), alpha=float(b["rgba"][3]),
                                    n_labels=len(b["labels"])),
                        distractors=[dict(name=o["name"], kind=o["kind"], size=o["size"],
                                          pos=o["pos"].round(4).tolist(),
                                          rgba=o["rgba"].round(3).tolist()) for o in distractors]))
        if calib_card:
            # Everything the solver may read is in `nominal`, `tag_size` and
            # `tags`; `T_card2gripper` is ground truth, for scoring only.
            info["calib"] = dict(frame="arm/gripper_link", nominal=card_nominal(),
                                 T_card2gripper=T_card, tag_size=CARD["tag"],
                                 tags=card_tag_poses())
        return model, data, info
    raise RuntimeError(f"could not build a valid scene for seed {seed} in {max_tries} tries: {stats}")


def render(model, data, cam="laptop_cam", W=None, H=None):
    """One RGB frame from a named camera at its native size."""
    W = W or CAM["W"]; H = H or CAM["H"]
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=cam)
        return r.render().copy()


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data, info = build(seed)
    b, g, c = info["bottle"], info["glass"], info["cam"]
    print(f"seed {seed}: built on attempt {info['attempt']}")
    print(f"  bottle body_r={b['body_r']:.3f} height={b['height']:.3f} neck_r={b['neck_r']:.3f} "
          f"mass={b['mass']:.2f} at {np.round(info['bottle_pos'][:2], 3)} yaw={info['bottle_yaw']:+.2f}")
    print(f"  glass  r={g['r']:.3f} h={g['h']:.3f} at {np.round(info['glass_pos'][:2], 3)}")
    print(f"  cam    pos={np.round(c['pos'], 3)} fovy={c['fovy']:.1f}")
    r = info["randomisation"]
    print(f"  table  half={r['table']['half']} centre={r['table']['pos'][:2]} "
          f"mat={r['table_material']['style']} floor={r['floor']['style']} sky={r['sky']['style']}")
    print(f"  light  n={r['lighting']['n']} total={r['lighting']['total']} "
          f"headlight={ {k: round(v, 3) for k, v in r['lighting']['headlight'].items()} }")
    print(f"  bottle labels={r['bottle']['n_labels']} alpha={r['bottle']['alpha']:.2f}  "
          f"distractors={[o['kind'] for o in r['distractors']]}")
    if len(sys.argv) > 2:
        from PIL import Image
        Image.fromarray(render(model, data)).save(sys.argv[2])
        print("wrote", sys.argv[2])
