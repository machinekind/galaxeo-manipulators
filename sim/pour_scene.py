#!/usr/bin/env python3
"""Randomised pour scene: one A1X on the table, a bottle, a small glass, and
a laptop webcam at a random, unknown pose.

    sim/.venv/bin/python sim/pour_scene.py            # build seed 0, print the layout
    sim/.venv/bin/python sim/pour_scene.py 3 out.png  # build seed 3, render the laptop cam

`build(seed)` returns `(model, data, info)` and is deterministic for a seed.
Table, arm mount and contact recipe are shared with `pickplace_scene.py`; the
bottle and glass come from `bottle.py`.

The camera stands in for a laptop put down somewhere in front of the arm: its
pose is drawn from `CAM` (distance, bearing and height about the workspace)
and is *not* given to the planner. `info["cam"]` keeps the ground truth so the
calibration in `calib/` can be scored against it.

AprilTags (36h11) are rendered on four faces of the gripper body and on a
plate beside the arm base, with a site at each tag centre in the tag frame
(x right, y up, z out of the face -- OpenCV's marker convention), so the same
detector and hand-eye solver run in sim and on the real arm.
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from a1x_control import Arm, rot                                             # noqa: E402
from bottle import add_bottle_meshes, bottle_xml, glass_xml, sample_bottle, sample_glass  # noqa: E402
from pickplace_scene import ARM_BASE, BASE_XML, TABLE, TABLE_TOP, Q_HOME, look_at  # noqa: E402

REGION = dict(x=(-0.18, 0.18), y=(0.03, 0.20))      # where bottle and glass may stand
MIN_APART = 0.11                                    # bottle to glass, centre to centre [m]
CAM = dict(W=640, H=480, fovy=(44.0, 54.0),         # laptop webcam, 4:3
           dist=(0.50, 0.85), bearing=(-1.6, 1.6),  # about the workspace centre, rad off +y
           height=(0.12, 0.38), jitter=0.04)        # lens height above the table, look-at jitter
WORKSPACE = np.array([0.0, 0.10, TABLE_TOP + 0.10])
GRASP_PITCH = 0.9                                   # approach pitch below level [rad]
POUR_TILT, POUR_ABOVE = 1.95, 0.03                 # rad past upright; mouth height over the rim [m]
TAG_PLATE = 0.045                                   # printed plate side [m]; marker is 160/200 of it
TAG_SIZE = TAG_PLATE * 160 / 200
TAGS_DIR = os.path.join(HERE, "assets", "tags")

# A 40 mm printed cube sits on top of the gripper body (x toward the fingers,
# y the closing axis, z up at home) with a tag on its top and three sides.
# tag id -> (position, rotation) of the tag frame in the gripper_link frame.
CUBE_C, CUBE_H = np.array([0.0, 0.011, 0.0556 + 0.020]), 0.020
_GRIPPER_TAGS = {
    0: (CUBE_C + [0, 0, CUBE_H + 0.0005], rot([1, 0, 0], [0, 1, 0])),          # top (+z)
    1: (CUBE_C + [-CUBE_H - 0.0005, 0, 0], rot([0, -1, 0], [0, 0, 1])),        # back (-x)
    2: (CUBE_C + [0, CUBE_H + 0.0005, 0], rot([1, 0, 0], [0, 0, -1])),         # +y side
    3: (CUBE_C + [0, -CUBE_H - 0.0005, 0], rot([1, 0, 0], [0, 0, 1])),         # -y side
}
BASE_TAG_ID, BASE_TAG_POS = 4, np.array([-0.13, 0.0, 0.035])                 # upright plate by the mount, facing +y
BASE_TAG_R = rot([-1, 0, 0], [0, 0, 1])
GRIP_KP, GRIP_CMD = 1500.0, -0.03   # finger servo gain and the 'closed' command, see build()
_ARM = None


def _arm_spec():
    global _ARM
    if _ARM is None:
        _ARM = mujoco.MjSpec.from_file(os.path.join(HERE, "a1x.xml"))
    return _ARM


def _add_tag_assets(spec, ids):
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
    h = TAG_PLATE / 2
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


def build(seed=0, max_tries=60):
    """Compile one randomised episode. Returns (model, data, info)."""
    rng = np.random.default_rng(seed)
    stats = {"grasp": 0, "pour": 0}
    for attempt in range(max_tries):
        b, g = sample_bottle(rng), sample_glass(rng)
        for _ in range(100):
            bp = np.array([rng.uniform(*REGION["x"]), rng.uniform(*REGION["y"])])
            gp = np.array([rng.uniform(*REGION["x"]), rng.uniform(*REGION["y"])])
            if np.linalg.norm(bp - gp) > MIN_APART:
                break
        else:
            continue
        b_yaw = rng.uniform(-np.pi, np.pi)
        cam = _sample_cam(rng)

        extra = [bottle_xml(b, (bp[0], bp[1], TABLE_TOP + 0.001), b_yaw),
                 glass_xml(g, (gp[0], gp[1], TABLE_TOP + 0.001)),
                 f'<camera name="laptop_cam" mode="fixed" fovy="{cam["fovy"]:.3f}" '
                 f'pos="{cam["pos"][0]:.4f} {cam["pos"][1]:.4f} {cam["pos"][2]:.4f}" '
                 f'xyaxes="{look_at(cam["pos"], cam["target"], cam["up"])}"/>',
                 '<camera name="table_cam" mode="fixed" pos="0.95 -1.75 1.45" '
                 f'xyaxes="{look_at((0.95, -1.75, 1.45), (0.03, -0.1, 0.72))}"/>']
        spec = mujoco.MjSpec.from_string(
            BASE_XML.replace("</worldbody>", "".join(extra) + "</worldbody>"))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = max(1600, cam["W"]), max(1000, cam["H"])
        add_bottle_meshes(spec, b)
        _add_tag_assets(spec, list(_GRIPPER_TAGS) + [BASE_TAG_ID])
        _add_tag(spec, spec.body("arm_mount"), BASE_TAG_ID, BASE_TAG_POS, BASE_TAG_R, "base/")
        spec.attach(_arm_spec().copy(), prefix="arm/", frame=spec.body("arm_mount").add_frame())
        _split_finger_pads(spec)
        cube = spec.body("arm/gripper_link").add_geom()
        cube.name, cube.type, cube.size = "arm/tag_cube", mujoco.mjtGeom.mjGEOM_BOX, [CUBE_H] * 3
        cube.pos, cube.rgba = CUBE_C, [0.95, 0.95, 0.95, 1]
        cube.contype = cube.conaffinity = 0
        cube.mass, cube.group = 0.0, 1
        for tid, (pos, R) in _GRIPPER_TAGS.items():
            _add_tag(spec, spec.body("arm/gripper_link"), tid, pos, R, "arm/")
        model = spec.compile()
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
        tags = {tid: dict(body="arm/gripper_link", pos=pos, R=R) for tid, (pos, R) in _GRIPPER_TAGS.items()}
        tags[BASE_TAG_ID] = dict(body="arm_mount", pos=BASE_TAG_POS, R=BASE_TAG_R)
        info = dict(seed=int(seed), attempt=attempt, rejected=stats, bottle=b, glass=g,
                    bottle_pos=np.array([bp[0], bp[1], TABLE_TOP]), bottle_yaw=float(b_yaw),
                    glass_pos=np.array([gp[0], gp[1], TABLE_TOP]),
                    cam=cam, tags=tags, tag_size=TAG_SIZE,
                    arm_base=base, table_top=TABLE_TOP, q_home=Q_HOME.copy())
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
    if len(sys.argv) > 2:
        from PIL import Image
        Image.fromarray(render(model, data)).save(sys.argv[2])
        print("wrote", sys.argv[2])
