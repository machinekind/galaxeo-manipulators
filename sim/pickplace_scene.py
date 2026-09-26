#!/usr/bin/env python3
"""Randomised pick-and-place scene: one A1X on a table, 1-3 primitives to pick,
and a placeholder quadruped ("dog") standing beside the table to place them on.

    sim/.venv/bin/python sim/pickplace_scene.py            # build seed 0, print the layout
    sim/.venv/bin/python sim/pickplace_scene.py 3 out.png  # build seed 3, render table_cam

`build(seed)` returns `(model, data, info)` and is deterministic for a seed.
The scene is assembled with `mujoco.MjSpec` so every episode recompiles with
fresh object sizes; the arm itself is attached from the generated `a1x.xml`.

The dog is a mocap body: its pose is written per episode into `data.mocap_pos`
/ `mocap_quat`, and `apply_sway()` adds a few millimetres of 0.5 Hz motion so
the planner cannot trust a pose it read ten seconds ago.

Frame conventions: table top at z = 0.7, the arm sits on the table near its -y
edge, objects spawn on the +y side of the arm, the dog stands on the floor on
the -y side. Placing therefore means reaching down past the table edge.
"""
import os
import sys

import mujoco
import numpy as np

from a1x_control import Arm, rot

HERE = os.path.dirname(os.path.abspath(__file__))

TABLE = dict(pos=(0.0, 0.0, 0.35), half=(0.45, 0.26, 0.35))     # top at z = 0.7
TABLE_TOP = TABLE["pos"][2] + TABLE["half"][2]
ARM_BASE = (0.0, -0.20, TABLE_TOP)                              # mount on the table, -y edge
OBJ_REGION = dict(x=(-0.20, 0.20), y=(0.02, 0.20))              # reachable patch of table top
TORSO_HALF = (0.30, 0.125, 0.075)                               # dog back platform, 0.6 x 0.25 x 0.15
DOG_ARC = dict(r=(0.33, 0.42), yaw_off=(-0.5, 0.5))             # dog centre, polar about the arm base
BACK_TOP = (0.40, 0.50)                                         # randomised platform height [m]
Q_HOME = np.array([0.0, 1.0, -1.6, 0.6, 0.0, 0.0])
SHAPES = ("box", "cylinder", "sphere", "capsule")
_ARM = None                                                     # a1x.xml, parsed once

# Same stiff contact recipe as the handover ball -- light objects otherwise sink
# millimetres into the finger plates and creep out of the grasp under gravity --
# but with condim 6 and real rolling friction. A sphere or a toppled capsule on
# a perfectly flat platform that sways a few millimetres has nothing to stop it
# with condim 4, so it rolls off the dog's back every time.
CONTACT = dict(friction="1.5 0.02 0.005", condim="6", priority="1",
               solref="0.005 1", solimp="0.98 0.995 0.001")

BASE_XML = f"""
<mujoco model="a1x_pickplace">
  <option timestep="0.002" integrator="implicitfast" noslip_iterations="5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20" offwidth="1600" offheight="1000"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table" rgba="0.55 0.45 0.35 1"/>
    <material name="dog" rgba="0.25 0.27 0.30 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <light pos="1 -1 2" dir="-0.4 0.4 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <geom name="table" type="box" pos="{TABLE['pos'][0]} {TABLE['pos'][1]} {TABLE['pos'][2]}"
          size="{TABLE['half'][0]} {TABLE['half'][1]} {TABLE['half'][2]}" material="table"/>
    <body name="arm_mount" pos="{ARM_BASE[0]} {ARM_BASE[1]} {ARM_BASE[2]}"/>
  </worldbody>
</mujoco>
"""


def _arm_spec():
    global _ARM
    if _ARM is None:
        _ARM = mujoco.MjSpec.from_file(os.path.join(HERE, "a1x.xml"))
    return _ARM


def look_at(eye, target, up=(0, 0, 1)):
    """`xyaxes` string for a fixed camera at `eye` pointing at `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join(f"{v:.5f}" for v in np.concatenate([x, y]))


def yaw_quat(psi):
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


def _rect_corners(centre, half_xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    return np.asarray(centre, float) + (signs * half_xy) @ R.T


def _rects_overlap(a_c, a_half, a_yaw, b_c, b_half, b_yaw, margin=0.0):
    """Separating-axis test for two rotated rectangles in the xy plane."""
    A, B = _rect_corners(a_c, np.array(a_half) + margin, a_yaw), _rect_corners(b_c, b_half, b_yaw)
    for yaw in (a_yaw, b_yaw):
        for ax in (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])):
            pa, pb = A @ ax, B @ ax
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _sample_object(rng, name):
    """One graspable primitive: at least one horizontal extent <= 0.07 m."""
    kind = SHAPES[rng.integers(len(SHAPES))]
    height = rng.uniform(0.03, 0.10)
    if kind == "box":
        # narrow axis is local y, so the gripper closes across it
        hy = rng.uniform(0.012, 0.034)
        hx = rng.uniform(hy, 0.055)
        half = (hx, hy, height / 2)
        size = f"{hx} {hy} {height / 2}"
    elif kind == "cylinder":
        r = min(rng.uniform(0.016, 0.035), height / 2 + 0.03)
        half = (r, r, height / 2)
        size = f"{r} {height / 2}"
    elif kind == "sphere":
        r = np.clip(height / 2, 0.016, 0.034)
        height = 2 * r
        half = (r, r, r)
        size = f"{r}"
    else:                                   # capsule, standing upright
        r = rng.uniform(0.014, 0.030)
        h = max(0.005, height / 2 - r)
        height = 2 * (h + r)
        half = (r, r, h + r)
        size = f"{r} {h}"
    return dict(name=name, kind=kind, size=size, half=np.array(half, float), height=height,
                mass=float(rng.uniform(0.03, 0.2)),
                rgba=np.concatenate([rng.uniform(0.2, 0.95, 3), [1.0]]))


def _place_objects(rng, objs):
    """Random xy inside the reachable patch, random yaw, no mutual overlap."""
    placed = []
    for o in objs:
        for _ in range(200):
            x = rng.uniform(*OBJ_REGION["x"]); y = rng.uniform(*OBJ_REGION["y"])
            yaw = rng.uniform(-np.pi, np.pi)
            r = float(np.hypot(o["half"][0], o["half"][1]))
            if all(np.hypot(x - p["pos"][0], y - p["pos"][1]) > r + p["clear"] + 0.045 for p in placed):
                o["pos"] = np.array([x, y, TABLE_TOP + o["half"][2] + 0.001])
                o["yaw"] = yaw
                o["clear"] = r
                placed.append(o)
                break
        else:
            return False
    return True


def _sample_dog(rng):
    r = rng.uniform(*DOG_ARC["r"])
    a = rng.uniform(*DOG_ARC["yaw_off"])                   # angle off the -y axis, from the arm base
    centre = np.array([ARM_BASE[0] + r * np.sin(a), ARM_BASE[1] - r * np.cos(a)])
    yaw = rng.uniform(-0.55, 0.55)                          # torso roughly parallel to the table edge
    back_top = rng.uniform(*BACK_TOP)
    return centre, yaw, back_top


def _dog_xml(back_top):
    """Mocap torso plus four legs; the torso's top face is the back platform."""
    hx, hy, hz = TORSO_HALF
    leg_h = max(0.02, (back_top - 2 * hz) / 2)
    legs = "".join(
        f'<geom name="dog/leg{i}" type="cylinder" material="dog" size="0.025 {leg_h:.4f}" '
        f'pos="{sx * 0.24} {sy * 0.10} {-hz - leg_h:.4f}"/>'
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]))
    return (f'<body name="dog" mocap="true" pos="0 0 {back_top - hz:.4f}">'
            f'<geom name="dog/torso" type="box" material="dog" size="{hx} {hy} {hz}" friction="1.0 0.02 0.002"/>'
            f'{legs}'
            f'<site name="dog/tag" pos="0 0 {hz}" size="0.012" rgba="1 1 0 0.9" group="4"/>'
            f'</body>')


def _reachable(model, data, targets):
    """True if IK converges for every (pos, R) target.

    Uses the same seed fan as `planner.motion.Motion`: single-start DLS misses
    most of this arm's workspace, and a build-time check stricter than the
    planner would throw away perfectly good dog placements."""
    arm = Arm(model, "arm/")
    seeds = [np.clip(np.concatenate([[a], elbow, [0.0, 0.0]]), arm.lo, arm.hi)
             for elbow in ((1.0, -1.6, 0.6), (1.9, -2.4, 0.5), (0.7, -1.0, 0.3))
             for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    for pos, R in targets:
        if not any(arm.ik(s, np.asarray(pos, float), R)[1] for s in seeds):
            return False
    return True


def build(seed=0, sway=True, max_tries=60, arm_hook=None):
    """Compile one randomised episode. Returns (model, data, info).

    `arm_hook(spec)` is called on the fresh arm spec before it is attached,
    which is where `wrist_camera.attach_wrist_camera` has to run."""
    rng = np.random.default_rng(seed)
    for attempt in range(max_tries):
        n_obj = int(rng.integers(1, 4))
        objs = [_sample_object(rng, f"obj{i}") for i in range(n_obj)]
        if not _place_objects(rng, objs):
            continue
        dog_c, dog_yaw, back_top = _sample_dog(rng)
        # the dog must stand clear of the table, at any yaw
        if _rects_overlap(TABLE["pos"][:2], TABLE["half"][:2], 0.0,
                          dog_c, (TORSO_HALF[0] + 0.03, TORSO_HALF[1] + 0.03), dog_yaw):
            continue

        extra = [_dog_xml(back_top)]
        for o in objs:
            q = yaw_quat(o["yaw"])
            extra.append(
                f'<body name="{o["name"]}" pos="{o["pos"][0]:.5f} {o["pos"][1]:.5f} {o["pos"][2]:.5f}" '
                f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}">'
                f'<freejoint name="{o["name"]}"/>'
                f'<geom name="{o["name"]}" type="{o["kind"]}" size="{o["size"]}" mass="{o["mass"]:.4f}" '
                f'rgba="{" ".join(f"{v:.3f}" for v in o["rgba"])}" '
                + " ".join(f'{k}="{v}"' for k, v in CONTACT.items()) + '/></body>')
        eye = (0.95, -1.75, 1.45)
        extra.append(f'<camera name="table_cam" mode="fixed" pos="{eye[0]} {eye[1]} {eye[2]}" '
                     f'xyaxes="{look_at(eye, (0.03, -0.36, 0.62))}"/>')
        spec = mujoco.MjSpec.from_string(
            BASE_XML.replace("</worldbody>", "".join(extra) + "</worldbody>"))
        # a fresh copy per attach: MjSpec.attach takes ownership of the child,
        # and reusing one across compiles segfaults
        arm_spec = _arm_spec().copy()
        if arm_hook is not None:
            arm_hook(arm_spec)
        spec.attach(arm_spec, prefix="arm/", frame=spec.body("arm_mount").add_frame())
        model = spec.compile()

        data = mujoco.MjData(model)
        arm = Arm(model, "arm/")
        data.qpos[arm.qadr] = Q_HOME
        data.ctrl[arm.acts] = Q_HOME
        data.ctrl[arm.grip] = 0.05
        data.mocap_pos[0] = [dog_c[0], dog_c[1], back_top - TORSO_HALF[2]]
        data.mocap_quat[0] = yaw_quat(dog_yaw)
        mujoco.mj_forward(model, data)

        # Placing means reaching down past the table edge; check IK before committing.
        tag = data.site_xpos[model.site("dog/tag").id].copy()
        close = np.array([np.cos(dog_yaw + np.pi / 2), np.sin(dog_yaw + np.pi / 2), 0.0])
        R = rot([0, 0, -1], close)
        # The planner lands with the TCP ~35 mm above the tag (the object hangs
        # below the plates), so check that height, not a comfortable one.
        if not _reachable(model, data, [(tag + [0, 0, 0.20], R), (tag + [0, 0, 0.035], R)]):
            continue

        info = dict(seed=int(seed), attempt=attempt, objects=objs, n_obj=n_obj,
                    dog=dict(centre=dog_c, yaw=float(dog_yaw), back_top=float(back_top),
                             platform_half=np.array(TORSO_HALF), tag=tag,
                             nominal_pos=data.mocap_pos[0].copy(),
                             nominal_quat=data.mocap_quat[0].copy()),
                    arm_base=np.array(ARM_BASE), table_top=TABLE_TOP,
                    q_home=Q_HOME.copy(), sway=bool(sway))
        return model, data, info
    raise RuntimeError(f"could not build a valid scene for seed {seed} in {max_tries} tries")


def apply_sway(data, info, amp=0.004, freq=0.5):
    """A live dog is never still: a few mm at ~0.5 Hz, plus half a degree of yaw."""
    if not info["sway"]:
        return
    t = data.time
    d = info["dog"]
    data.mocap_pos[0] = d["nominal_pos"] + [amp * np.sin(2 * np.pi * freq * t),
                                            amp * np.sin(2 * np.pi * freq * t + 1.1),
                                            0.5 * amp * np.sin(2 * np.pi * freq * 1.7 * t)]
    data.mocap_quat[0] = yaw_quat(d["yaw"] + 0.009 * np.sin(2 * np.pi * freq * 0.8 * t))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model, data, info = build(seed)
    print(f"seed {seed}: {info['n_obj']} object(s), built on attempt {info['attempt']}")
    for o in info["objects"]:
        print(f"  {o['name']:5s} {o['kind']:9s} half={np.round(o['half'], 3)} "
              f"pos={np.round(o['pos'], 3)} yaw={o['yaw']:+.2f} mass={o['mass']:.3f}")
    d = info["dog"]
    print(f"  dog    centre={np.round(d['centre'], 3)} yaw={d['yaw']:+.2f} "
          f"back_top={d['back_top']:.3f} tag={np.round(d['tag'], 3)}")
    arm = Arm(model, "arm/")
    print(f"  tcp at home = {np.round(data.site_xpos[arm.site], 3)}")
    if len(sys.argv) > 2:
        with mujoco.Renderer(model, height=600, width=900) as r:
            r.update_scene(data, camera="table_cam")
            from PIL import Image
            Image.fromarray(r.render()).save(sys.argv[2])
        print("wrote", sys.argv[2])
