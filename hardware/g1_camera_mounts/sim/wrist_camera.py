#!/usr/bin/env python3
"""Put the G1 wrist-camera mount on the simulated arm.

    import sys; sys.path.insert(0, "hardware/g1_camera_mounts/sim")
    from wrist_camera import attach_wrist_camera, render_wrist, wrist_intrinsics

    spec = mujoco.MjSpec.from_file("sim/a1x.xml")     # or pour_scene._arm_spec().copy()
    info = attach_wrist_camera(spec, hand="right")    # BEFORE spec.attach(...) into a scene
    ...
    frame = render_wrist(model, data)                 # 640x480 RGB from the lens

The mount is one rigid child body of ``gripper_link``:

  * ``wrist_camera_mount`` carries the whole payload's mass, centre of mass and
    inertia from ``collision/payload.json`` (103.7 g at (-33.8, 26.7, 19.4) mm),
    so the wrist is loaded the way the real one will be -- not a massless
    decoration;
  * one visual-only mesh geom (``sim/meshes/wrist_mount_<hand>.stl``, gripper
    frame, metres, class ``visual`` exactly as ``urdf2mjcf.py`` writes the arm's
    own visual geoms);
  * NOTE: the planner also needs a patch -- ``planner/motion.py`` treats an
    ``arm/``-prefixed payload as self-contact and skips every mount-vs-arm and
    mount-vs-held-object contact.  See ``sim/DATASET.md`` section 1b;
  * the 57 collision primitives from ``collision/payload.json`` as boxes and
    cylinder in class ``collision``, so ``planner/motion.py`` -- which walks
    ``data.contact`` and calls anything under the ``arm/`` prefix "mine" --
    sees the bracket and refuses a waypoint that would drive it into the table;
  * a ``<camera>`` at the lens's entrance pupil with the recommended lens's
    vertical field of view.

Deliberately NOT a convex hull: the free corridor between the clamp band and
the camera plate is where a held bottle goes, and hulling the payload would
close it.  That is the whole reason ``payload.json`` ships primitives.

Handedness: ``hand="right"`` puts the camera on +Y of ``gripper_link``,
``"left"`` mirrors everything in Y (pose, inertia products, primitives and
mesh).  See the README for which one to use -- for the pour task it is the one
whose payload swings *away* from the table during the wrist roll.

Dependencies: ``mujoco`` and ``numpy``.  ``make_meshes.py`` (cadquery) is a
build step that has already run; its output is committed.
"""
import json
import os
import struct

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CAMERA_SPEC = os.path.join(PKG, "camera_spec.json")
PAYLOAD = os.path.join(PKG, "collision", "payload.json")
MESH_DIR = os.path.join(HERE, "meshes")

BODY = "wrist_camera_mount"
CAMERA = "wrist"
RESOLUTION = (640, 480)
MIRROR = np.diag([1.0, -1.0, 1.0])


# --------------------------------------------------------------------------
# the design, as data
# --------------------------------------------------------------------------

def load_spec(camera_spec=CAMERA_SPEC, payload=PAYLOAD):
    """The two JSON files the CAD package exports, as one dict.

    Everything downstream reads this, so a re-export of the mount changes the
    simulator with no code edit."""
    with open(camera_spec) as f:
        cam = json.load(f)
    with open(payload) as f:
        pay = json.load(f)
    assert cam["frame"] == pay["frame"] == "gripper_link", "specs are not in the gripper frame"
    assert cam["units"] == "metres" and pay["units"]["pos"] == "m"
    return dict(camera=cam, payload=pay)


def lens_fov(spec, lens="recommended", resolution=RESOLUTION):
    """(horizontal, vertical) full angles [deg] of a named lens option.

    The two are not independent.  MuJoCo drives a fixed camera from ``fovy`` alone
    (``cam_sensorsize`` is [0, 0]), and a real module with square pixels is the same:
    the horizontal angle follows from the vertical one and the aspect.  So this
    asserts that the spec's pair is one this sensor can produce -- a nominal
    "110 x 85" on 640 x 480 is 8.6 deg wider than the renderer or the lens gives, and
    quoting it made every horizontal number in G2'/G3 optimistic by 4.3 deg a side.
    """
    entry = spec["camera"]["lens"][lens]
    fov = tuple(float(v) for v in entry["fov_deg"])
    aspect = resolution[0] / resolution[1]
    expect = 2 * np.degrees(np.arctan(aspect * np.tan(np.radians(fov[1] / 2))))
    assert abs(fov[0] - expect) < 1.0, (
        f"{lens} lens: {fov[0]:.1f} deg horizontal is not what {fov[1]:.1f} deg "
        f"vertical gives on a {resolution[0]}x{resolution[1]} frame ({expect:.1f})")
    return fov


def camera_matrix(forward, up):
    """MuJoCo camera rotation: +X right, +Y up, **-Z** along the optical axis.

    Built from the optical axis and the image-up vector so it cannot disagree
    with ``camera_spec.json``; ``image_right`` there is the same thing and is
    asserted against this."""
    z = -np.asarray(forward, float)
    z /= np.linalg.norm(z)
    y = np.asarray(up, float)
    y = y - z * (z @ y)
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def camera_pose(spec, hand="right"):
    """(position, MuJoCo rotation matrix) of the lens in the gripper frame."""
    cam = spec["camera"]
    pos = np.asarray(cam["lens_entrance_pupil_m"], float)
    fwd = np.asarray(cam["optical_axis"], float)
    up = np.asarray(cam["image_up"], float)
    if _side(hand) != _side(cam["handedness"]):
        pos, fwd, up = MIRROR @ pos, MIRROR @ fwd, MIRROR @ up
    return pos, camera_matrix(fwd, up)


def _side(hand):
    if isinstance(hand, str):
        hand = hand.lower()
        if hand not in ("right", "left"):
            raise ValueError(f"hand must be 'right' or 'left', not {hand!r}")
        return 1 if hand == "right" else -1
    return 1 if hand >= 0 else -1


def inertial(spec, hand="right"):
    """(mass [kg], CoM in the gripper frame [m], fullinertia about the CoM).

    ``payload.json`` gives the inertia about the *gripper origin*; MuJoCo's
    ``<inertial>`` wants it about the centre of mass, so shift it back with the
    parallel-axis theorem.  ``fullinertia`` order is (xx, yy, zz, xy, xz, yz)."""
    pay = spec["payload"]
    mass = float(pay["mass_kg"])
    com = np.asarray(pay["com_m"], float)
    about_origin = np.asarray(pay["inertia_about_gripper_origin_kg_m2"], float)
    if _side(hand) < 0:
        com = MIRROR @ com
        about_origin = MIRROR @ about_origin @ MIRROR
    inertia = about_origin - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
    inertia = 0.5 * (inertia + inertia.T)
    eig = np.linalg.eigvalsh(inertia)
    assert eig.min() > 0, f"payload inertia is not positive definite: {eig}"
    assert eig[0] + eig[1] > eig[2] * (1 - 1e-9), f"inertia breaks the triangle rule: {eig}"
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return mass, com, np.array(full)


def primitives(spec, hand="right"):
    """Collision primitives in the gripper frame, mirrored for the left hand.

    A box is symmetric about each of its own planes and a cylinder about its
    radial ones, so mirroring in Y and then negating the local X column gives
    back a proper right-handed rotation describing the same mirrored solid."""
    out = []
    for prim in spec["payload"]["primitives"]:
        pos = np.asarray(prim["pos"], float)
        rot = np.asarray(prim["rot"], float)
        if _side(hand) < 0:
            pos = MIRROR @ pos
            rot = MIRROR @ rot
            rot[:, 0] *= -1.0
        assert np.linalg.det(rot) > 0.9, f"{prim['name']}: rotation is not right-handed"
        out.append(dict(name=prim["name"], type=prim["type"], pos=pos, rot=rot,
                        size=np.asarray(prim["size"], float)))
    return out


# --------------------------------------------------------------------------
# STL -> uservert/userface
# --------------------------------------------------------------------------

def read_stl(path):
    """(vertices, faces) from a binary STL, in the file's own units.

    Parsed here rather than handed to MuJoCo as a file so the mesh travels with
    the spec: ``MjSpec.attach`` moves a scene's meshes around and a relative
    ``meshdir`` that resolves from one working directory need not resolve from
    another.  numpy only -- no trimesh at runtime."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:5].lstrip().lower().startswith(b"solid") and b"facet normal" in blob[:512]:
        raise ValueError(f"{path}: ASCII STL; re-run make_meshes.py, which writes binary")
    n = struct.unpack("<I", blob[80:84])[0]
    if len(blob) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL of {n} triangles")
    rec = np.frombuffer(blob, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("a", "<u2")]), count=n, offset=84)
    tri = rec["v"].reshape(-1, 3)
    verts, faces = np.unique(tri, axis=0, return_inverse=True)
    return verts.astype(np.float32), faces.astype(np.int32).reshape(-1, 3)


# --------------------------------------------------------------------------
# attachment
# --------------------------------------------------------------------------

def attach_wrist_camera(spec, hand="right", gripper_body="gripper_link",
                        camera_name=CAMERA, collision=True, visual=True,
                        body_name=BODY, lens="recommended", fovy=None,
                        design=None, jitter=None, rng=None):
    """Bolt the mount onto an arm ``MjSpec``, in place. Returns an info dict.

    Call this on the arm spec **before** it is attached into a scene -- both
    ``pickplace_scene._arm_spec()`` and ``pour_scene._arm_spec()`` hand out a
    spec that is then ``spec.attach(..., prefix="arm/")``-ed, and the prefix is
    applied to whatever is already there, so the body comes out as
    ``arm/wrist_camera_mount`` and the camera as ``arm/wrist``.

    ``lens`` picks ``"recommended"`` (the wide M12, 101.4 x 85 on this frame) or
    ``"alternative"`` (69.5 x 55) from ``camera_spec.json``; ``fovy`` overrides the
    vertical angle outright.  ``lens_fov`` asserts the pair is one the frame can
    actually produce.

    ``jitter`` is the domain-randomisation hook: a dict with any of
    ``pos_mm``, ``rot_deg`` and ``fovy_deg``, each a half-range, drawn
    uniformly with ``rng``.  It moves only the *camera*, not the collision
    geometry -- it stands for clamp and assembly tolerance, which a policy must
    not be able to lean on, while the bracket's shape is known exactly.
    """
    design = design or load_spec()
    side = _side(hand)
    if _has_body(spec, body_name):
        raise ValueError(f"{body_name} is already on this spec: attaching a second "
                         "mount would duplicate its mass. Start from a fresh spec, "
                         "or pass body_name= for a second camera.")
    body = spec.body(gripper_body).add_body()
    body.name = body_name
    body.pos = [0.0, 0.0, 0.0]
    body.quat = [1.0, 0.0, 0.0, 0.0]

    mass, com, full = inertial(design, side)
    body.mass = mass
    body.ipos = com
    body.fullinertia = full
    body.explicitinertial = True

    if visual:
        path = os.path.join(MESH_DIR, f"wrist_mount_{'right' if side > 0 else 'left'}.stl")
        verts, faces = read_stl(path)
        mesh = spec.add_mesh()
        mesh.name = f"{body_name}_mesh"
        mesh.uservert = verts.ravel()
        mesh.userface = faces.ravel()
        geom = _add_geom(spec, body, "visual", contype=0, conaffinity=0, group=2)
        geom.name = body_name
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.rgba = [0.86, 0.58, 0.18, 1.0]
        geom.mass = 0.0

    n_prim = 0
    if collision:
        for prim in primitives(design, side):
            geom = _add_geom(spec, body, "collision", contype=1, conaffinity=1, group=3)
            geom.name = f"{body_name}_{prim['name']}"
            geom.pos = prim["pos"]
            geom.quat = _quat(prim["rot"])
            geom.mass = 0.0
            if prim["type"] == "box":
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = prim["size"] / 2.0          # payload.json gives full extents
            elif prim["type"] == "cylinder":
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size = [prim["size"][0], prim["size"][1] / 2.0, 0.0]
            else:
                raise ValueError(f"unknown primitive type {prim['type']!r}")
            n_prim += 1

    pos, R = camera_pose(design, side)
    fov = fovy if fovy is not None else lens_fov(design, lens)[1]
    if jitter:
        rng = rng or np.random.default_rng()
        pos = pos + rng.uniform(-1, 1, 3) * jitter.get("pos_mm", 0.0) / 1000.0
        if jitter.get("rot_deg"):
            R = R @ _small_rotation(rng.uniform(-1, 1, 3) * np.radians(jitter["rot_deg"]))
        fov += rng.uniform(-1, 1) * jitter.get("fovy_deg", 0.0)
    cam = body.add_camera()
    cam.name = camera_name
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    cam.pos = pos
    cam.quat = _quat(R)
    cam.fovy = float(fov)
    cam.resolution = RESOLUTION

    return dict(body=body_name, camera=camera_name, hand="right" if side > 0 else "left",
                mass_kg=mass, com_m=com, fullinertia=full, n_collision_geoms=n_prim,
                pos_m=pos, mat=R, fovy_deg=float(fov), lens=lens,
                fov_deg=lens_fov(design, lens), resolution=RESOLUTION,
                K=wrist_intrinsics(*RESOLUTION, fov))


def _has_body(spec, name):
    try:
        return spec.body(name) is not None
    except (KeyError, ValueError):
        return False


def _add_geom(spec, body, cls, **fallback):
    """Add a geom in one of ``a1x.xml``'s geom classes.

    ``urdf2mjcf.py`` writes a ``visual`` class (``contype``/``conaffinity`` 0,
    group 2) and a ``collision`` class (group 3); the planners read those
    groups and so does the viewer.  The class has to be handed to ``add_geom``:
    assigning ``.classname`` afterwards renames the class in the XML but does
    not re-apply its values, so the geom keeps the root default's
    ``contype=1, group=0`` -- a "visual" mesh that collides with the table.
    With no such class in the spec -- someone else's arm XML -- set the same
    attributes by hand instead of landing in the root class silently."""
    default = spec.find_default(cls)
    if default is not None:
        return body.add_geom(default=default)
    return body.add_geom(**fallback)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _small_rotation(rotvec):
    """Rotation matrix from a small rotation vector, Rodrigues."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(rotvec, float) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def jitter_camera(model, rng, pos_mm=2.0, rot_deg=1.5, fovy_deg=0.0, camera=None):
    """Perturb a *compiled* model's wrist camera. Returns what it drew.

    This, not the ``jitter`` argument of ``attach_wrist_camera``, is the one to
    use for per-episode domain randomisation: both scene builders cache one arm
    ``MjSpec`` and attach a copy of it to every scene, so a draw made at attach
    time is made once for the whole run.  A compiled model is per episode.

    Each range is a half-range, drawn uniformly.  The collision geoms are left
    alone on purpose: this stands for clamp and assembly tolerance, which the
    policy must not be able to lean on, while the bracket's shape is known.

    ``fovy_deg`` defaults to 0.  Jittering the field of view changes K, and nothing
    downstream records K per episode -- ``sim/DATASET.md`` argues for storing no
    ``wrist_K`` precisely because the mount's datum makes it a constant of the robot.
    Both cannot be true at once.  Pose jitter is what a policy must be robust to and
    it leaves K alone; if you do want lens-to-lens variation, pass ``fovy_deg`` and
    store ``observation.wrist_K`` with it.
    """
    cid = model.camera(camera or wrist_camera_name(model)).id
    dpos = rng.uniform(-1, 1, 3) * pos_mm / 1000.0
    drot = rng.uniform(-1, 1, 3) * np.radians(rot_deg)
    dfov = float(rng.uniform(-1, 1) * fovy_deg)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.cam_quat[cid])
    model.cam_pos[cid] = model.cam_pos[cid] + dpos
    model.cam_quat[cid] = _quat(R.reshape(3, 3) @ _small_rotation(drot))
    model.cam_fovy[cid] = model.cam_fovy[cid] + dfov
    return dict(pos_m=dpos, rot_rad=drot, fovy_deg=dfov,
                K=wrist_intrinsics(*RESOLUTION, float(model.cam_fovy[cid])))


def wrist_intrinsics(W=RESOLUTION[0], H=RESOLUTION[1], fovy=None, design=None,
                     lens="recommended"):
    """Pinhole K of the wrist render, same convention as ``pour_scene.cam_intrinsics``."""
    if fovy is None:
        fovy = lens_fov(design or load_spec(), lens)[1]
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])


def render_wrist(model, data, W=RESOLUTION[0], H=RESOLUTION[1], camera=None, renderer=None):
    """One RGB frame from the wrist camera.

    ``camera`` defaults to ``arm/wrist`` if the scene has it and ``wrist``
    otherwise, so this works on a bare arm and on an attached one.  Pass
    ``renderer`` to reuse one across a whole episode; building one per frame
    costs an OpenGL context each time."""
    camera = camera or wrist_camera_name(model)
    if renderer is not None:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=camera)
        return r.render().copy()


def wrist_camera_name(model, camera=CAMERA):
    """The wrist camera's name in this model, with or without the scene prefix."""
    for name in (f"arm/{camera}", camera):
        try:
            model.camera(name)
        except KeyError:
            continue
        return name
    raise KeyError(f"no wrist camera in this model (looked for arm/{camera} and {camera})")


def camera_pose_world(model, data, camera=None):
    """(position, OpenCV rotation: z forward, y down) of the wrist lens in world.

    Same convention ``pour_scene._cam_pose_world`` uses, so a wrist frame can
    be fed to the same projection code as the laptop frames."""
    cid = model.camera(camera or wrist_camera_name(model)).id
    return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])


def project(K, T_world2cam, points):
    """World points -> (u, v) pixels and depth, for checking what is in frame."""
    pts = np.atleast_2d(np.asarray(points, float))
    local = pts @ T_world2cam[:3, :3].T + T_world2cam[:3, 3]
    uv = (local @ np.asarray(K, float).T)
    depth = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / depth[:, None]
    return uv, depth


# --------------------------------------------------------------------------
# the same thing as XML, for people not using MjSpec
# --------------------------------------------------------------------------

def write_mjcf_fragment(path=None, hand="right", design=None, lens="recommended",
                        body_name=BODY, camera_name=CAMERA, mesh_dir=None):
    """The mount as an MJCF snippet to paste inside ``<body name="gripper_link">``.

    Returns the XML text and, with ``path``, writes it.  The ``<mesh>`` line belongs
    in ``<asset>``; everything else is the body.

    ``mesh_dir`` is where the reader's XML will look for the STL.  It defaults to a
    path relative to ``path``'s directory, or to MESH_DIR only when nothing else is
    known -- the first cut always wrote this package's absolute path, so the snippet
    the README told people to paste resolved on one machine and nowhere else, and the
    in-process equality test could never notice."""
    if mesh_dir is None:
        mesh_dir = (os.path.relpath(MESH_DIR, os.path.dirname(os.path.abspath(path)))
                    if path else MESH_DIR)
    design = design or load_spec()
    side = _side(hand)
    mass, com, full = inertial(design, side)
    mesh = os.path.join(mesh_dir, f"wrist_mount_{'right' if side > 0 else 'left'}.stl")
    pos, R = camera_pose(design, side)
    q = _quat(R)
    lines = [f'<!-- G1 wrist-camera mount, {("right" if side > 0 else "left")} hand.',
             '     Generated by sim/wrist_camera.py: python -m wrist_camera --xml -->',
             f'<!-- in <asset>: --> <mesh name="{body_name}_mesh" file="{mesh}"/>',
             f'<body name="{body_name}" pos="0 0 0">',
             f'  <inertial pos="{_f(com)}" mass="{mass:.6f}" fullinertia="{_f(full)}"/>',
             f'  <geom class="visual" name="{body_name}" type="mesh" '
             f'mesh="{body_name}_mesh" rgba="0.86 0.58 0.18 1" mass="0"/>']
    for prim in primitives(design, side):
        pq = _quat(prim["rot"])
        if prim["type"] == "box":
            size = prim["size"] / 2.0
        else:
            size = [prim["size"][0], prim["size"][1] / 2.0]
        lines.append(f'  <geom class="collision" name="{body_name}_{prim["name"]}" '
                     f'type="{prim["type"]}" pos="{_f(prim["pos"])}" quat="{_f(pq)}" '
                     f'size="{_f(size)}" mass="0"/>')
    lines.append(f'  <camera name="{camera_name}" mode="fixed" pos="{_f(pos)}" '
                 f'quat="{_f(q)}" fovy="{lens_fov(design, lens)[1]:.4f}" '
                 f'resolution="{RESOLUTION[0]} {RESOLUTION[1]}"/>')
    lines.append('</body>')
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text


def _f(v):
    return " ".join(f"{x:.9g}" for x in np.asarray(v, float).ravel())


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hand", default="right", choices=("right", "left"))
    ap.add_argument("--lens", default="recommended", choices=("recommended", "alternative"))
    ap.add_argument("--xml", nargs="?", const="-", help="print (or write) the MJCF fragment")
    args = ap.parse_args()
    if args.xml:
        text = write_mjcf_fragment(None if args.xml == "-" else args.xml,
                                   hand=args.hand, lens=args.lens)
        if args.xml == "-":
            print(text, end="")
        return
    design = load_spec()
    mass, com, full = inertial(design, args.hand)
    pos, R = camera_pose(design, args.hand)
    print(f"{args.hand} hand, lens {args.lens} {lens_fov(design, args.lens)} deg")
    print(f"  mass        {mass * 1000:.1f} g")
    print(f"  com         {np.round(com * 1000, 2)} mm")
    print(f"  fullinertia {np.round(full * 1e6, 3)} g mm^2 x1e-3")
    print(f"  lens at     {np.round(pos * 1000, 2)} mm, optical axis {np.round(-R[:, 2], 4)}")
    print(f"  primitives  {len(primitives(design, args.hand))}")
    print(f"  K           {np.round(wrist_intrinsics(lens=args.lens, design=design), 2).tolist()}")


if __name__ == "__main__":
    main()
