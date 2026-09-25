#!/usr/bin/env python3
"""Prove the wrist camera is usable for training, on both sims, with numbers.

    venv/bin/python sim/test_wrist_camera.py \
        --pr1-sim  /path/to/pr1/sim \
        --pour-sim /path/to/sim_pour

Six phases, ``--phase`` to run one on its own:

  ``unit``    (a) both arm specs compile with the mount; the total mass grows by
              exactly the payload mass; no configuration gains a contact -- home,
              every jaw opening 0..50 mm and a wrist-joint sweep -- plus a random
              full-arm audit for comparison with gate G5.  (c) the compiled
              camera's pose in the gripper frame equals camera_spec.json's
              ``T_gripper_camera_opencv`` to 1e-6, and the documented MJCF
              fragment compiles to the same model as the MjSpec route.
  ``frames``  (b) the wrist render at home and at a real pre-grasp pose puts both
              fingertips and a point 250 mm ahead on the tool axis inside the
              frame, checked by projecting them with the published intrinsics;
              and how much of the frame the mount and its shadow take.  Seconds
              to run, so re-run it whenever the camera pose moves.
  ``filter``  (f) what the `DATASET.md` 1b planner patch is worth on *this*
              geometry: over 6000 random configurations, how many put a mount
              geom against another arm link and how many of those the shipped
              ``Motion.collides`` skips as a self-contact. Those two counts were
              quoted in DATASET.md from a measurement taken on an earlier
              revision of the bracket; this measures them.
  ``view``    which hand sees the task: glass and bottle mouth in frame at the
              planner's own grasp poses over N seeds, and which way it rolls the
              wrist to pour.  Planning only, no stepping.
  ``pr1``     (d) PR #1 pick-and-place over N seeds, bare / right / left.
  ``pour``    (e) the bottle pour the same way, plus the closest approach of any
              mount geom to the bottle, the glass, the table, the fingers and the
              arm during every episode, and one episode's wrist frames as a
              contact sheet.

The two sims cannot share a process -- both ship a ``planner`` package and a
``pickplace_scene`` -- so each phase runs as a subprocess of this same file.
Nothing under pr1/ or sim_pour/ is written to or edited: the arm spec is
swapped by rebinding ``_arm_spec`` on the scene module.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402

from wrist_camera import (attach_wrist_camera, load_spec, primitives,  # noqa: E402
                          render_wrist, wrist_camera_name, wrist_intrinsics)

RENDERS = os.path.join(HERE, "renders")
# Three ranges, every time.  Seeds 0-19 on their own are one sample, and the
# previous revision quoted their minima as if they were a property of the design;
# on 100-119 the right-hand mount touches the glass.  A reviewer then ran 200-207
# held out and the *recommendation* flipped -- the right hand scored better there
# -- which is what two ranges cannot tell you and three can: the verdict is a vote
# rather than a draw. 200-219 is the third.
DEFAULT_SEED_RANGES = "0-19,100-119,200-219"
MOUNT_BODY = "arm/wrist_camera_mount"
BARE_BODY = "wrist_camera_mount"
CAMERA = "wrist"
JAW_OPENINGS = (0.0, 0.01, 0.02, 0.035, 0.05)
FORWARD_MM = 250.0          # past the fingertips, on the tool axis (gate G3)
FINGERTIP_LOCAL = np.array([0.042, -0.012, 0.0])   # tip pad OUTER corner (see below)
AUDIT = 2000                # random full-arm configurations in check (a)


def g5_reference():
    """What validation.json records, for the comparison in phase `unit`.

    Read, not typed: these were two module constants that had to be edited by hand
    whenever the CAD changed, and nothing checked that they still matched the file
    they claim to quote.
    """
    path = os.path.join(os.path.dirname(HERE), "validation.json")
    with open(path) as f:
        audit = json.load(f)["G5_arm_audit"]
    return ({h: v["payload_collisions"] for h, v in audit["per_hand"].items()},
            audit["per_hand"]["right"]["bare_arm_clear"])


class Report:
    """Collects pass/fail lines and prints them as they happen."""

    def __init__(self):
        self.rows, self.failed = [], 0

    def check(self, name, ok, got, want):
        self.rows.append(dict(name=name, ok=bool(ok), got=str(got), want=str(want)))
        self.failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got} (want {want})", flush=True)
        return bool(ok)

    def note(self, text):
        self.rows.append(dict(note=text))
        print(f"       {text}", flush=True)


# --------------------------------------------------------------------------
# helpers shared by the phases
# --------------------------------------------------------------------------

def mounted_arm_spec(xml, hand="right", **kw):
    spec = mujoco.MjSpec.from_file(xml)
    info = attach_wrist_camera(spec, hand=hand, **kw)
    return spec, info


def patch_arm_spec(module, xml, hand="right", lens="recommended"):
    """Make a scene module hand out a mounted arm. Returns the info dict.

    Both scene modules cache one ``MjSpec`` in a module global and ``.copy()``
    it per attach, so replacing the function is enough and costs one compile."""
    cache = {}

    def _arm_spec():
        if "spec" not in cache:
            cache["spec"], cache["info"] = mounted_arm_spec(xml, hand=hand,
                                                            lens=lens)
        return cache["spec"]

    module._arm_spec = _arm_spec
    _arm_spec()
    return cache["info"]


def mount_geoms(model, body=MOUNT_BODY):
    bid = model.body(body).id
    return [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_contype[g]]


def fingertip_points(model, data, prefix="arm/"):
    """World positions of the two fingertip pads' inner front corners.

    Taken from the finger bodies' own frames rather than from a named site, because
    ``a1x.xml`` leaves the tip collision boxes unnamed.  The local offset is that
    box's **outer** front corner at z = 0 -- ``a1x.xml`` puts the tip box at
    pos (0.038, -0.0085, 0) with half-sizes (0.004, 0.0035, 0.006), so y spans
    -0.012..-0.005 and -0.012 is the outer face.  Outer is deliberate: it is the
    harder corner to keep in frame, so the G3 framing check is conservative.  Do not
    reuse this constant for a jaw-gap or grasp-width measurement -- for that the
    inner face is -0.005, 14 mm away."""
    out = []
    for i, sign in ((1, 1.0), (2, -1.0)):
        b = model.body(f"{prefix}gripper_finger_link{i}").id
        R = data.xmat[b].reshape(3, 3)
        out.append(data.xpos[b] + R @ (FINGERTIP_LOCAL * [1.0, sign, sign]))
    return np.array(out)


def forward_point(model, data, prefix="arm/", ahead=FORWARD_MM / 1000.0):
    """A point on the tool axis `ahead` metres past the fingertips."""
    site = model.site(f"{prefix}tcp").id
    pos = data.site_xpos[site]
    x = data.site_xmat[site].reshape(3, 3)[:, 0]
    return pos + x * (0.033 + ahead)


def in_frame(model, data, points, W, H, camera=None):
    """(uv, visible) of world points in the wrist image, by the published K."""
    from wrist_camera import camera_pose_world, project
    name = camera or wrist_camera_name(model)
    pos, R = camera_pose_world(model, data, name)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ pos
    K = wrist_intrinsics(W, H, float(model.cam_fovy[model.camera(name).id]))
    uv, depth = project(K, T, points)
    ok = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, ok


def save_png(path, frame):
    """Write one RGB frame, optimised: these live in the repo."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(frame, np.uint8)).save(path, optimize=True)
    return path


def save_sequence(path, frames, order=("pick_start", "pick_end", "lift_end",
                                       "pour_start", "pour_end", "place_end"),
                  cell=(320, 240), cols=3):
    """One labelled contact sheet from an episode's wrist frames.

    A sheet rather than a dozen files: the point of these is the *sequence*,
    and a reviewer should not have to open six images to see it."""
    from PIL import Image, ImageDraw
    keys = [k for k in order if k in frames]
    if not keys:
        return None
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cell[0] * cols, cell[1] * rows), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    for i, key in enumerate(keys):
        tile = Image.fromarray(np.asarray(frames[key], np.uint8)).resize(cell, Image.LANCZOS)
        x, y = cell[0] * (i % cols), cell[1] * (i // cols)
        sheet.paste(tile, (x, y))
        draw.rectangle([x + 3, y + 3, x + 9 + 6 * len(key), y + 16], fill=(16, 18, 22))
        draw.text((x + 6, y + 5), key, fill=(250, 210, 120))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, optimize=True)
    return path


def annotate(frame, uv, ok, radius=6):
    """Mark projected points on a copy of the frame: green in, red out."""
    out = np.array(frame, np.uint8, copy=True)
    H, W = out.shape[:2]
    for (u, v), good in zip(np.asarray(uv), np.asarray(ok)):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        u, v = int(round(u)), int(round(v))
        colour = (60, 230, 90) if good else (240, 60, 60)
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if abs(du) != radius and abs(dv) != radius:
                    continue
                if 0 <= v + dv < H and 0 <= u + du < W:
                    out[v + dv, u + du] = colour
    return out


def min_distance(model, data, geoms_a, geoms_b, distmax=0.2):
    """(closest approach [m], the pair) between two geom sets, ``mj_geomDistance``."""
    best, pair = distmax, None
    for ga in geoms_a:
        for gb in geoms_b:
            d = mujoco.mj_geomDistance(model, data, ga, gb, best, None)
            if d < best:
                best, pair = d, (ga, gb)
    return best, pair


def body_geoms(model, bodies, collidable=True):
    """Geom ids of the named bodies, by default only the ones that collide.

    The filter matters: a body's *visual* mesh geom is the whole vendor STL,
    and MuJoCo measures distance to its **convex hull**. The hull of a finger
    spans from the carriage inside the rail to the blade tip, filling in the
    Z = 0 band where this bracket's locating features live, so distances to it
    read as several millimetres of penetration that no collision model has."""
    ids = {model.body(b).id for b in bodies if _has_body(model, b)}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids
            and (model.geom_contype[g] or model.geom_conaffinity[g] or not collidable)]


def _joint_limits(model):
    ids = [model.joint(f"arm_joint{i}").id for i in range(1, 7)]
    return model.jnt_range[ids, 0].copy(), model.jnt_range[ids, 1].copy()


def _contact_sweep(bare, model, configs, jaws=JAW_OPENINGS):
    """(configs that gained a contact, configs skipped, first offender).

    A configuration where the *bare* arm already self-collides is skipped, not
    counted: the mount cannot be blamed for a fold the arm does on its own, and
    this is the same exclusion ``checks.g5_arm_audit`` makes."""
    d0, d1 = mujoco.MjData(bare), mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(f"arm_joint{i}").id] for i in range(1, 7)]
    fadr = [model.jnt_qposadr[model.joint(f"gripper_finger_joint{i}").id] for i in (1, 2)]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or
             (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "?")
             for g in range(model.ngeom)]
    extra, skipped, first = 0, 0, None
    for label, q in configs:
        busy = False
        for jaw in jaws:
            for data, mdl in ((d0, bare), (d1, model)):
                data.qpos[:] = 0.0
                data.qpos[qadr] = q
                data.qpos[fadr] = (jaw, -jaw)
                mujoco.mj_forward(mdl, data)
            if d0.ncon:
                busy = True
                continue
            if d1.ncon:
                extra += 1
                if first is None:
                    pairs = sorted({f"{names[c.geom1]} / {names[c.geom2]}"
                                    for c in (d1.contact[i] for i in range(d1.ncon))})
                    first = f"{label} jaw {jaw * 1000:.0f} mm: {pairs}"
        skipped += busy
    return extra, skipped, first


def _has_body(model, name):
    try:
        model.body(name)
        return True
    except KeyError:
        return False


# --------------------------------------------------------------------------
# (a) compiles, mass, no new contacts   +   (c) extrinsics
# --------------------------------------------------------------------------

def phase_unit(args, rep):
    """(a) compiles, mass, contacts, and (c) the camera's pose in the model."""
    design = load_spec()
    specs = {"pr1": os.path.join(args.pr1_sim, "a1x.xml"),
             "pour": os.path.join(args.pour_sim, "a1x.xml")}
    print("a  both arm specs, mass and contacts")
    models = {}
    for tag, xml in specs.items():
        bare = mujoco.MjSpec.from_file(xml).compile()
        for hand in ("right", "left"):
            spec, info = mounted_arm_spec(xml, hand=hand)
            model = spec.compile()
            models[(tag, hand)] = (bare, model, info)
            delta = float(model.body_mass.sum() - bare.body_mass.sum())
            rep.check(f"a {tag} {hand}: mass delta == payload",
                      abs(delta - info["mass_kg"]) < 1e-9,
                      f"{delta * 1000:.4f} g", f"{info['mass_kg'] * 1000:.4f} g")
            n_geom = model.ngeom - bare.ngeom
            rep.check(f"a {tag} {hand}: geoms added",
                      n_geom == info["n_collision_geoms"] + 1,
                      n_geom, f"{info['n_collision_geoms']} collision + 1 visual")

    bare, model, info = models[("pour", "right")]
    # inertia the compiler actually built, back-transformed to the gripper origin
    bid = model.body(BARE_BODY).id
    quat = model.body_iquat[bid]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    I_com = R @ np.diag(model.body_inertia[bid]) @ R.T
    c = model.body_ipos[bid]
    m = float(model.body_mass[bid])
    I_org = I_com + m * ((c @ c) * np.eye(3) - np.outer(c, c))
    want = np.asarray(design["payload"]["inertia_about_gripper_origin_kg_m2"], float)
    rep.check("a compiled inertia == payload.json about the gripper origin",
              np.abs(I_org - want).max() < 1e-9,
              f"max |diff| {np.abs(I_org - want).max():.2e} kg m2", "< 1e-9")

    # ---- no new contacts: home, every jaw opening, the whole wrist sweep
    for hand in ("right", "left"):
        bare, model, info = models[("pour", hand)]
        home = np.array([0, 1, -1.6, 0.6, 0, 0], float)
        lo, hi = _joint_limits(model)
        configs = [("home", home)]
        for j4 in np.linspace(lo[3], hi[3], 7):
            for j5 in np.linspace(lo[4], hi[4], 7):
                for j6 in np.linspace(lo[5], hi[5], 13):
                    q = home.copy()
                    q[3], q[4], q[5] = j4, j5, j6
                    configs.append((f"wrist {j4:+.2f}/{j5:+.2f}/{j6:+.2f}", q))
        extra, skipped, first = _contact_sweep(bare, model, configs)
        rep.check(f"a {hand}: no new contacts over home + the wrist sweep "
                  f"({len(configs)} configs x {len(JAW_OPENINGS)} jaw openings)",
                  extra == 0, f"{extra} config(s) with a new contact", 0)
        if first:
            rep.note(f"first offender: {first}")
        if skipped:
            rep.note(f"{skipped} config(s) skipped: the bare arm already self-collides there")

        # Reported, not gated: the same random full-arm audit G5 runs, but in
        # MuJoCo against the convex hulls instead of python-fcl against the
        # meshes. G5 allows 61 hits in 10,000; this is the independent check
        # that the number is of that order and not ten times it.
        rng = np.random.default_rng(20260920)
        audit = [(f"random {k}", rng.uniform(lo, hi)) for k in range(AUDIT)]
        jaws = (0.0, 0.02, 0.05)
        hits, bare_busy, example = _contact_sweep(bare, model, audit, jaws=jaws)
        # `_contact_sweep` counts (config, jaw) pairs; `checks.g5_arm_audit` tests one
        # jaw per config and counts configs.  Comparing the two directly -- which the
        # first cut of this did -- is out by the number of jaws, and the "18 vs 24,
        # against the 10 predicted" note it produced was that factor of three, not a
        # finding.
        clear = len(audit) - bare_busy
        g5, g5_clear = g5_reference()
        rep.note(f"{hand}: random audit {hits} hit(s) in {clear * len(jaws)} "
                 f"(config, jaw) pairs from {clear} bare-clear configs of "
                 f"{len(audit)}, i.e. {hits / (clear * len(jaws)):.3%}; G5 measures "
                 f"{g5[hand]}/{g5_clear} = {g5[hand] / g5_clear:.3%} of configs "
                 f"with python-fcl against the meshes. e.g. {example}")

    print("c  extrinsics")
    for hand in ("right", "left"):
        _, model, _ = models[("pour", hand)]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        gid = model.body("gripper_link").id
        cid = model.camera("wrist").id
        R_g = data.xmat[gid].reshape(3, 3)
        R_cam = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # OpenCV
        T = np.eye(4)
        T[:3, :3] = R_g.T @ R_cam
        T[:3, 3] = R_g.T @ (data.cam_xpos[cid] - data.xpos[gid])
        want = np.asarray(design["camera"]["T_gripper_camera_opencv"], float)
        if hand == "left":
            M = np.diag([1.0, -1.0, 1.0, 1.0])
            want = M @ want @ np.diag([-1.0, 1.0, 1.0, 1.0])
        err = float(np.abs(T - want).max())
        rep.check(f"c {hand}: T_gripper_camera matches camera_spec.json",
                  err <= 1e-6, f"max |diff| {err:.2e}", "<= 1e-6")
        rep.check(f"c {hand}: fovy is the recommended lens's vertical FoV",
                  abs(model.cam_fovy[cid] - design["camera"]["lens"]["recommended"]["fov_deg"][1]) < 1e-9,
                  model.cam_fovy[cid], design["camera"]["lens"]["recommended"]["fov_deg"][1])
        rep.check(f"c {hand}: primitives are proper rotations",
                  all(np.linalg.det(p["rot"]) > 0.999 for p in primitives(design, hand)),
                  f"{len(primitives(design, hand))} primitives", "det == +1")

    # The documented pure-XML route has to produce the same model, or it is a
    # snippet nobody can trust.
    for hand in ("right", "left"):
        xml_model = _compile_fragment(os.path.join(args.pour_sim, "a1x.xml"), hand)
        _, spec_model, info = models[("pour", hand)]
        worst, where = _same_mount(xml_model, spec_model, info)
        rep.check(f"c {hand}: write_mjcf_fragment() compiles to the same mount",
                  worst < 1e-9, f"max |diff| {worst:.2e} ({where})", "< 1e-9 everywhere")
    return rep


def _same_mount(a, b, info, body=BARE_BODY):
    """Largest disagreement between two models' mount, element by element.

    Compared by *name*: the XML route lands the body among ``gripper_link``'s
    children in a different order than ``add_body`` does, so the raw arrays are
    a permutation of each other even when every element agrees."""
    worst, where = 0.0, "nothing"

    def cmp(label, x, y):
        nonlocal worst, where
        d = float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max())
        if d > worst:
            worst, where = d, label

    for model_attr in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
        cmp(model_attr, getattr(a, model_attr)[a.body(body).id],
            getattr(b, model_attr)[b.body(body).id])
    for model_attr in ("cam_pos", "cam_quat", "cam_fovy"):
        cmp(model_attr, getattr(a, model_attr)[a.camera(CAMERA).id],
            getattr(b, model_attr)[b.camera(CAMERA).id])
    names = [body] + [f"{body}_{p['name']}" for p in primitives(load_spec(), info["hand"])]
    for name in names:
        ga, gb = a.geom(name).id, b.geom(name).id
        for model_attr in ("geom_pos", "geom_quat", "geom_size", "geom_type",
                           "geom_contype", "geom_group"):
            cmp(f"{name}.{model_attr}", getattr(a, model_attr)[ga], getattr(b, model_attr)[gb])
    return worst, where


def _compile_fragment(xml, hand):
    """Compile a1x.xml with the documented MJCF snippet pasted in by hand."""
    from wrist_camera import write_mjcf_fragment
    text = write_mjcf_fragment(hand=hand)
    asset, body = [], []
    for line in text.splitlines():
        (asset if "<mesh " in line else body).append(line)
    with open(xml) as f:
        source = f.read()
    mesh_line = [ln.split("-->", 1)[1].strip() for ln in asset if "<mesh " in ln][0]
    source = source.replace("</asset>", f"  {mesh_line}\n  </asset>")
    source = source.replace('<site name="tcp"',
                            "\n".join(l for l in body if not l.startswith("<!--"))
                            + '\n<site name="tcp"')
    spec = mujoco.MjSpec.from_string(source)
    spec.modelfiledir = os.path.dirname(os.path.abspath(xml))
    return spec.compile()


# --------------------------------------------------------------------------
# (d) PR #1 pick-and-place regression
# --------------------------------------------------------------------------

def phase_pr1(args, rep):
    """(d) the PR #1 pick-and-place planner, bare against both hands."""
    sys.path.insert(0, args.pr1_sim)
    import pickplace_scene
    from planner.run_episodes import episode

    print(f"d  PR #1 pick-and-place, {args.seeds} seeds")
    xml = os.path.join(args.pr1_sim, "a1x.xml")
    results = {}
    for tag in ("bare", "right", "left"):
        if tag != "bare":
            patch_arm_spec(pickplace_scene, xml, hand=tag)
        else:
            pickplace_scene._arm_spec = _plain_arm_spec(xml)
        ok, outcomes, per_seed = 0, Counter(), {}
        t0 = time.time()
        for i in range(args.seeds):
            seed = args.seed + i
            _, _, _, res, _ = episode(seed)
            ok += res.success
            outcomes[res.stage if not res.success else "success"] += 1
            per_seed[seed] = res.success
            print(f"    [{tag:5s}] seed {seed:3d} {'SUCCESS' if res.success else 'FAIL   '} "
                  f"{res.stage:12s} {res.detail}", flush=True)
        results[tag] = dict(ok=ok, outcomes=dict(outcomes), per_seed=per_seed,
                            secs=round(time.time() - t0, 1))
        print(f"    [{tag:5s}] {ok}/{args.seeds} in {results[tag]['secs']}s  {dict(outcomes)}",
              flush=True)
    base = results["bare"]["ok"]
    for hand in ("right", "left"):
        got = results[hand]["ok"]
        rep.check(f"d pick-and-place with the {hand}-hand mount loses at most 1 seed",
                  got >= base - 1, f"{got}/{args.seeds} vs {base}/{args.seeds} bare",
                  f">= {base - 1}")
        lost = [s for s, v in results["bare"]["per_seed"].items()
                if v and not results[hand]["per_seed"][s]]
        gained = [s for s, v in results[hand]["per_seed"].items()
                  if v and not results["bare"]["per_seed"][s]]
        if lost or gained:
            rep.note(f"{hand}: lost seeds {lost}, gained seeds {gained} "
                     f"({results[hand]['outcomes']})")
    rep.rows.append(dict(data=dict(pickplace=results)))
    return rep


# --------------------------------------------------------------------------
# (b) + (e) pour scene: renders and the pour regression
# --------------------------------------------------------------------------

class PourWatch:
    """Per-step audit of one pour episode.

    Records the closest approach of any mount collision geom to the bottle, the
    glass, the table, the fingers and the arm's links -- overall and during the
    pour roll on its own, which is what decides the handedness -- every contact
    a mount geom makes, and the first and last wrist frame of each stage.

    ``gripper_link`` is left out of the distance report on purpose: MuJoCo
    collides its *convex hull*, which spans out to the rail's +-54 mm, so the
    clamp band that grips the 60 mm housing reads as 22 mm inside it.  The real
    clearance there is a mesh question and gate G1 answers it (5.40 mm); the
    pair is excluded from contacts anyway, being parent and child.
    """

    def __init__(self, model, data, renderer, hz=8.0):
        self.m, self.d, self.r, self.hz = model, data, renderer, hz
        self.mount = mount_geoms(model)
        mount_body = model.body(MOUNT_BODY).id
        skip = {mount_body, model.body("arm/gripper_link").id}
        self.targets = {
            "bottle": body_geoms(model, ["bottle"]),
            "glass": body_geoms(model, ["glass"]),
            "table": [model.geom("table").id] if _has_geom(model, "table") else [],
            "fingers": body_geoms(model, [f"arm/gripper_finger_link{i}" for i in (1, 2)]),
            "arm_links": [g for g in range(model.ngeom)
                          if model.geom_contype[g] and model.geom_bodyid[g] not in skip
                          and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                 model.geom_bodyid[g]) or "").startswith("arm/")
                          and "finger" not in (mujoco.mj_id2name(
                              model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")],
        }
        self.closest = {k: 1.0 for k in self.targets}
        self.closest_pour = {k: 1.0 for k in self.targets}
        self.witness = {}
        self.contacts = Counter()
        self.frames, self.n = {}, 0

    def __call__(self, pp):
        if self.n > int(self.d.time * self.hz):
            return
        self.n += 1
        pouring = pp.label == "pour"
        for name, geoms in self.targets.items():
            if not geoms:
                continue
            dist, pair = min_distance(self.m, self.d, self.mount, geoms)
            if dist < self.closest[name]:
                self.closest[name] = dist
                if pair:
                    self.witness[name] = "/".join(
                        mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
                        or (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                              self.m.geom_bodyid[g]) or "?") for g in pair)
            if pouring:
                self.closest_pour[name] = min(self.closest_pour[name], dist)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 not in self.mount and c.geom2 not in self.mount:
                continue
            other = c.geom2 if c.geom1 in self.mount else c.geom1
            name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                      self.m.geom_bodyid[other]) or "world")
            self.contacts[f"{pp.label}:{name}"] += 1
        if self.r is not None:
            frame = render_wrist(self.m, self.d, renderer=self.r)
            self.frames.setdefault(f"{pp.label}_start", frame)
            self.frames[f"{pp.label}_end"] = frame


def _has_geom(model, name):
    try:
        model.geom(name)
        return True
    except KeyError:
        return False


def phase_frames(args, rep):
    """(b) what the lens actually sees, in a real pour scene."""
    sys.path.insert(0, args.pour_sim)
    sys.path.insert(0, os.path.join(args.pour_sim, "planner"))
    import pour_scene
    from planner.perception import SimPourPerception
    from planner.pour import Pour

    xml = os.path.join(args.pour_sim, "a1x.xml")
    W, H = 640, 480
    print("b  wrist render at home and at a pre-grasp pose")
    info = patch_arm_spec(pour_scene, xml, hand="right")
    model, data, sceneinfo = pour_scene.build(args.seed)
    mujoco.mj_forward(model, data)
    cam = wrist_camera_name(model)
    os.makedirs(args.renders, exist_ok=True)
    shares = {}
    with mujoco.Renderer(model, height=H, width=W) as r:
        for label, place in (("home", None), ("pregrasp", "pregrasp")):
            if place == "pregrasp":
                per = SimPourPerception(model, data, sceneinfo)
                pp = Pour(model, data, per, sceneinfo)
                ok, detail = pp._perceive()
                assert ok, detail
                ok, detail = pp._plan_grasp()
                assert ok, f"no grasp on seed {args.seed}: {detail}"
                data.qpos[pp.arm.qadr] = pp.grasp.q_pre
                data.qpos[pp.arm.fadr] = (0.05, -0.05)
                mujoco.mj_forward(model, data)
                rep.note(f"pre-grasp pose from the planner: {pp.grasp.label}")
            tips = fingertip_points(model, data)
            fwd = forward_point(model, data)[None, :]
            pts = np.vstack([tips, fwd])
            uv, vis = in_frame(model, data, pts, W, H, camera=cam)
            frame = render_wrist(model, data, W, H, camera=cam, renderer=r)
            if args.render:
                save_png(os.path.join(args.renders, f"wrist_{label}.png"), frame)
                save_png(os.path.join(args.renders, f"wrist_{label}_annotated.png"),
                         annotate(frame, uv, vis))
            rep.check(f"b {label}: both fingertips in frame", bool(vis[0] and vis[1]),
                      f"uv {np.round(uv[:2], 1).tolist()}", f"inside {W}x{H}")
            rep.check(f"b {label}: point {FORWARD_MM:.0f} mm ahead of the tips in frame",
                      bool(vis[2]), f"uv {np.round(uv[2], 1).tolist()}", f"inside {W}x{H}")
            own, shade = own_image_share(model, data, camera=cam)
            shares[label] = (round(own, 4), round(shade, 4))
            rep.note(f"{label}: the mount is {own:.2%} of the frame and its shadow "
                     f"darkens another {shade:.2%}")
    # The same measurement at the OTHER lens, on a model compiled with it.  The
    # README said "own_image_share measures 0.00 % at both fields of view" and
    # MuJoCo had only ever been asked about the wide one; the narrow figure beside
    # it came from the CAD ray cast.  One more compile and one more render pass is
    # what it costs to make the sentence true.
    spec_alt = mujoco.MjSpec.from_file(xml)
    alt = attach_wrist_camera(spec_alt, hand="right", lens="alternative")
    rep.note(f"alternative lens fovy {alt['fovy_deg']:.0f} deg: "
             f"K f = {alt['K'][0][0]:.1f} px vs {info['K'][0][0]:.1f} px for the wide one")
    patch_arm_spec(pour_scene, xml, hand="right", lens="alternative")
    model_a, data_a, _ = pour_scene.build(args.seed)
    mujoco.mj_forward(model_a, data_a)
    own_a, shade_a = own_image_share(model_a, data_a,
                                     camera=wrist_camera_name(model_a))
    rep.note(f"home, alternative lens (fovy {alt['fovy_deg']:.0f} deg): the mount "
             f"is {own_a:.2%} of the frame and its shadow darkens another "
             f"{shade_a:.2%}")
    rep.rows.append(dict(data=dict(own_image_share={
        "wide": dict(fovy_deg=round(float(info["fovy_deg"]), 1),
                     home_own=shares["home"][0], home_shadow=shares["home"][1],
                     pregrasp_own=shares["pregrasp"][0],
                     pregrasp_shadow=shares["pregrasp"][1]),
        "alternative": dict(fovy_deg=round(float(alt["fovy_deg"]), 1),
                            home_own=round(own_a, 4),
                            home_shadow=round(shade_a, 4))})))
    for name, azimuth in (("mount_on_wrist", -120.0), ("mount_on_wrist_2", 60.0)):
        if args.render:
            save_png(os.path.join(args.renders, f"{name}.png"),
                     third_person(model, data, azimuth))
    return rep


def third_person(model, data, azimuth, distance=0.45, elevation=-20.0,
                 body="arm/gripper_link", W=900, H=600):
    """A view of the wrist from outside, to see the bracket in the scene."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = data.xpos[model.body(body).id]
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=cam)
        return r.render().copy()


def own_image_share(model, data, camera=None, body=MOUNT_BODY, threshold=6):
    """(mount's own pixels, extra pixels its shadow changes), as fractions.

    Rendered twice with the mount's visual geom made invisible, once with MuJoCo's
    shadow pass on and once off.  The two are very different numbers here and the
    difference is the point: the bracket never appears -- not because it is inside
    the near plane, which it is not, but because the gripper is in front of it on
    every ray that reaches it -- while it does stand between the lights and
    everything the lens looks at."""
    gid = model.geom(body).id
    alpha = model.geom_rgba[gid, 3]
    out = []
    with mujoco.Renderer(model, height=480, width=640) as r:
        for shadow in (0, 1):
            shots = []
            for value in (alpha, 0.0):
                model.geom_rgba[gid, 3] = value
                r.update_scene(data, camera=camera or wrist_camera_name(model))
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
                shots.append(r.render().copy())
            model.geom_rgba[gid, 3] = alpha
            changed = np.abs(shots[0].astype(int) - shots[1].astype(int)).max(axis=2) > threshold
            out.append(float(changed.mean()))
    return out[0], max(0.0, out[1] - out[0])


def phase_view(args, rep):
    """Which hand sees the task: glass and bottle in frame, over N seeds.

    Planning only, no stepping, so it runs in a couple of minutes.  The grasp
    pose comes from ``_plan_grasp``; the pour pose is that grasp rolled by the
    angle the planner's own hint names, which is where the camera points while
    the liquid is running."""
    sys.path.insert(0, args.pour_sim)
    sys.path.insert(0, os.path.join(args.pour_sim, "planner"))
    import pour_scene
    from planner.perception import SimPourPerception
    from planner.pour import Pour

    xml = os.path.join(args.pour_sim, "a1x.xml")
    W, H = 640, 480
    print(f"view  what each hand sees, {args.seeds} seeds")
    seen, rolls = {}, Counter()
    for hand in ("right", "left"):
        patch_arm_spec(pour_scene, xml, hand=hand)
        tally = Counter()
        for i in range(args.seeds):
            seed = args.seed + i
            model, data, sceneinfo = pour_scene.build(seed)
            per = SimPourPerception(model, data, sceneinfo, rng=np.random.default_rng(seed))
            pp = Pour(model, data, per, sceneinfo)
            if not pp._perceive()[0] or not pp._plan_grasp()[0]:
                tally["no plan"] += 1
                continue
            if hand == "right":
                rolls[(str(pp.pour_hint).split() or ["none"])[0]] += 1
            targets = np.array([pp.glass.rim, pp.bottle.base + [0, 0, pp.bottle.height]])
            for label, q in (("grasp", pp.grasp.q_grasp),):
                data.qpos[pp.arm.qadr] = q
                data.qpos[pp.arm.fadr] = (0.05, -0.05)
                mujoco.mj_forward(model, data)
                _, vis = in_frame(model, data, targets, W, H)
                tally[f"{label}: glass"] += bool(vis[0])
                tally[f"{label}: bottle mouth"] += bool(vis[1])
        seen[hand] = dict(tally)
        rep.note(f"{hand}: {dict(tally)} of {args.seeds} seeds")
    rep.note(f"pour roll sign the planner chose: {dict(rolls)} "
             "(roll- swings a +Y payload down)")
    rep.rows.append(dict(data=dict(view=seen, roll_sign=dict(rolls))))
    return rep


FILTER_CONFIGS = 6000


def phase_filter(args, rep):
    """(f) what the DATASET.md 1b patch is worth, measured on this geometry.

    DATASET.md's central claim is a pair of counts: over N random configurations,
    how many put a mount geom against another ``arm/`` body, and for how many of
    those the shipped ``Motion.collides`` returns None -- i.e. skips a real
    collision as a self-contact.  Those counts were taken once, on a previous
    revision of the bracket, and then carried forward as a present-tense
    measurement while the geometry changed underneath them.  The mechanism is a
    string-prefix test and does not move; the counts do.  So they are measured.

    Both planners are run in one process: the shipped ``Motion.collides`` and a
    reimplementation of the patched rule (``patched_planner``'s diff, applied to
    the same contact list), over exactly the same configurations.
    """
    sys.path.insert(0, args.pour_sim)
    sys.path.insert(0, os.path.join(args.pour_sim, "planner"))
    import pour_scene
    from planner.motion import Motion
    from planner.perception import SimPourPerception
    from planner.pour import Pour

    xml = os.path.join(args.pour_sim, "a1x.xml")
    print(f"f  planner collision filter, {FILTER_CONFIGS} configurations")
    out = {}
    for hand in ("right", "left"):
        patch_arm_spec(pour_scene, xml, hand=hand)
        model, data, sceneinfo = pour_scene.build(args.seed)
        per = SimPourPerception(model, data, sceneinfo,
                                rng=np.random.default_rng(args.seed))
        pp = Pour(model, data, per, sceneinfo)
        assert pp._perceive()[0], "no scene"
        motion = Motion(model, data, pp.arm)
        # The scene compiles the arm under an "arm/" prefix, so the joints are
        # not the bare names _joint_limits uses on a plain arm model.
        ids = [model.joint(f"arm/arm_joint{i}").id for i in range(1, 7)]
        lo = model.jnt_range[ids, 0].copy()
        hi = model.jnt_range[ids, 1].copy()
        rng = np.random.default_rng(20260921)
        poses = rng.uniform(lo, hi, (FILTER_CONFIGS, len(lo)))
        mount_bodies = {MOUNT_BODY}
        touching = missed = 0
        for q in poses:
            s = motion._pose(q, 0.02, None)
            hits = []
            for i in range(s.ncon):
                c = s.contact[i]
                b1 = motion.body_of_geom[c.geom1]
                b2 = motion.body_of_geom[c.geom2]
                on_mount = [b in mount_bodies for b in (b1, b2)]
                if on_mount[0] == on_mount[1]:
                    continue
                other = b2 if on_mount[0] else b1
                # The mount's own parent is excluded by MuJoCo's parent filter
                # anyway, and is not a collision.  Only other *arm* bodies are
                # counted here, because that is the claim: the shipped filter
                # calls an arm/-vs-arm/ contact a self-contact and skips it.
                if other == "arm/gripper_link" or not other.startswith("arm/"):
                    continue
                hits.append(other)
            if not hits:
                continue
            touching += 1
            # What the SHIPPED filter says about this configuration.
            if motion.collides(q, 0.02) is None:
                missed += 1
        out[hand] = dict(configurations=FILTER_CONFIGS,
                         mount_against_arm=touching,
                         shipped_filter_missed=missed,
                         # The `hits` loop above *is* the patched rule (the same
                         # test MOTION_PATCH inserts), so a configuration it
                         # counts is one the patched filter refuses. Zero by
                         # construction, and said so rather than run twice.
                         patched_filter_missed=0,
                         jaw_half_gap_mm=20.0)
        rep.note(f"f {hand}: of {FILTER_CONFIGS} random configurations, {touching} "
                 f"put a mount geom against another arm/ body and the shipped "
                 f"Motion.collides returned None for {missed} of them; with the "
                 f"DATASET.md 1b patch, {touching} and 0")
        rep.check(f"f {hand}: the shipped filter really does skip mount-vs-arm "
                  f"contacts, which is what 1b is for",
                  missed > 0 or touching == 0,
                  f"{missed} of {touching} skipped unpatched",
                  "the shipped filter skips them; the patch does not")
    rep.rows.append(dict(data=dict(collision_filter=out)))
    return rep


def phase_pour(args, rep):
    """(e) the pour regression, both hands.

    With ``--guard`` the planner's own collision filter is patched first (see
    ``patched_planner``): unpatched, ``Motion.collides`` calls every mount-vs-arm and
    mount-vs-held-object contact a self-contact and skips it, so the collision geoms
    this package adds protect against the table, the bottle and the glass but not
    against the arm's own links.  The README quotes both runs.
    """
    sys.path.insert(0, args.pour_sim)
    sys.path.insert(0, os.path.join(args.pour_sim, "planner"))
    import pour_scene
    scratch = None
    if args.guard:
        # After `import pour_scene`, because that module puts its own directory at
        # sys.path[0] on import -- and that directory contains the `planner` package
        # this is trying to shadow. Inserting first and importing second loses.
        scratch = tempfile.mkdtemp(prefix="wrist_camera_guard_")
        sys.path.insert(0, patched_planner(args.pour_sim, scratch))
        for name in [m for m in sys.modules
                     if m == "planner" or m.startswith("planner.")]:
            del sys.modules[name]
    from planner.perception import SimPourPerception
    from planner.pour import Pour

    xml = os.path.join(args.pour_sim, "a1x.xml")
    W, H = 640, 480
    import planner.motion as _motion
    guarded = hasattr(_motion, "MOUNT_BODIES")
    if args.guard != guarded:
        raise SystemExit(f"--guard {args.guard} but planner.motion guarded={guarded}")
    rep.note(f"planner mount guard: {'ON (patched copy)' if guarded else 'OFF (as shipped)'}")
    print(f"e  bottle pour, {args.seeds} seeds"
          + (" [mount guard ON]" if guarded else ""))
    results = {}
    for tag in ("bare", "right", "left"):
        if tag == "bare":
            pour_scene._arm_spec = _plain_arm_spec(xml)
        else:
            patch_arm_spec(pour_scene, xml, hand=tag)
        ok, outcomes, per_seed = 0, Counter(), {}
        closest, closest_pour, witness = {}, {}, {}
        contacts, rolls = Counter(), Counter()
        t0 = time.time()
        # Reset per tag.  Left as it was, `frames is None` is already False by the
        # time the left hand runs, so the left hand recorded nothing and
        # `pour_left_sequence.png` was written from the RIGHT hand's frames -- which
        # is why the committed pair does not regenerate from the documented command.
        frames = None
        for i in range(args.seeds):
            seed = args.seed + i
            model, data, sceneinfo = pour_scene.build(seed)
            per = SimPourPerception(model, data, sceneinfo,
                                    rng=np.random.default_rng(seed))
            pp = Pour(model, data, per, sceneinfo)
            watch = None
            if tag != "bare":
                want_frames = frames is None and i == 0
                r = mujoco.Renderer(model, height=H, width=W) if want_frames else None
                watch = PourWatch(model, data, r)
                try:
                    res = pp.run(on_step=watch)
                finally:
                    if r is not None:
                        r.close()
                for k, v in watch.closest.items():
                    if v < closest.get(k, 1.0):
                        closest[k] = v
                        witness[k] = watch.witness.get(k, "?")
                for k, v in watch.closest_pour.items():
                    closest_pour[k] = min(closest_pour.get(k, 1.0), v)
                contacts.update(watch.contacts)
                if want_frames:
                    frames = watch.frames
            else:
                res = pp.run()
            ok += res.success
            outcomes[res.stage if not res.success else "success"] += 1
            per_seed[seed] = res.success
            # Which way the wrist rolled decides which hand the payload swings
            # away from the table on; `_pour_axes` offers roll+ first but
            # `_try_pours` takes the first that *solves*, which is often roll-.
            # `or "none"` after the index is too late: an empty hint splits to []
            # and the subscript raises before `or` is ever evaluated.
            rolls[(str(getattr(pp, "pour_hint", "")).split() or ["none"])[0]] += 1
            b = sceneinfo["bottle"]
            print(f"    [{tag:5s}] seed {seed:3d} {'SUCCESS' if res.success else 'FAIL   '} "
                  f"{res.stage:13s} [r={b['body_r'] * 1000:.0f} h={b['height'] * 1000:.0f}] "
                  f"t={res.duration:5.1f}s {res.detail}", flush=True)
        results[tag] = dict(ok=ok, outcomes=dict(outcomes), per_seed=per_seed,
                            closest_mm={k: round(v * 1000, 2) for k, v in closest.items()},
                            closest_during_pour_mm={k: round(v * 1000, 2)
                                                    for k, v in closest_pour.items()},
                            closest_pair=witness, pour_roll_sign=dict(rolls),
                            mount_contacts=dict(contacts),
                            secs=round(time.time() - t0, 1))
        print(f"    [{tag:5s}] {ok}/{args.seeds} in {results[tag]['secs']}s "
              f"{dict(outcomes)} closest {results[tag]['closest_mm']}", flush=True)
        if frames and args.render:
            save_sequence(os.path.join(args.renders,
                                       f"pour_{tag}_sequence.png"), frames)

    base = results["bare"]["ok"]
    for hand in ("right", "left"):
        r = results[hand]
        rep.check(f"e pour with the {hand}-hand mount loses at most 1 seed",
                  r["ok"] >= base - 1, f"{r['ok']}/{args.seeds} vs {base}/{args.seeds} bare",
                  f">= {base - 1}")
        rep.check(f"e {hand}: the mount touched nothing at all over seeds "
                  f"{args.seed}-{args.seed + args.seeds - 1}",
                  not r["mount_contacts"], r["mount_contacts"] or "none", "none")
        rep.note(f"{hand}: closest approach over the whole episode [mm] {r['closest_mm']}")
        rep.note(f"{hand}: the geoms that got there {r['closest_pair']}")
        rep.note(f"{hand}: pour roll sign chosen by the planner {r['pour_roll_sign']} "
                 "(roll- swings a +Y payload down)")
        rep.note(f"{hand}: closest approach during the pour roll [mm] "
                 f"{r['closest_during_pour_mm']}")
        lost = [s for s, v in results["bare"]["per_seed"].items() if v and not r["per_seed"][s]]
        gained = [s for s, v in r["per_seed"].items() if v and not results["bare"]["per_seed"][s]]
        if lost or gained:
            rep.note(f"{hand}: lost seeds {lost}, gained seeds {gained} ({r['outcomes']})")

    def margin(h):
        """Worst clearance to anything that is not the arm, during the pour."""
        d = results[h]["closest_during_pour_mm"]
        return min(d.get(k, 0.0) for k in ("bottle", "glass", "table"))

    better = max(("right", "left"), key=lambda h: (results[h]["ok"], margin(h)))
    rep.note(f"handedness: {better} -- success {results[better]['ok']}/{args.seeds} and "
             f"{margin(better):.1f} mm worst clearance during the pour, against "
             f"{results['right' if better == 'left' else 'left']['ok']}/{args.seeds} and "
             f"{margin('right' if better == 'left' else 'left'):.1f} mm for the other hand")
    # In the data as well as in the prose: this verdict is one seed range's, it
    # flips between ranges, and the README has to be able to render the vote rather
    # than quote whichever range was run last.
    rep.rows.append(dict(data=dict(
        pour=results, guard=bool(args.guard),
        handedness=dict(better=better,
                        margin_mm={h: round(margin(h), 2) for h in ("right", "left")},
                        success={h: results[h]["ok"] for h in ("right", "left")},
                        seeds=[args.seed, args.seed + args.seeds - 1]))))
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return rep


# --------------------------------------------------------------------------
# the planner patch: a wrist payload is not "self"
# --------------------------------------------------------------------------

MOTION_ANCHOR = """            b1, b2 = self.body_of_geom[c.geom1], self.body_of_geom[c.geom2]
            mine = [b.startswith(ARM) or b in carried for b in (b1, b2)]"""

MOTION_PATCH = """            b1, b2 = self.body_of_geom[c.geom1], self.body_of_geom[c.geom2]
            # A wrist payload is bolted to the arm, but it is not "self": the arm can
            # drive it into its own links and into the object it is carrying, and
            # both of those are collisions a planner has to refuse. The ARM prefix
            # test below calls them self-contacts and skips them.
            on_mount = [b in MOUNT_BODIES for b in (b1, b2)]
            if on_mount[0] != on_mount[1]:
                g = c.geom2 if on_mount[0] else c.geom1
                other = b2 if on_mount[0] else b1
                label = self.geom_name[g] or other or "world"
                if other not in MOUNT_EXEMPT and label not in allow and other not in allow:
                    return label
                continue
            mine = [b.startswith(ARM) or b in carried for b in (b1, b2)]"""

MOTION_CONSTS = """ARM = "arm/"
MOUNT_BODIES = ("arm/wrist_camera_mount",)
# The mount's own parent; MuJoCo's parent filter excludes the pair anyway.
MOUNT_EXEMPT = ("arm/gripper_link",)"""


def patched_planner(pour_sim, scratch):
    """A copy of the pour sim's ``planner`` package with Motion.collides fixed.

    Copied rather than edited: nothing under sim_pour/ is written to.  The copy goes
    first on sys.path, so ``planner.*`` resolves to it while ``pour_scene`` and
    ``a1x_control`` still come from the sim itself.
    """
    out = os.path.join(scratch, "patched")
    os.makedirs(out, exist_ok=True)
    shutil.copytree(os.path.join(pour_sim, "planner"),
                    os.path.join(out, "planner"), dirs_exist_ok=True)
    path = os.path.join(out, "planner", "motion.py")
    with open(path) as f:
        source = f.read()
    if MOTION_ANCHOR not in source or 'ARM = "arm/"' not in source:
        raise SystemExit(f"{path}: planner/motion.py is not the version this patch "
                         "was written against; re-read it and update MOTION_PATCH")
    source = source.replace(MOTION_ANCHOR, MOTION_PATCH)
    source = source.replace('ARM = "arm/"', MOTION_CONSTS, 1)
    with open(path, "w") as f:
        f.write(source)
    return out


def _plain_arm_spec(xml):
    """A clean, unmounted arm-spec factory: the baseline the mount is judged against."""
    cache = {}

    def _arm_spec():
        if "spec" not in cache:
            cache["spec"] = mujoco.MjSpec.from_file(xml)
        return cache["spec"]

    return _arm_spec


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

PHASES = {"unit": phase_unit, "frames": phase_frames, "view": phase_view,
          "filter": phase_filter,
          "pr1": phase_pr1, "pour": phase_pour}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pr1-sim", required=True)
    ap.add_argument("--pour-sim", required=True)
    ap.add_argument("--seeds", type=int, default=20, help="episodes per configuration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", choices=sorted(PHASES), help="run one phase in this process")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--guard", action="store_true",
                    help="patch a copy of the pour sim's planner so a mount-vs-arm "
                         "or mount-vs-held-object contact is not treated as a "
                         "self-contact (phase `pour` only)")
    ap.add_argument("--seed-ranges", default=DEFAULT_SEED_RANGES,
                    help="comma-separated FIRST-LAST ranges the pour and view "
                         "phases are run over, e.g. '0-19,100-119'. Every range "
                         "run lands in the report; nothing is merged in by hand.")
    ap.add_argument("--renders", default=RENDERS,
                    help="where the PNGs go (a --check run redirects this)")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="run the checks but write no images")
    ap.add_argument("--check", action="store_true",
                    help="write nothing: run everything into a temp directory and "
                         "report any tracked file that differs")
    args = ap.parse_args()
    args.pr1_sim = os.path.abspath(args.pr1_sim)
    args.pour_sim = os.path.abspath(args.pour_sim)
    args.renders = os.path.abspath(args.renders)

    if args.phase:
        rep = Report()
        PHASES[args.phase](args, rep)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(phase=args.phase, failed=rep.failed, rows=rep.rows), f, indent=1)
        print(f"\n{args.phase}: {rep.failed} failure(s)")
        return 1 if rep.failed else 0

    ranges = parse_seed_ranges(args.seed_ranges)
    scratch = tempfile.mkdtemp(prefix="wrist_camera_test_")
    tracked_renders, out_json = args.renders, args.json
    if args.check:
        args.renders = os.path.join(scratch, "renders")
        out_json = os.path.join(scratch, "sim_validation.json")
    os.makedirs(args.renders, exist_ok=True)

    # Each sim owns a `planner` package and a `pickplace_scene`; they cannot
    # share sys.modules, so every phase gets its own interpreter.
    #
    # The run plan is built from the seed ranges rather than hard-coded, and every
    # run it produces goes into the report.  The committed file used to be one
    # 0-19 run with a seeds-100 block pasted in beside it by hand, which is not the
    # record of anything the documented command does.
    plan = [("unit", False, None), ("frames", False, None), ("filter", False, None),
            ("pr1", False, None)]
    for first, last in ranges:
        plan.append(("view", False, (first, last)))
        plan.append(("pour", False, (first, last)))
        plan.append(("pour", True, (first, last)))

    failed, runs = 0, []
    for phase, guard, span in plan:
        first, count = (span[0], span[1] - span[0] + 1) if span else (args.seed, args.seeds)
        label = f"{first}-{first + count - 1}" if span else None
        tag = phase + ("_guarded" if guard else "") + (f"_{label}" if label else "")
        out = os.path.join(scratch, f"{tag}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", phase,
               "--pr1-sim", args.pr1_sim, "--pour-sim", args.pour_sim,
               "--seeds", str(count), "--seed", str(first), "--json", out,
               "--renders", args.renders]
        if guard:
            cmd.append("--guard")
        # Only the first seed range writes the contact sheets; a second range
        # would otherwise overwrite the images the README points at.
        if span and span != ranges[0]:
            cmd.append("--no-render")
        if not args.render:
            cmd.append("--no-render")
        print(f"\n===== {tag} =====", flush=True)
        rc = subprocess.call(cmd)
        failed += rc != 0
        if os.path.exists(out):
            with open(out) as f:
                body = json.load(f)
            body["guard"] = bool(guard)
            body["seed_range"] = label
            body["seeds"] = [first, first + count - 1]
            runs.append(body)

    report = dict(
        command=("python sim/test_wrist_camera.py --pr1-sim <pr1>/sim "
                 "--pour-sim <sim_pour> --seed-ranges " + args.seed_ranges
                 + " --json sim/validation.json"),
        seed_ranges=[f"{a}-{b}" for a, b in ranges],
        phases=runs)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1)

    drift = []
    if args.check:
        drift = compare_outputs(tracked_renders, args.renders,
                                args.json, out_json)
        for line in drift:
            print(f"  DRIFT {line}")
        print(f"  {len(drift)} tracked file(s) differ from this run")
    shutil.rmtree(scratch, ignore_errors=True)
    checks = [r for p in runs for r in p["rows"] if "ok" in r]
    bad = [r["name"] for r in checks if not r["ok"]]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed"
          + (f"; failed: {bad}" if bad else ""))
    return 1 if bad or failed or drift else 0


def parse_seed_ranges(text):
    """'0-19,100-119' -> [(0, 19), (100, 119)]."""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        first, _, last = chunk.partition("-")
        out.append((int(first), int(last or first)))
    if not out:
        raise SystemExit(f"--seed-ranges: nothing to run in {text!r}")
    return out


def compare_outputs(tracked_renders, fresh_renders, tracked_json, fresh_json):
    """Which committed renders and which parts of sim/validation.json drifted."""
    drift = []
    for name in sorted(os.listdir(fresh_renders)):
        here = os.path.join(tracked_renders, name)
        there = os.path.join(fresh_renders, name)
        if not os.path.exists(here):
            drift.append(f"renders/{name}: missing from the tree")
        elif open(here, "rb").read() != open(there, "rb").read():
            drift.append(f"renders/{name}: differs")
    for name in sorted(os.listdir(tracked_renders)):
        if not os.path.exists(os.path.join(fresh_renders, name)):
            drift.append(f"renders/{name}: this run did not produce it")
    if tracked_json and os.path.exists(tracked_json):
        with open(tracked_json) as f:
            a = json.load(f)
        with open(fresh_json) as f:
            b = json.load(f)
        strip_timings(a)
        strip_timings(b)
        if a != b:
            drift.append(f"{os.path.basename(tracked_json)}: differs "
                         "(timings excluded)")
    elif tracked_json:
        drift.append(f"{tracked_json}: missing from the tree")
    return drift


def strip_timings(node):
    """Drop the wall-clock fields, which are the run and not the result."""
    if isinstance(node, dict):
        node.pop("secs", None)
        for value in node.values():
            strip_timings(value)
    elif isinstance(node, list):
        for value in node:
            strip_timings(value)


if __name__ == "__main__":
    raise SystemExit(main())
