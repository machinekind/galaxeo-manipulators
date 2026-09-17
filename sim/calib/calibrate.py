#!/usr/bin/env python3
"""Automated per-session camera calibration: the arm waves a card, the camera watches.

    sim/.venv/bin/python sim/calib/calibrate.py --sim --seed 3          # score against ground truth
    sim/.venv/bin/python sim/calib/calibrate.py --sim --n 20 --seed 0   # 20 seeds, error table
    sim/.venv/bin/python sim/calib/calibrate.py --sim --seed 3 --out session.json

The laptop is put down wherever; nothing about its pose is known. The only
fiducial is a calibration card pinched in the gripper for the length of the
session and taken out afterwards: a thin rectangle sticking out past the
fingertips with one AprilTag 36h11 per face, back to back, so whichever side
the laptop ended up on one face is readable. A human puts the card in, so its
pose in the gripper is unknown too; the solver estimates it jointly with the
camera pose (12 dof) from the wave's images (`handeye.solve`).

The wave has two phases. First a search: poses over the table with the wrist
rolled all the way round, until a tag is seen at all. That first detection
gives a rough camera position, and the collect phase then aims the card's
normal at it while spreading the gripper's orientation over three non-parallel
rotation axes, which is what makes hand-eye conditioned. The session is written
as JSON and refuses to be trusted above `MAX_PX` of residual, below
`MIN_SPREAD` of rotation spread, or under `MIN_OBS` observations.

The robot sits behind `Robot`: `SimRobot` drives the MuJoCo scene through
its position servos and renders `laptop_cam`; the real arm gets the same
three methods on top of the CAN driver. Forward kinematics for both comes
from `a1x.xml` at the *measured* joint angles, so servo error is included.
"""
import argparse
import json
import os
import sys
import time
from typing import Protocol

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a1x_control import Arm, Script, rot                                  # noqa: E402
from calib.handeye import Observation, pose_error, rotation_spread_R, solve  # noqa: E402
from calib.tags import detect, tag_pose                                   # noqa: E402
from planner.motion import Motion                                         # noqa: E402

ARM_XML = os.path.join(os.path.dirname(HERE), "a1x.xml")
MAX_PX = 1.5           # residual above this and the session is not trusted
MIN_SPREAD = 8.0       # rotation-axis spread [deg] below this and the wave was degenerate
MIN_OBS, MIN_POSES = 14, 8


class Robot(Protocol):
    def q(self) -> np.ndarray: ...                    # measured joint angles, 6
    def move(self, q, dur: float) -> None: ...        # ramp to q and settle
    def image(self) -> np.ndarray: ...                # RGB frame from the session camera


class FK:
    """Gripper body pose in the arm base frame, from a1x.xml alone."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(ARM_XML)
        self.d = mujoco.MjData(self.m)
        self.body = self.m.body("gripper_link").id
        self.qadr = np.array([self.m.joint(f"arm_joint{i}").qposadr[0] for i in range(1, 7)])

    def gripper2base(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.body].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.body]
        return T


class SimRobot:
    def __init__(self, model, data, info, cam="laptop_cam"):
        self.m, self.d, self.info = model, data, info
        self.arm = Arm(model, "arm/")
        self.script = Script(model, data, {"arm": "arm/"})
        self.motion = Motion(model, data, self.arm)
        self.cam = cam
        self.r = mujoco.Renderer(model, height=info["cam"]["H"], width=info["cam"]["W"])

    def q(self):
        return self.d.qpos[self.arm.qadr].copy()

    def move(self, q, dur=1.5):
        self.script.move_q("arm", q, dur, "calib").wait(0.4)
        while self.d.time < self.script.t - 1e-9:
            self.script.apply(self.d.time)
            mujoco.mj_step(self.m, self.d)

    def image(self):
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


# --------------------------------------------------------------------- wave
WAVE = dict(x=(-0.22, 0.22), y=(0.0, 0.28), z=(0.14, 0.34),   # TCP box over the table [m]
            yaw=0.9, pitch=(0.15, 1.25),                      # yaw spread about outward, pitch down
            roll_jitter=0.75,                                 # how far off "normal at the camera" [rad]
            edge_on=0.70,                                     # reject if |approach . to_camera| above this
            tag_ahead=0.078,                                  # tag centre past the TCP [m], nominal
            candidates=16, tries=6,
            step=1.2, speed=0.8, dur=(1.2, 3.5))       # travel cap [rad], peak joint speed [rad/s]


def move_time(q0, q1):
    """How long to take over a move, so the position servos do not lag.

    The servos are pure position control with kv/kp = 0.05 s on every joint,
    so they trail the commanded ramp by 0.05 s of travel, and gravity sag adds
    a per-joint offset on top. A 2 rad swing in 1.5 s trails by 6 degrees,
    which is 5 cm at the fingertips and enough to put the card through a bottle
    the collision check had cleared. Bounding the peak joint speed keeps that
    error under a degree, and `WAVE['step']` keeps the moves short as well."""
    travel = float(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))))
    return float(np.clip(1.5 * travel / WAVE["speed"], *WAVE["dur"]))


def _frame(approach):
    """Two unit vectors spanning the plane the closing axis lives in."""
    e1 = np.cross([0.0, 0.0, 1.0], approach)
    n = np.linalg.norm(e1)
    e1 = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else e1 / n
    return e1, np.cross(approach, e1)


def wave_candidates(rng, base, table_top, cam_hint, n, roll_jitter=WAVE["roll_jitter"], cfg=None):
    """Candidate (position, rotation) TCP poses for the wave.

    Position and approach direction are drawn over the table. What the roll
    does depends on whether the camera has been located yet: before the first
    detection it sweeps the whole circle, so some pose eventually turns a face
    towards the lens; after it, the card's normal (the closing axis) is aimed
    at the camera, plus or minus a jitter that is the third rotation axis and
    plus an optional half turn that shows the other face."""
    cfg = cfg or WAVE
    out = []
    for _ in range(n * cfg["tries"]):
        p = np.array([rng.uniform(*cfg["x"]), rng.uniform(*cfg["y"]),
                      table_top + rng.uniform(*cfg["z"])])
        a = p[:2] - np.asarray(base)[:2]
        yaw = np.arctan2(a[1], a[0]) + rng.uniform(-cfg["yaw"], cfg["yaw"])
        pitch = rng.uniform(*cfg["pitch"])
        approach = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        e1, e2 = _frame(approach)
        if cam_hint is None:
            theta = rng.uniform(-np.pi, np.pi)
        else:
            d = np.asarray(cam_hint) - (p + cfg["tag_ahead"] * approach)
            d = d / np.linalg.norm(d)
            if abs(d @ approach) > cfg["edge_on"]:
                continue                       # camera down the approach: the card is edge on
            theta = (np.arctan2(d @ e2, d @ e1) + np.pi * rng.integers(2)
                     + rng.uniform(-roll_jitter, roll_jitter))
        close = np.cos(theta) * e1 + np.sin(theta) * e2
        out.append((p, rot(approach, close)))
        if len(out) >= n:
            break
    return out


def next_pose(motion, rng, q_prev, home, base, table_top, cam_hint, seen_R, diversify=True,
              roll_jitter=WAVE["roll_jitter"], cfg=None):
    """One collision-free joint configuration, chosen to add rotation diversity.

    Candidates are cheap and IK is not, so a batch is scored first by how much
    each would raise `rotation_spread_R` over the orientations already
    collected, and only then solved in that order. Collision checking is the
    ordinary planner one, so the card (which has a padded collider) and the arm
    stay off the table, the mount and whatever else is standing there."""
    cfg = cfg or WAVE
    cands = wave_candidates(rng, base, table_top, cam_hint, cfg["candidates"], roll_jitter, cfg)
    if diversify and seen_R:
        cands.sort(key=lambda c: -rotation_spread_R(seen_R + [c[1]]))
    for p, R in cands:
        q, why = motion.solve(q_prev, p, R, 0.0, home=home, q_from=q_prev)
        if not why and np.max(np.abs(q - np.asarray(q_prev, float))) <= cfg["step"]:
            return q
    return None


def escape(robot, q_prev, home, rng):
    """Get the arm out of a corner it can plan nothing from.

    With the card in the hand the gripper sweeps a lot of room, and from some
    poses every straight ramp to anywhere useful crosses the bottle. Back off
    to the home pose, which is where the wave started, or if even that path is
    blocked, take the first small random step that is clear."""
    mo = robot.motion
    if not np.allclose(q_prev, home) and mo.path_clear(q_prev, home, 0.0) is None:
        robot.move(home, move_time(q_prev, home))
        return np.asarray(home, float)
    for _ in range(40):
        q = np.clip(q_prev + rng.normal(0, 0.3, len(q_prev)), robot.arm.lo, robot.arm.hi)
        if mo.collides(q, 0.0) is None and mo.path_clear(q_prev, q, 0.0) is None:
            robot.move(q, move_time(q_prev, q))
            return q
    return np.asarray(q_prev, float)


def collect(robot, fk, card, home, base, table_top, min_obs=MIN_OBS, min_poses=MIN_POSES,
            max_poses=90, min_spread=MIN_SPREAD, seed=0, verbose=False, corner_noise=0.0,
            single_face=False, diversify=True, roll_jitter=WAVE["roll_jitter"], cfg=None,
            on_detect=None):
    """Wave until enough observations with enough rotation diversity are in hand.

    `card` is what the solver is allowed to know about the fiducial: the
    nominal mount pose in the gripper, the printed tag side, and each tag's
    exact pose in the card frame."""
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 7717)
    obs, seen_ids, seen_R, poses_with_tags = [], set(), [], 0
    q_prev, cam_hint, i = np.asarray(home, float), None, -1
    for i in range(max_poses):
        q = next_pose(robot.motion, rng, q_prev, home, base, table_top, cam_hint, seen_R,
                      diversify, roll_jitter, cfg)
        if q is None:
            q_prev = escape(robot, q_prev, home, rng)
            q = next_pose(robot.motion, rng, q_prev, home, base, table_top, cam_hint, seen_R,
                          diversify, roll_jitter, cfg)
        if q is None:
            continue
        robot.move(q, move_time(q_prev, q))
        q_prev = q
        img = robot.image()
        found = {t: c for t, c in detect(img).items() if t in card["tags"]
                 and not (single_face and seen_ids and t not in seen_ids)}
        T_g2b = fk.gripper2base(robot.q())
        for tid, corners in found.items():
            if corner_noise:
                corners = corners + nrng.normal(0, corner_noise, corners.shape).astype(corners.dtype)
            obs.append(Observation(T_g2b, card["tags"][tid], corners, tid, "card"))
            seen_ids.add(tid)
        if found:
            poses_with_tags += 1
            seen_R.append(T_g2b[:3, :3])
            if on_detect is not None:
                on_detect(img, found, i)
            if cam_hint is None:            # first sighting: where is the lens, roughly?
                tid, corners = next(iter(found.items()))
                T = T_g2b @ card["nominal"] @ card["tags"][tid] @ _inv(
                    tag_pose(corners, card["tag_size"], card["K"], card["dist"]))
                # FK is in the arm base frame, the wave is planned in world
                # coordinates, and the base frame is the world shifted to the
                # mount with no rotation.
                cam_hint = np.asarray(base, float) + T[:3, 3]
        spread = rotation_spread_R(seen_R)
        if verbose:
            print(f"  pose {i:2d}: tags {sorted(found)}  ({len(obs)} obs, spread {spread:.1f} deg)")
        if len(obs) >= min_obs and poses_with_tags >= min_poses and spread >= min_spread:
            break
    return obs, seen_ids, i + 1


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def card_from_info(info, K, dist=None):
    """What the solver may read about the card: nominal mount, printed tag side,
    tag geometry. The true mount in `info['calib']` is ground truth and is not
    copied here on purpose."""
    c = info["calib"]
    return dict(nominal=np.asarray(c["nominal"], float), tag_size=float(c["tag_size"]),
                tags={int(k): np.asarray(v, float) for k, v in c["tags"].items()},
                K=np.asarray(K, float), dist=dist)


def session(robot, card, home, base, table_top, fk=None, seed=0, max_poses=90, verbose=False,
            **kw):
    """Wave, fit, and judge. Returns (fit, obs, seen ids, poses visited).

    `fit.trusted` is false, with `fit.reason` saying why, when the residual is
    above `MAX_PX` or the wave never turned the card about enough axes; a
    degenerate wave is refused rather than returned, because its answer can be
    confidently wrong."""
    fk = fk or FK()
    obs, seen, n_poses = collect(robot, fk, card, home, base, table_top, seed=seed,
                                 max_poses=max_poses, verbose=verbose, **kw)
    if len(obs) < 6:
        raise RuntimeError(f"only {len(obs)} tag observations in {n_poses} poses; cannot calibrate")
    fit = solve(obs, card["K"], card["dist"], card["tag_size"], {"card": card["nominal"]},
                max_px=MAX_PX, seed=seed)
    fit.K, fit.dist = card["K"], card["dist"]
    reasons = []
    if fit.rms_px > MAX_PX:
        reasons.append(f"residual {fit.rms_px:.2f}px > {MAX_PX}px")
    if fit.spread_deg < MIN_SPREAD:
        reasons.append(f"rotation spread {fit.spread_deg:.1f}deg < {MIN_SPREAD}deg (degenerate wave)")
    if fit.n_obs < MIN_OBS:
        reasons.append(f"only {fit.n_obs} observations, wanted {MIN_OBS}")
    fit.trusted, fit.reason = not reasons, "; ".join(reasons)
    return fit, obs, seen, n_poses


def to_json(fit, K, dist, extra=None):
    d = dict(T_cam2base=fit.T_cam2base.tolist(), K=np.asarray(K).tolist(),
             dist=np.asarray(dist if dist is not None else np.zeros(5)).tolist(),
             rms_px=fit.rms_px, max_px=fit.max_px, n_obs=fit.n_obs,
             spread_deg=fit.spread_deg, trusted=bool(fit.trusted), reason=fit.reason,
             card_in_gripper={k: v.tolist() for k, v in fit.mounts.items()},
             time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    d.update(extra or {})
    return d


# ---------------------------------------------------------------- sim harness
def draw_detections(rgb, found):
    """The frame with every detected tag outlined and its corners marked."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for tid, c in found.items():
        c = np.asarray(c, float)
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
        for k, (x, y) in enumerate(c):
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
            cv2.putText(img, str(k), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"id {tid}", (int(c[:, 0].min()), int(c[:, 1].min()) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def run_sim(seed, max_poses, verbose, a):
    from pour_scene import build
    model, data, info = build(seed, calib_card=True, card_extreme=a.card_extreme)
    arm = Arm(model, "arm/")
    data.qpos[arm.fadr] = 0.0                    # fingers pinching the card
    data.ctrl[arm.grip] = 0.0
    mujoco.mj_forward(model, data)
    if a.empty_table:                            # the real table may be bare; nothing may depend on it
        for name in ("bottle", "glass"):
            try:
                adr = model.jnt_qposadr[model.body(name).jntadr[0]]
            except KeyError:
                continue
            data.qpos[adr:adr + 3] += [8.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
    robot = SimRobot(model, data, info)
    shot = {}

    def on_detect(img, found, i):
        if not a.frame_out:
            return
        area = max(cv2.contourArea(np.asarray(c, np.float32)) for c in found.values())
        if area > shot.get("area", 0.0):          # keep the clearest view of the card
            shot["area"], shot["img"] = area, draw_detections(img, found)

    try:
        cam = info["cam"]
        card = card_from_info(info, cam["K"], None)
        kw = dict(corner_noise=a.corner_noise, on_detect=on_detect)
        if a.one_face:
            kw["single_face"] = True
        if a.degenerate:
            # A deliberately unconditioned wave: one TCP pose, one approach
            # direction, and only the wrist roll about that one axis moving.
            # Every relative rotation then shares an axis and hand-eye has
            # nothing to fix the camera along it.
            kw.update(diversify=False, roll_jitter=0.6, min_spread=0.0,
                      cfg=dict(WAVE, x=(0.0, 0.0), y=(0.14, 0.14), z=(0.26, 0.26),
                               yaw=0.0, pitch=(0.7, 0.7)))
        fit, obs, seen, n = session(robot, card, info["q_home"], info["arm_base"],
                                    info["table_top"], seed=seed, max_poses=max_poses,
                                    verbose=verbose, **kw)
    finally:
        robot.close()
    if a.frame_out and "img" in shot:
        cv2.imwrite(a.frame_out, shot["img"])
    dt, dr = pose_error(cam["T_cam2base"], fit.T_cam2base)
    ct, cr = pose_error(info["calib"]["T_card2gripper"], fit.mounts["card"])
    return fit, dt, dr, ct, cr, seen, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="calibrate against pour_scene ground truth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="number of seeds (sim)")
    ap.add_argument("--poses", type=int, default=90, help="most poses to visit")
    ap.add_argument("--corner-noise", type=float, default=0.0,
                    help="sigma [px] of Gaussian noise added to detected corners")
    ap.add_argument("--card-extreme", action="store_true",
                    help="place the card at the limits of the misplacement, not inside them")
    ap.add_argument("--one-face", action="store_true",
                    help="use only the first face seen, as if the camera never saw the other")
    ap.add_argument("--empty-table", action="store_true", help="take the bottle and glass away")
    ap.add_argument("--degenerate", action="store_true",
                    help="cripple the wave's diversity, to check the conditioning gate refuses it")
    ap.add_argument("--frame-out", help="write the first frame with a detection, corners drawn")
    ap.add_argument("--out", help="write the session JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not a.sim:
        sys.exit("only --sim is implemented; the real-arm Robot needs the CAN driver")
    rows = []
    for i in range(a.n):
        seed = a.seed + i
        try:
            fit, dt, dr, ct, cr, seen, n_poses = run_sim(seed, a.poses, a.verbose, a)
        except RuntimeError as e:
            print(f"seed {seed:3d}  FAILED: {e}")
            rows.append(None)
            continue
        rows.append((dt, dr, fit.rms_px, fit.n_obs, ct, cr, fit.trusted))
        flag = "" if fit.trusted else f"  UNTRUSTED ({fit.reason})"
        print(f"seed {seed:3d}  poses={n_poses:2d} obs={fit.n_obs:3d} tags={sorted(seen)} "
              f"spread={fit.spread_deg:4.1f}deg rms={fit.rms_px:.2f}px max={fit.max_px:.2f}px "
              f"init={fit.init}  err={dt * 1000:.1f}mm {np.degrees(dr):.2f}deg  "
              f"card={ct * 1000:.1f}mm {np.degrees(cr):.2f}deg{flag}")
        if a.out and a.n == 1:
            with open(a.out, "w") as f:
                json.dump(to_json(fit, fit.K, fit.dist, dict(seed=seed)), f, indent=1)
            print("wrote", a.out)
    ok = [r for r in rows if r is not None and r[6]]
    if a.n > 1:
        print(f"\ntrusted: {len(ok)}/{a.n}")
        if ok:
            e = np.array([r[:6] for r in ok])
            for name, col, scale, unit in (("translation", 0, 1000, "mm"), ("rotation", 1, np.degrees(1), "deg"),
                                           ("residual", 2, 1, "px"), ("observations", 3, 1, ""),
                                           ("card translation", 4, 1000, "mm"),
                                           ("card rotation", 5, np.degrees(1), "deg")):
                v = e[:, col] * scale
                print(f"{name:>17}: median {np.median(v):7.2f}{unit}  worst {v.max():7.2f}{unit}"
                      f"  (min {v.min():.2f})")


if __name__ == "__main__":
    main()
