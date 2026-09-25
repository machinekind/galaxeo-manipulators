#!/usr/bin/env python3
"""Automated calibration on the REAL arm: the sim session's Robot, on CAN-FD.

    # 0. power on, bus up (outside this script):
    #    48V supply ON, arm switch ON, ./can_up.sh
    # 1. is the arm alive? (read-only, transmits nothing)
    python3 move_to_point_a1x.py --iface can0 --check
    # 2. camera intrinsics, once per webcam:
    python3 calib_intrinsics.py --camera 0 --out calib_session/K.npz
    # 3. put the calibration card in the gripper, pinched between the pads,
    #    protruding end out; put the webcam on the table looking at the arm;
    #    then the session:
    python3 calibrate_real.py --iface can0 --camera 0 --K calib_session/K.npz --dry-run
    python3 calibrate_real.py --iface can0 --camera 0 --K calib_session/K.npz --tx --out calib_session/session.json

Fills the slot `sim/calib/calibrate.py` leaves open ("only --sim is
implemented; the real-arm Robot needs the CAN driver"): the same session
code -- `collect` (the wave), `solve` (reprojection LM), the residual and
conditioning gates -- driven by `RealRobot` from `a1x_arm.py` instead
of the MuJoCo `SimRobot`. Forward kinematics for the fit comes from the URDF
at the MEASURED joint angles (`a1x_arm.GripperFK`), so servo error is
included, exactly as in sim.

What differs from sim, and why:

  * Intrinsics come from a one-shot checkerboard fit (`calib_intrinsics.py`)
    instead of MuJoCo; the session refuses to start without them.
  * Pose candidates are the sim's `wave_candidates` (pure numpy, unchanged),
    but collision checking is URDF-limit and joint-step checks instead of the
    MuJoCo planner -- the real table may be bare, and the wave box is over the
    empty table in front of the arm.
  * The wave is over the arm's own base frame (world == base with no
    rotation, like the sim's comment on the cam_hint), so `table_top` is 0
    and the TCP box hangs at the wave heights.
  * No bottle, no glass: the table may be bare, and nothing may depend on it
    (the sim's --empty-table philosophy, made the default).
  * DRY-RUN is the default: the wave runs against a STALE pose (no fresh
    feedback => detections are skipped) and transmits nothing, so you can
    watch the search aim before anything moves. LIVE needs --tx.
  * The gripper is never commanded. The card is a human step: pinch it in,
    run, take it out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sim"))

# `calib.calibrate` is imported lazily inside run_real(): it imports mujoco,
# a1x_control and planner.motion at module scope for its SimRobot, and the
# real-arm path never needs any of them. Everything the session shares -- WAVE,
# move_time, the gates, to_json, draw_detections, wave_candidates -- comes
# from it at call time, unchanged.
from calib.handeye import rotation_spread_R                       # noqa: E402  (sim)
from calib.tags import detect, tag_pose                           # noqa: E402  (sim)
from kinematics import Chain, ik                                  # noqa: E402
from a1x_arm import A1XArm, GripperFK, RealRobot, deg     # noqa: E402


def _inv(T):
    """4x4 inverse. Same construction as calib.calibrate._inv."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def rot(x_axis, y_axis):
    """Rotation matrix whose columns are the gripper's x (approach) and y
    (finger closing) axes. Inlined from `a1x_control.rot` so this module has
    no MuJoCo dependency: a1x_control imports mujoco at module scope, and the
    real-arm path never needs it."""
    x = np.asarray(x_axis, float); x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, float); y = y - x * (x @ y); y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])

ARM_BASE = np.zeros(3)          # the base frame IS the world here, no rotation
TABLE_TOP = 0.0                 # wave heights are TCP z above the mount, sim's WAVE
MAX_POSES = 90
DEFAULT_OUT_DIR = os.path.join(HERE, "calib_session")


def _sim():
    """Import `calib.calibrate` lazily: it imports mujoco, a1x_control and
    planner.motion at module scope for its SimRobot, and the real-arm path
    never needs any of them. Everything the session shares -- WAVE, move_time,
    the gates, to_json, draw_detections, wave_candidates -- comes from here
    unchanged."""
    import calib.calibrate
    return (calib.calibrate, calib.calibrate.MAX_PX, calib.calibrate.MIN_OBS,
            calib.calibrate.MIN_POSES, calib.calibrate.MIN_SPREAD,
            calib.calibrate.WAVE)


# --------------------------------------------------------- real-arm plumbing
def next_pose_real(ik_solve, rng, q_prev, home, cam_hint, seen_R, cfg, limits, diversify=True):
    """The sim's `next_pose`, with the MuJoCo planner swapped for a URDF IK.

    Candidates come from the sim's `wave_candidates` unchanged (position and
    approach over the table, the card's normal aimed at the camera hint once
    it exists). Each is scored by how much it would raise `rotation_spread_R`
    -- the conditioning metric -- and only then solved, best first. A candidate
    is refused if its IK misses (2 mm, 1 deg) or moves a joint more than
    `cfg['step']`; the sim's planner checked collisions too, which the bare
    real table cannot guarantee anyway.
    """
    ccal, MAX_PX, MIN_OBS, MIN_POSES, MIN_SPREAD, WAVE = _sim()
    cands = ccal.wave_candidates(rng, ARM_BASE, TABLE_TOP, cam_hint,
                                 cfg["candidates"], cfg.get("roll_jitter", WAVE["roll_jitter"]), cfg)
    if diversify and seen_R:
        cands.sort(key=lambda c: -rotation_spread_R(seen_R + [c[1]]))
    for p, R in cands:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        q, T_got = ik_solve(T, q_prev)
        e = T_got @ _inv(T)
        ep = np.linalg.norm(e[:3, 3])
        if ep < 5e-3 and np.max(np.abs(q - np.asarray(q_prev, float))) <= cfg["step"]:
            return q
    return None


def collect_real(robot, fk, card, home, min_obs=None, min_poses=None,
                 max_poses=MAX_POSES, min_spread=None, seed=0, verbose=False,
                 diversify=True, on_detect=None):
    """The sim's `collect` on the real arm: wave until enough observations.

    Same shape as the sim's: `next_pose` from a candidate fan, `escape` when
    IK can plan nothing, one detection gives the camera hint, and the loop
    stops at the same gates. Differences the hardware forces:

      * every pose waits for FRESH feedback before capturing, so T_frame2base
        is the pose the image was actually taken in;
      * `escape` falls back to small random steps (the home path may be
        blocked by the table edge in a way the URDF checks cannot see);
      * a pose the arm cannot reach is skipped, not retried.
    """
    _, MAX_PX, MIN_OBS, MIN_POSES, MIN_SPREAD, WAVE = _sim()
    min_obs = MIN_OBS if min_obs is None else min_obs
    min_poses = MIN_POSES if min_poses is None else min_poses
    min_spread = MIN_SPREAD if min_spread is None else min_spread
    move_time = _sim()[0].move_time
    rng = np.random.default_rng(seed)
    obs, seen_ids, seen_R, poses_with_tags = [], set(), [], 0
    q_prev, cam_hint, i = np.asarray(home, float), None, -1
    chain = robot.chain
    ik_solve = lambda T, q0: ik(chain, T, q0, iters=200, q_bias=q0)   # noqa: E731
    cfg = dict(WAVE)
    for i in range(max_poses):
        q = next_pose_real(ik_solve, rng, q_prev, home, cam_hint, seen_R, cfg,
                           chain.lower, diversify)
        if q is None:
            q_prev = _escape_real(robot, q_prev, home, rng, ik_solve)
            q = next_pose_real(ik_solve, rng, q_prev, home, cam_hint, seen_R, cfg,
                               chain.lower, diversify)
        if q is None:
            continue
        dur = move_time(q_prev, q)
        robot.move(q, dur)
        if robot.aborted:
            print(f"  ABORT during wave: {robot.aborted}; stopping the session")
            break
        if not robot.wait_fresh():
            print("  no fresh feedback after the move; skipping this pose")
            continue
        q_meas = robot.q()
        img = robot.image()
        found = {t: c for t, c in detect(img).items() if t in card["tags"]}
        T_g2b = fk.gripper2base(q_meas)
        for tid, corners in found.items():
            obs.append(_Observation(T_g2b, card["tags"][tid], corners, tid, "card"))
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
                cam_hint = np.asarray(ARM_BASE, float) + T[:3, 3]
        spread = rotation_spread_R(seen_R)
        if verbose:
            print(f"  pose {i:2d}: tags {sorted(found)}  ({len(obs)} obs, "
                  f"spread {spread:.1f} deg)", flush=True)
        if len(obs) >= min_obs and poses_with_tags >= min_poses and spread >= min_spread:
            break
    return obs, seen_ids, i + 1


def _escape_real(robot, q_prev, home, rng, ik_solve):
    """The sim's `escape`, with the home path checked against nothing but reachability.

    From some poses every IK solution near the wave box is unreachable; back
    off toward the home pose by joint slew, then take small random steps that
    IK can plan from."""
    ccal, _MAX_PX, _MIN_OBS, _MIN_POSES, _MIN_SPREAD, WAVE = _sim()
    move_time = ccal.move_time
    if not np.allclose(q_prev, home):
        robot.move(home, move_time(q_prev, home))
        if robot.aborted:
            return np.asarray(q_prev, float)
        return np.asarray(home, float)
    for _ in range(40):
        q = np.clip(q_prev + rng.normal(0, 0.3, len(q_prev)),
                    robot.chain.lower, robot.chain.upper)
        # a random joint step only needs to be reachable and small; solve FK
        # so the caller's move() can check the tip too (not a full IK solve)
        if np.max(np.abs(q - np.asarray(q_prev, float))) <= 2 * WAVE["step"]:
            robot.move(q, move_time(q_prev, q))
            if not robot.aborted:
                return q
    return np.asarray(q_prev, float)


def _Observation(T_frame2base, T_tag2mount, corners, tag_id, mount):
    """Import indirection so the module imports without calib at module scope."""
    from calib.handeye import Observation
    return Observation(T_frame2base, T_tag2mount, corners, tag_id, mount)


# ------------------------------------------------------------------ session
def run_real(a):
    """One real-arm session. Returns (fit, obs, seen, n_poses)."""
    from a1x_arm import Webcam
    ccal, MAX_PX, MIN_OBS, MIN_POSES, MIN_SPREAD, WAVE = _sim()

    K, dist = load_K(a.K, a.size)
    # The card the solver may know: the nominal mount (square to the hand,
    # sticking out past the tips), printed tag geometry, the measured K.
    card = dict(nominal=_card_nominal(), tag_size=a.size,
                tags=_CARD_TAGS(a.size), K=K, dist=dist)

    live = a.tx
    arm = A1XArm(a.iface, dry_run=not live, tx=live)
    cam = Webcam(a.camera, a.width, a.camera_height) if a.camera >= 0 else None
    robot = RealRobot(arm, cam, speed=a.speed, rate=a.rate,
                      track_tol=a.track_tol, verbose=a.verbose)
    shot = {}

    def on_detect(img, found, i):
        if a.frame_dir is None:
            return
        area = max(float(np.prod(np.ptp(np.asarray(c, float), axis=0))) for c in found.values())
        if area > shot.get("area", 0.0):
            shot["area"] = area
            shot["i"] = i
            shot["img"] = ccal.draw_detections(img, found)

    try:
        if live and a.enable:
            print("sending enable (FF 1 -> 6); setpoint stays at the measured pose")
            arm.send_enable()
        robot.wait_fresh(required=True)
        home = robot.q()
        print(f"home (measured, deg): {deg(home)}", flush=True)
        print(f"wave: {MAX_POSES} poses max, TCP box x{WAVE['x']} y{WAVE['y']} "
              f"z{WAVE['z']} above the mount, {a.speed:g} deg/s", flush=True)
        obs, seen, n_poses = collect_real(robot, GripperFK(), card, home,
                                          seed=a.seed, verbose=a.verbose,
                                          on_detect=on_detect,
                                          diversify=not a.degenerate)
    finally:
        arm.close()
        if cam is not None:
            cam.close()
    if a.frame_dir and "img" in shot:
        out = os.path.join(a.frame_dir, "best_frame.png")
        import cv2
        cv2.imwrite(out, shot["img"])
        print("wrote", out)
    if len(obs) < 6:
        raise RuntimeError(f"only {len(obs)} tag observations in {n_poses} poses; "
                           "cannot calibrate (is the card in? is the camera looking "
                           "at the arm?)")
    fit = _solve_session(obs, card, a.seed)
    return fit, obs, seen, n_poses


def _solve_session(obs, card, seed):
    """The sim's `session` gates, minus the sim harness plumbing."""
    _, MAX_PX, MIN_OBS, MIN_POSES, MIN_SPREAD, _WAVE = _sim()
    from calib.handeye import solve
    fit = solve(obs, card["K"], card["dist"], card["tag_size"], {"card": card["nominal"]},
                max_px=MAX_PX, seed=seed)
    fit.K, fit.dist = card["K"], card["dist"]
    reasons = []
    if fit.rms_px > MAX_PX:
        reasons.append(f"residual {fit.rms_px:.2f}px > {MAX_PX}px")
    if fit.spread_deg < MIN_SPREAD:
        reasons.append(f"rotation spread {fit.spread_deg:.1f}deg < {MIN_SPREAD}deg "
                       "(degenerate wave)")
    if fit.n_obs < MIN_OBS:
        reasons.append(f"only {fit.n_obs} observations, wanted {MIN_OBS}")
    fit.trusted, fit.reason = not reasons, "; ".join(reasons)
    return fit


# ------------------------------------------------------------- card geometry
def _card_nominal():
    """The nominal card pose in the gripper, from the sim's pour_scene CARD.

    Card frame: x along the approach, y across the card, z the normal. The
    nominal is square to the hand, sticking CARD['out'] = 90 mm past the
    fingertips, tag centre at half that beyond the tip plane (0.045 + 0.033 m
    in the gripper_link frame). The hand's actual placement is solved for;
    this is only the starting guess."""
    TIP_X = 0.045 + 0.033                    # gripper_link -> fingertip plane [m]
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    T[:3, 3] = np.array([TIP_X + 0.090 / 2, 0.0, 0.0])
    return T


def _CARD_TAGS(size):
    """tag id -> 4x4 pose in the card frame, exact printed geometry: the two
    faces back to back, the 2 mm thickness apart. Full transforms, not
    (centre, R) pairs -- `Observation.T_tag2mount` and the cam_hint chain
    multiply them directly, as the sim's info['calib']['tags'] does."""
    def T(centre, R):
        out = np.eye(4)
        out[:3, :3] = R
        out[:3, 3] = centre
        return out
    return {0: T(np.array([0.0, 0.0, 0.001]), np.eye(3)),
            1: T(np.array([0.0, 0.0, -0.001]), np.diag([-1.0, 1.0, -1.0]))}


def load_K(path, tag_size):
    """Intrinsics from a calib_intrinsics.py .npz (K, dist); refuses without."""
    if path is None:
        sys.exit("no --K: camera intrinsics are required. Run calib_intrinsics.py first "
                 "(once per webcam).")
    if not os.path.exists(path):
        sys.exit(f"--K {path} does not exist")
    d = np.load(path)
    K = np.asarray(d["K"], float)
    dist = np.asarray(d["dist"], float).ravel()
    return K, dist


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="can0")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV device index; "
                    "-1 runs the wave with no camera (not useful for a session)")
    ap.add_argument("--width", type=int, default=0, help="capture width (0 = driver default)")
    ap.add_argument("--camera-height", type=int, default=0, help="capture height")
    ap.add_argument("--K", help="intrinsics .npz from calib_intrinsics.py (K, dist)")
    ap.add_argument("--size", type=float, default=0.070,
                    help="AprilTag 36h11 side [m] as printed on the card")
    ap.add_argument("--dry-run", action="store_true",
                    help="the wave runs but transmits nothing (stale pose => no detections)")
    ap.add_argument("--tx", action="store_true", help="LIVE: the arm may MOVE")
    ap.add_argument("--enable", action="store_true",
                    help="send the enable sequence (FF 1 -> 6) first; only if the arm "
                         "reports but ignores 0x050")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poses", type=int, default=MAX_POSES, help="most poses to visit")
    ap.add_argument("--speed", type=float, default=8.0, help="peak joint slew, deg/s")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--track-tol", type=float, default=15.0,
                    help="abort if a joint lags its setpoint by this, deg")
    ap.add_argument("--frame-dir", help="write the clearest detection frame here")
    ap.add_argument("--out", help="write the session JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if not a.tx and not a.dry_run:
        a.dry_run = True              # DRY-RUN is the default: nothing moves
    live = a.tx

    try:
        fit, obs, seen, n_poses = run_real(a)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    flag = "" if fit.trusted else f"  UNTRUSTED ({fit.reason})"
    print(f"poses={n_poses} obs={fit.n_obs} tags={sorted(seen)} "
          f"spread={fit.spread_deg:.1f}deg rms={fit.rms_px:.2f}px "
          f"max={fit.max_px:.2f}px{flag}", flush=True)
    T = fit.T_cam2base
    print(f"T_cam2base (camera in the arm base frame):")
    print(np.round(T, 4))
    print(f"camera position [m]: {np.round(T[:3, 3], 3)}")
    mount = fit.mounts["card"]
    print(f"card in gripper (solved misplacement): "
          f"t {np.round(mount[:3, 3] - _card_nominal()[:3, 3], 4)} m")
    if a.out:
        out = a.out or os.path.join(DEFAULT_OUT_DIR, "session.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        d = to_json(fit, fit.K, fit.dist, dict(seed=a.seed, iface=a.iface,
                                               camera=a.camera, tag_size=a.size))
        with open(out, "w") as f:
            json.dump(d, f, indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
