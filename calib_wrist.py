#!/usr/bin/env python3
"""Where is the wrist camera on the gripper? Eye-in-hand calibration on the real A1X.

    # 0. arm on, CAN dongle in. The calibration card (two 70 mm AprilTag 36h11,
    #    ids 0 and 1, back to back) lies FLAT on the table in front of the arm.
    # 1. intrinsics of the wrist camera (it hangs upside down, hence --rotate):
    python3 calib_intrinsics.py --camera 0 --rotate --out calib_session/wrist_K.npz
    # 2. the session. Dry run first: plans the poses, grabs one frame where the
    #    arm stands, moves NOTHING. Then live.
    python3 calib_wrist.py session --K calib_session/wrist_K.npz --card 0.35 0.0 --dry-run
    python3 calib_wrist.py session --K calib_session/wrist_K.npz --card 0.35 0.0 --tx
    # 3. re-solve offline from the saved frames (a new K, a different gate):
    python3 calib_wrist.py solve calib_session/wrist --K calib_session/wrist_K.npz
    # 4. write the simulator's camera spec from the fit:
    python3 calib_wrist.py spec calib_session/wrist/fit.json --K calib_session/wrist_K.npz \\
        --out hardware/g1_camera_mounts/camera_spec_lashup.json

Mirror of `calibrate_real.py`. There the camera stands still and the card
rides on the gripper; here the card lies still and the camera rides on the
gripper. It is the same problem with the frames swapped, so the same
reprojection solver (`sim/calib/handeye.py`) fits both: hand it the INVERSE
gripper pose as `T_frame2base` and "the camera in the base" becomes the camera
in the gripper, and "the mount on the frame" becomes the card on the table.
Six unknowns each, fitted jointly from every corner of every tag, with the
residual in pixels deciding whether the session is trusted.

The arm goes through `galaxeo.arm.Arm` (reach box, IK, collision and jump
checks, guarded moves) on `galaxeo.bus` -- so on the Mac `--iface xcan`, on
Linux `can0`. Poses are drawn around the card: the tool approach axis aims at
the card from 15-32 cm, from 30-85 deg above the table, with the roll about
the approach swept over the whole circle so the rotations are diverse enough
to separate the camera's offset from the card's pose.

What comes out (`fit.json`):

    T_cam2gripper   4x4, OpenCV camera (z forward, x right, y down) in gripper_link
    T_card2base     4x4, where the card lay
    rms_px, spread_deg, n_obs, trusted, reason

`spec` turns that into the fields `hardware/g1_camera_mounts/camera_spec.json`
carries (lens entrance pupil, optical axis, image up, field of view), so
`sim/wrist_camera.attach_wrist_camera(..., design=load_spec(that_file))` puts
the simulated camera where the real one is.

Frames are 1920 x 1080 and are rotated 180 deg on capture, the way every other
consumer of this camera does it. The simulator renders 4:3; to match, crop the
real frame to the centre 1440 x 1080 before resizing -- the vertical field of
view is then the same and the horizontal one follows from it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sim"))

from calibrate_real import _CARD_TAGS, load_K          # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "calib_session", "wrist")
TCP_FROM_GRIPPER = 0.045          # tool site sits this far along x of gripper_link [m]
MAX_PX = 1.5                      # residual gate, as calib.calibrate.MAX_PX
MIN_OBS = 24
MIN_POSES = 12
MIN_SPREAD = 15.0                 # deg, rotation-axis spread gate, as calib.calibrate
POSE = dict(dist=(0.15, 0.32), elev=(0.55, 1.45), az=1.2,      # m, rad above table, +-rad
            aim_jitter=0.12, rot_tol=0.14, pos_tol=0.006,       # rad, rad, m
            candidates=24, tries=8)
# Where the camera roughly is, so the poses aim IT at the card and not the tool
# axis: the lash-up sits behind and above the housing, looking down the jaw gap
# at the fingertips, so its axis is the approach pitched down. In gripper_link
# coordinates (x approach, z up): offset [m] and optical axis. A first fit
# replaces this (`--cam-guess fit.json`).
CAM_GUESS = dict(pos=(-0.03, 0.0, 0.055), axis=(0.82, 0.0, -0.57))


# ------------------------------------------------------------------ helpers
def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def _rot(x_axis, y_axis):
    """Rotation whose columns are the tool's x (approach) and y (jaw closing) axes."""
    x = np.asarray(x_axis, float); x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, float); y = y - x * (x @ y); y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _frame(a):
    """Two unit vectors spanning the plane normal to `a`."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    h = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, h); e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _small_rotation(rotvec):
    th = np.linalg.norm(rotvec)
    if th < 1e-12:
        return np.eye(3)
    k = rotvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _rot_angle(R_a, R_b):
    c = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def gripper_from_tool(T_tool2base):
    """gripper_link pose from the tool-site pose: the site is 45 mm out along x."""
    off = np.eye(4); off[0, 3] = -TCP_FROM_GRIPPER
    return T_tool2base @ off


def card_nominal(xy, z=0.0):
    """The card flat on the table, tag 0 up, yaw unknown (solved for)."""
    T = np.eye(4)
    T[:3, 3] = [float(xy[0]), float(xy[1]), float(z)]
    return T


# ------------------------------------------------------------------- camera
class WristCam:
    """The upside-down UVC board camera: frames come out rotated 180 deg."""

    def __init__(self, index=0, width=0, height=0, rotate=True):
        import cv2
        self.cv2 = cv2
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"camera {index} did not open (macOS: grant the terminal Camera access)")
        self.rotate = rotate
        for _ in range(10):
            self.cap.read()
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def image(self, flush=4):
        """One fresh BGR frame (the driver queues a few stale ones)."""
        for _ in range(flush):
            self.cap.grab()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if self.rotate:
            f = self.cv2.rotate(f, self.cv2.ROTATE_180)
        return f

    def close(self):
        self.cap.release()


def _detect(bgr):
    """Tag corners in a BGR frame (the detector greys it; channel order is moot)."""
    from calib.tags import detect
    return detect(bgr[:, :, ::-1])


# ---------------------------------------------------------------- the arm
def open_arm(iface, live):
    """`galaxeo.arm.Arm` on the named bus; `sim` is a MuJoCo transport for tests."""
    from galaxeo.arm import Arm
    if iface == "sim":
        from galaxeo.arm.transport import SimTransport
        return Arm(SimTransport(), armed=live)
    from galaxeo.arm.transport import BusTransport
    from galaxeo.bus import open_bus
    return Arm(BusTransport(open_bus(iface), tx=live), armed=live)


def measured(arm, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        q = arm.measured()
        if q is not None:
            return q
        time.sleep(0.02)
    raise RuntimeError(f"no fresh joint feedback after {timeout:.0f} s: is the arm on, the bus up?")


def _align(v_from, v_to):
    """Minimal rotation taking unit vector v_from onto v_to."""
    a = np.asarray(v_from, float) / np.linalg.norm(v_from)
    b = np.asarray(v_to, float) / np.linalg.norm(v_to)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else _small_rotation(_frame(a)[0] * math.pi)
    return _small_rotation(v / s * math.atan2(s, c))


def candidates(rng, card_xyz, n, cfg=POSE, cam=CAM_GUESS):
    """(position, rotation) TOOL poses whose (guessed) camera looks at the card.

    The camera is placed on a shell around the card (distance, elevation,
    azimuth toward the base), its axis aimed at the card centre with a little
    jitter, then rolled about that axis over the whole circle. The tool pose
    follows from the guessed camera offset; it need not point at the card."""
    c = np.asarray(card_xyz, float)
    az0 = math.atan2(-c[1], -c[0])                # from the card back toward the base
    ax_g = np.asarray(cam["axis"], float) / np.linalg.norm(cam["axis"])
    off_g = np.asarray(cam["pos"], float)
    out = []
    for _ in range(n * cfg["tries"]):
        d = rng.uniform(*cfg["dist"])
        el = rng.uniform(*cfg["elev"])
        az = az0 + rng.uniform(-cfg["az"], cfg["az"])
        p_cam = c + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        look = (c - p_cam) / d
        look = _small_rotation(rng.normal(0, cfg["aim_jitter"], 3)) @ look
        th = rng.uniform(-math.pi, math.pi)
        R = _small_rotation(look * th) @ _align(ax_g, look)      # gripper rotation in base
        p = p_cam - R @ off_g                                     # gripper origin
        # gripper_link -> tool site is 45 mm along the same x
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p + R @ np.array([TCP_FROM_GRIPPER, 0.0, 0.0])
        out.append((T[:3, 3], R))
        if len(out) >= n:
            break
    return out


def next_pose(arm, rng, q_prev, card_xyz, seen_R, cfg=POSE, cam=CAM_GUESS):
    """The next planned joint vector, chosen to add rotation diversity, or None."""
    from calib.handeye import rotation_spread_R
    cands = candidates(rng, card_xyz, cfg["candidates"], cfg, cam)
    if seen_R:
        cands.sort(key=lambda c: -rotation_spread_R(seen_R + [c[1]]))
    for p, R in cands:
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        plan = arm.plan(T, start=q_prev)
        if not plan:
            continue
        got = arm.fk(plan.joints)
        if np.linalg.norm(got[:3, 3] - p) > cfg["pos_tol"] or _rot_angle(got[:3, :3], R) > cfg["rot_tol"]:
            continue
        return np.asarray(plan.joints, float), plan
    return None, None


# ------------------------------------------------------------------ session
def run_session(a):
    import cv2
    from calib.handeye import Observation, rotation_spread_R
    live = bool(a.tx)
    K, dist = load_K(a.K, a.size) if a.K else (None, None)
    tags = _CARD_TAGS(a.size)
    card_xyz = (a.card[0], a.card[1], a.table)
    out = a.out or DEFAULT_OUT
    os.makedirs(out, exist_ok=True)

    cam_guess = CAM_GUESS
    if a.cam_guess:
        with open(a.cam_guess) as f:
            T = np.asarray(json.load(f)["T_cam2gripper"], float)
        cam_guess = dict(pos=T[:3, 3].tolist(), axis=T[:3, 2].tolist())
    arm = open_arm(a.iface, live)
    cam = WristCam(a.camera, a.width, a.camera_height) if a.camera >= 0 else None
    rng = np.random.default_rng(a.seed)
    records, obs, seen_R, poses_with_tags = [], [], [], 0
    try:
        q = measured(arm)
        print(f"measured (deg): {np.round(np.degrees(q), 1).tolist()}")
        print(f"{'LIVE' if live else 'DRY RUN'}: card nominal at {card_xyz}, "
              f"{a.poses} poses max, {a.speed:g} deg/s")

        def capture(q_meas, i):
            nonlocal poses_with_tags
            if cam is None:
                return {}
            img = cam.image()
            found = {t: c for t, c in _detect(img).items() if t in tags}
            T_g2b = gripper_from_tool(arm.fk(q_meas))
            name = f"{i:03d}.png"
            cv2.imwrite(os.path.join(out, name), img)
            records.append(dict(frame=name, q=[float(x) for x in q_meas],
                                T_gripper2base=T_g2b.tolist(),
                                tags={str(t): np.asarray(c, float).tolist() for t, c in found.items()}))
            for tid, corners in found.items():
                obs.append(Observation(_inv(T_g2b), tags[tid], corners, tid, "card"))
            if found:
                poses_with_tags += 1
                seen_R.append(T_g2b[:3, :3])
            return found

        if not live:
            found = capture(q, 0)
            print(f"  standing pose: tags {sorted(found)} (dry run: one frame where the arm is)")

        q_prev = q.copy()
        for i in range(1, a.poses + 1):
            q_goal, plan = next_pose(arm, rng, q_prev, card_xyz, seen_R, cam=cam_guess)
            if q_goal is None:
                print(f"  pose {i:2d}: nothing plannable from here; stepping back toward the last good pose")
                q_goal = q_prev if len(records) else q
                if np.allclose(q_goal, q_prev):
                    continue
            if not live:
                print(f"  pose {i:2d}: would go to (deg) {np.round(np.degrees(q_goal), 1).tolist()}")
                q_prev = q_goal
                continue
            r = arm.move(q_goal, speed=a.speed)
            if r.state != "reached":
                print(f"  pose {i:2d}: move {r.state} ({r.reason}); "
                      f"{'stopping' if r.state in ('backoff', 'hold') else 'skipping'}")
                if r.state in ("backoff", "hold"):
                    break
                continue
            time.sleep(a.settle)
            q_meas = measured(arm)
            found = capture(q_meas, i)
            spread = rotation_spread_R(seen_R) if seen_R else 0.0
            print(f"  pose {i:2d}: tags {sorted(found)}  ({len(obs)} obs, "
                  f"{poses_with_tags} poses, spread {spread:.1f} deg)", flush=True)
            q_prev = q_meas
            if len(obs) >= a.min_obs and poses_with_tags >= MIN_POSES and spread >= MIN_SPREAD:
                break
        if live and a.home:
            print("returning to the start pose")
            arm.move(q, speed=a.speed)
    finally:
        try:
            arm.stop()
        except Exception:
            pass
        if cam is not None:
            cam.close()
        with open(os.path.join(out, "poses.json"), "w") as f:
            json.dump(dict(camera=a.camera, size=cam.size if cam else None, rotate=True,
                           tag_size=a.size, card_nominal=card_xyz, iface=a.iface,
                           live=live, records=records), f, indent=1)
        print("wrote", os.path.join(out, "poses.json"), f"({len(records)} frames)")

    if not live:
        return None
    return solve_dir(out, K, dist, a.size, card_xyz, seed=a.seed, max_px=a.max_px, min_obs=a.min_obs)


# -------------------------------------------------------------------- solve
def solve_dir(d, K, dist, tag_size, card_xyz=None, seed=0, max_px=MAX_PX, min_obs=MIN_OBS):
    """Re-detect every saved frame and fit. Writes fit.json next to them."""
    import cv2
    from calib.handeye import Observation, solve
    with open(os.path.join(d, "poses.json")) as f:
        meta = json.load(f)
    tags = _CARD_TAGS(tag_size)
    card_xyz = card_xyz or meta.get("card_nominal") or (0.35, 0.0, 0.0)
    obs, seen_R = [], []
    for r in meta["records"]:
        img = cv2.imread(os.path.join(d, r["frame"]))
        if img is None:
            continue
        found = {t: c for t, c in _detect(img).items() if t in tags}
        T_g2b = np.asarray(r["T_gripper2base"], float)
        for tid, corners in found.items():
            obs.append(Observation(_inv(T_g2b), tags[tid], corners, tid, "card"))
        if found:
            seen_R.append(T_g2b[:3, :3])
    if len(obs) < 6:
        raise SystemExit(f"only {len(obs)} tag observations in {len(meta['records'])} frames; "
                         "is the card in view? (tags 0/1, 36h11)")
    fit = solve(obs, K, dist, tag_size, {"card": card_nominal(card_xyz[:2], card_xyz[2])},
                max_px=max_px, seed=seed)
    reasons = []
    if fit.rms_px > max_px:
        reasons.append(f"residual {fit.rms_px:.2f}px > {max_px}px")
    if fit.spread_deg < MIN_SPREAD:
        reasons.append(f"rotation spread {fit.spread_deg:.1f}deg < {MIN_SPREAD}deg")
    if fit.n_obs < min_obs:
        reasons.append(f"only {fit.n_obs} observations, wanted {min_obs}")
    T_c2g = np.asarray(fit.T_cam2base, float)          # "base" was the gripper here
    T_card = np.asarray(fit.mounts["card"], float)
    result = dict(T_cam2gripper=T_c2g.tolist(), T_card2base=T_card.tolist(),
                  rms_px=float(fit.rms_px), spread_deg=float(fit.spread_deg),
                  n_obs=int(fit.n_obs), trusted=not reasons, reason="; ".join(reasons),
                  K=np.asarray(K).tolist(), dist=np.asarray(dist).ravel().tolist(),
                  tag_size=tag_size, frames=len(meta["records"]), init=getattr(fit, "init", ""))
    with open(os.path.join(d, "fit.json"), "w") as f:
        json.dump(result, f, indent=1)
    report(result)
    print("wrote", os.path.join(d, "fit.json"))
    return result


def report(res):
    T = np.asarray(res["T_cam2gripper"], float)
    t, R = T[:3, 3] * 1000, T[:3, :3]
    axis = R[:, 2]
    pitch = math.degrees(math.asin(-axis[2]))              # below the tool axis is +x, z down
    yaw = math.degrees(math.atan2(axis[1], axis[0]))
    print(f"camera in gripper_link: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm, "
          f"optical axis {np.round(axis, 3).tolist()} (yaw {yaw:.1f} deg, pitch {pitch:.1f} deg)")
    print(f"residual {res['rms_px']:.2f} px, rotation spread {res['spread_deg']:.1f} deg, "
          f"{res['n_obs']} observations -> {'TRUSTED' if res['trusted'] else 'NOT trusted: ' + res['reason']}")


# ---------------------------------------------------------------------- spec
def write_spec(fit_path, K_path, out, size_px=(1920, 1080)):
    """camera_spec.json fields for the simulator, from a fit and the intrinsics."""
    with open(fit_path) as f:
        res = json.load(f)
    K, _dist = load_K(K_path, 0.0)
    d = np.load(K_path)
    W = int(d["width"]) if "width" in d else size_px[0]
    H = int(d["height"]) if "height" in d else size_px[1]
    fovx = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
    fovy = math.degrees(2 * math.atan(H / (2 * K[1, 1])))
    fovx_43 = math.degrees(2 * math.atan((4 / 3) * math.tan(math.radians(fovy / 2))))
    T = np.asarray(res["T_cam2gripper"], float)
    R = T[:3, :3]
    fwd, right, up = R[:, 2], R[:, 0], -R[:, 1]
    spec = {
        "frame": "gripper_link",
        "units": "metres",
        "handedness": "right",
        "source": f"calib_wrist.py fit {os.path.relpath(fit_path, HERE)}: residual "
                  f"{res['rms_px']:.2f} px over {res['n_obs']} tag views, spread {res['spread_deg']:.1f} deg"
                  + ("" if res["trusted"] else f" -- NOT trusted: {res['reason']}"),
        "lens_entrance_pupil_m": T[:3, 3].tolist(),
        "optical_axis": fwd.tolist(),
        "image_up": up.tolist(),
        "image_right": right.tolist(),
        "T_gripper_camera_opencv": T.tolist(),
        "pitch_below_tool_axis_deg": math.degrees(math.asin(-fwd[2])),
        "yaw_inboard_deg": math.degrees(math.atan2(fwd[1], fwd[0])),
        "roll_deg": 0.0,
        "lens": {
            "recommended": {
                "name": "as fitted (M12 wide on the 32 mm UVC board)",
                "fov_deg": [round(fovx_43, 2), round(fovy, 2)],
                "fov_note": f"vertical angle from the calibrated K on {W}x{H}; the horizontal one is "
                            f"what a 4:3 render gives for it. The real frame is {fovx:.1f} deg wide: "
                            f"crop it to the centre 4:3 ({int(round(H * 4 / 3))}x{H}) before resizing "
                            "and the two match.",
            }
        },
        "resolution_px": [640, 480],
        "intrinsics_full_frame": {"K": K.tolist(), "width": W, "height": H,
                                  "fov_deg": [round(fovx, 2), round(fovy, 2)]},
        "sensor_module": {"kind": "32 x 32 mm UVC board camera, wide M12, upside down (frames rotated 180)",
                          "mass_g": 22.0},
        "notes": "Measured pose of the high-rear lash-up fitted 2026-09-25 (strut behind the housing, "
                 "lens aimed down the jaw gap). Not the PR #2 outboard strut: payload.json's mass and "
                 "collision primitives do not describe this mount.",
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"fov {fovx:.1f} x {fovy:.1f} deg on {W}x{H} ({fovx_43:.1f} x {fovy:.1f} on 4:3)")
    print("wrote", out)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("session", help="drive the arm around the card and fit")
    s.add_argument("--iface", default="xcan" if platform.system() == "Darwin" else "can0",
                   help="xcan (macOS), can0 (Linux), sim (MuJoCo transport, for tests)")
    s.add_argument("--camera", type=int, default=0, help="OpenCV device index of the wrist camera; -1 none")
    s.add_argument("--width", type=int, default=0)
    s.add_argument("--camera-height", type=int, default=0)
    s.add_argument("--K", help="intrinsics .npz from calib_intrinsics.py --rotate")
    s.add_argument("--size", type=float, default=0.070, help="AprilTag side [m] on the card")
    s.add_argument("--card", type=float, nargs=2, default=(0.35, 0.0), metavar=("X", "Y"),
                   help="rough card centre in the base frame [m]")
    s.add_argument("--table", type=float, default=0.0, help="table top z in the base frame [m]")
    s.add_argument("--cam-guess", metavar="FIT_JSON",
                   help="aim the poses with the camera pose from an earlier fit.json instead of the built-in guess")
    s.add_argument("--poses", type=int, default=40)
    s.add_argument("--speed", type=float, default=10.0, help="peak joint speed, deg/s")
    s.add_argument("--settle", type=float, default=0.7, help="seconds to wait after a move before the frame")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-obs", type=int, default=MIN_OBS)
    s.add_argument("--max-px", type=float, default=MAX_PX)
    s.add_argument("--dry-run", action="store_true", help="plan and print; move nothing (default)")
    s.add_argument("--tx", action="store_true", help="LIVE: the arm MOVES")
    s.add_argument("--home", action="store_true", help="return to the start pose at the end")
    s.add_argument("--out", help=f"session directory (default {os.path.relpath(DEFAULT_OUT, HERE)})")

    v = sub.add_parser("solve", help="re-fit a saved session")
    v.add_argument("dir")
    v.add_argument("--K", required=True)
    v.add_argument("--size", type=float, default=0.070)
    v.add_argument("--card", type=float, nargs=3, metavar=("X", "Y", "Z"))
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--max-px", type=float, default=MAX_PX)
    v.add_argument("--min-obs", type=int, default=MIN_OBS)

    w = sub.add_parser("spec", help="write the simulator camera spec from a fit")
    w.add_argument("fit")
    w.add_argument("--K", required=True)
    w.add_argument("--out", default=os.path.join(HERE, "hardware", "g1_camera_mounts", "camera_spec_lashup.json"))

    a = ap.parse_args()
    if a.cmd == "session":
        if a.tx and a.dry_run:
            sys.exit("--tx and --dry-run are exclusive")
        if a.K is None and a.tx:
            sys.exit("no --K: run calib_intrinsics.py --camera 0 --rotate first")
        run_session(a)
    elif a.cmd == "solve":
        K, dist = load_K(a.K, a.size)
        solve_dir(a.dir, K, dist, a.size, a.card, seed=a.seed, max_px=a.max_px, min_obs=a.min_obs)
    else:
        write_spec(a.fit, a.K, a.out)


if __name__ == "__main__":
    main()
