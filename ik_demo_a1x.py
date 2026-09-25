#!/usr/bin/env python3
"""IK demo on the real A1X: home -> unfold -> a small box traced by the tip -> home.

    python ik_demo_a1x.py --dry-run                       # read the pose, print the plan, transmit nothing
    python ik_demo_a1x.py                                 # live: the pose the arm is in now is home
    python ik_demo_a1x.py --plan --home 92.6,0,0,0,0,0    # no bus: print the plan for a given home

Home is the pose measured over the bus when the script starts (--home here,
the default), so the arm comes back to exactly where it was. Pass --home with
six angles instead to demand a specific pose: the script then refuses unless
the measured pose is within --start-tol of it.

The plan, all solved before a single frame is sent:

    1. unfold    joint slew from the rest pose to a "ready" pose: J2/J3/J4 set
                 by --ready (default = the sim's home keyframe), J1/J5/J6 kept
    2. up        tip +--dz  (IK, kinematics.py on the URDF, orientation held)
    3. forward   tip +--dx along the arm's own plane (IK)
    4. down      tip -dz (IK)
    5. back      tip -dx: the ready tip pose again (IK)
    6. ready     the exact ready joints (removes any IK residual)
    7. fold      joint slew back to the pose measured at start

Every IK solve must converge (tip error < 2 mm, 1 deg), stay inside the URDF
limits, and no leg may move a joint more than --max-leg; otherwise the plan is
rejected and nothing is sent. Legs run as joint-space slews at --speed deg/s,
so the tip path between IK waypoints is not a straight line, only its ends are.

Safety, same rules as jog_a1x.py (see docs/SAFETY.md):

  * p_des is seeded from the MEASURED pose and only ever moves at --speed.
  * Streams at --rate while running. Feedback older than 150 ms aborts.
  * Any joint more than --track-tol behind its setpoint aborts.
  * Ctrl-C stops the stream; an uncommanded A1X re-latches where it is.
  * The gripper is never commanded.
  * Targets are clamped to the URDF limits, widened to include the measured
    start pose (J3 reads ~1.5 deg past its limit at rest).
"""
import argparse
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jog_a1x import A1X, LIMITS, N, NAMES                    # noqa: E402
from kinematics import Chain, ik, pose_error                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "ros2_ws/src/galaxea_a1xy_description/urdf/a1x.urdf")
JOINTS = [f"arm_joint{i}" for i in range(1, 7)]


def deg(q):
    return ", ".join(f"{math.degrees(x):6.1f}" for x in q)


def build_plan(home, ready_j234, dx, dz, max_leg_deg):
    """[(label, q_goal)] from `home` and back. Raises SystemExit on a bad plan."""
    chain = Chain(URDF, JOINTS)
    home = np.asarray(home, float)
    ready = home.copy(); ready[1:4] = ready_j234
    T0 = chain.fk(ready)
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    fwd = np.array([math.cos(home[0]), math.sin(home[0]), 0.0])   # the arm's plane
    up = np.array([0.0, 0.0, 1.0])
    targets = [("up", p0 + dz * up), ("forward", p0 + dz * up + dx * fwd),
               ("down", p0 + dx * fwd), ("back", p0)]

    legs = [("unfold", ready)]
    q = ready.copy()
    for label, p in targets:
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = p
        q, T_got = ik(chain, T, q, iters=200, q_bias=ready)
        e = pose_error(T_got, T)
        ep, ew = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if ep > 2e-3 or ew > math.radians(1.0):
            raise SystemExit(f"plan rejected: IK for '{label}' did not converge "
                             f"(tip error {ep * 1e3:.1f} mm, {math.degrees(ew):.1f} deg)")
        legs.append((label, q.copy()))
    legs += [("ready", ready), ("fold", home)]

    prev = home
    for label, qg in legs:
        for j in range(N):
            lo, hi = LIMITS[j]
            if not (lo - 1e-6 <= qg[j] <= hi + 1e-6) and label != "fold":
                raise SystemExit(f"plan rejected: '{label}' puts {NAMES[j]} at "
                                 f"{math.degrees(qg[j]):.1f} deg, outside [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]")
        step = np.max(np.abs(qg - prev))
        if label not in ("unfold", "fold") and step > math.radians(max_leg_deg):
            raise SystemExit(f"plan rejected: '{label}' moves a joint {math.degrees(step):.1f} deg "
                             f"in one leg (limit --max-leg {max_leg_deg:g})")
        prev = qg
    return chain, legs


def print_plan(chain, home, legs, speed):
    print(f"{'leg':<8} {'J1':>7} {'J2':>7} {'J3':>7} {'J4':>7} {'J5':>7} {'J6':>7}   {'tip xyz [m]':<24} {'max dq':>7} {'time':>6}")
    print(f"{'start':<8} {deg(home)}   {np.round(chain.fk(home)[:3, 3], 3)}")
    prev, total = np.asarray(home, float), 0.0
    for label, q in legs:
        d = math.degrees(np.max(np.abs(q - prev))); t = d / speed; total += t
        print(f"{label:<8} {deg(q)}   {str(np.round(chain.fk(q)[:3, 3], 3)):<24} {d:6.1f}° {t:5.1f}s")
        prev = q
    print(f"total {total:.0f} s at {speed:g} deg/s, plus dwells")


def read_pose(arm, secs=2.0):
    """Listen for feedback; the measured joint angles, or None with a reason printed."""
    print(f"listening for feedback on {arm.bus.channel_info} ...")
    t0 = time.time()
    while time.time() - t0 < secs:
        arm.drain(); time.sleep(0.005)
    if arm.q is None or time.time() - arm.t > 0.15:
        print("no fresh 0x052 feedback: arm off, or bus down?"); return None
    if arm.same > 50:
        print("telemetry FROZEN (released state, FF 2): not starting"); return None
    return np.array(arm.q[:N], float)


def run(arm, a, legs):
    """Stream the plan. Returns an exit code; never raises past Ctrl-C."""
    period = 1.0 / a.rate
    q_start = read_pose(arm)
    if q_start is None:
        return 1
    home = legs[-1][1]
    off = np.abs(q_start - home)
    print(f"measured : {deg(q_start)}")
    print(f"--home   : {deg(home)}")
    if np.max(off) > math.radians(a.start_tol):
        j = int(np.argmax(off))
        print(f"REFUSING: {NAMES[j]} is {math.degrees(off[j]):.1f} deg from --home "
              f"(--start-tol {a.start_tol:g}). Jog the arm home first, or pass\n"
              f"    --home {','.join(f'{math.degrees(x):.1f}' for x in q_start)}")
        return 1
    # The final leg returns to where the arm actually was, not the nominal
    # --home: commanding a joint into its mechanical stop pushes there forever.
    legs = legs[:-1] + [("fold", q_start.copy())]
    bounds = [(min(lo, q_start[j]), max(hi, q_start[j])) for j, (lo, hi) in enumerate(LIMITS)]
    target = q_start.copy()
    step_max = math.radians(a.speed)
    mode = "DRY-RUN, transmitting nothing" if arm.dry_run else "LIVE"
    print(f"{mode}: streaming at {a.rate:g} Hz, {a.speed:g} deg/s, Ctrl-C stops the stream")

    def send():
        arm.send_arm(list(target), a.kp, a.kd)

    try:
        for label, goal in legs:
            goal = np.array([min(hi, max(lo, goal[j])) for j, (lo, hi) in enumerate(bounds)])
            nxt = time.time(); prev = nxt; rep = 0.0; done_at = None
            while True:
                arm.drain()
                now = time.time()
                if now < nxt:
                    time.sleep(min(0.001, nxt - now)); continue
                nxt += period
                if nxt < now - 0.05:
                    nxt = now
                dt = min(0.05, now - prev); prev = now
                if now - arm.t > 0.15:
                    print(f"\nABORT: stale feedback ({(now - arm.t) * 1e3:.0f} ms). Stream stopped; arm holds.")
                    return 2
                meas = np.array(arm.q[:N], float)
                lag = np.abs(meas - target)
                if not arm.dry_run and np.max(lag) > math.radians(a.track_tol):
                    j = int(np.argmax(lag))
                    print(f"\nABORT: {NAMES[j]} is {math.degrees(lag[j]):.1f} deg from its setpoint "
                          f"(--track-tol {a.track_tol:g}). Stream stopped; arm holds.")
                    return 2
                d = goal - target
                target += np.clip(d, -step_max * dt, step_max * dt)
                send()
                worst = math.degrees(np.max(np.abs(goal - target)))
                if now - rep >= 0.5:
                    rep = now
                    print(f"\r{label:<8} {worst:6.1f} deg to go   target {deg(target)}   "
                          f"lag {math.degrees(np.max(lag)):4.1f} deg   ", end="", flush=True)
                if worst < 0.05:
                    if done_at is None:
                        done_at = now
                    elif now - done_at >= a.dwell:
                        break
            print(f"\r{label:<8} reached.{' ' * 90}")
    except KeyboardInterrupt:
        print("\nCtrl-C: stream stopped; the arm re-latches where it is.")
        return 130
    print("done: back at the start pose, stream stopped. The arm holds there.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="xcan" if platform.system() == "Darwin" else "can0")
    ap.add_argument("--plan", action="store_true", help="print the joint plan and exit, no bus")
    ap.add_argument("--dry-run", action="store_true", help="read feedback, run the loop, transmit nothing")
    ap.add_argument("--home", default="here",
                    help="'here' (default): the pose measured at start. Or six joint angles in "
                         "degrees, e.g. what jog_a1x.py 'Set home = here' printed")
    ap.add_argument("--ready", default="57.3,-91.7,34.4",
                    help="J2,J3,J4 of the unfolded ready pose, degrees (default = the sim home keyframe)")
    ap.add_argument("--dx", type=float, default=0.05, help="box length along the arm's plane, m")
    ap.add_argument("--dz", type=float, default=0.05, help="box height, m")
    ap.add_argument("--speed", type=float, default=8.0, help="joint slew, deg/s, every leg")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--dwell", type=float, default=0.5, help="pause at each waypoint, s")
    ap.add_argument("--start-tol", type=float, default=5.0, help="refuse unless measured is this close to --home, deg")
    ap.add_argument("--track-tol", type=float, default=15.0, help="abort if a joint lags its setpoint by this, deg")
    ap.add_argument("--max-leg", type=float, default=30.0, help="reject an IK leg that moves a joint more than this, deg")
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=1.0)
    a = ap.parse_args()
    try:
        home = None if a.home == "here" else [math.radians(float(x)) for x in a.home.split(",")]
        assert home is None or len(home) == N
        ready = [math.radians(float(x)) for x in a.ready.split(",")]; assert len(ready) == 3
    except (ValueError, AssertionError):
        sys.exit("--home needs 'here' or six angles, --ready three comma-separated angles, in degrees")
    if a.plan and home is None:
        sys.exit("--plan does not open the bus, so it needs an explicit --home")

    arm = None
    if not a.plan:
        try:
            arm = A1X(a.iface, dry_run=a.dry_run)
        except Exception as ex:
            sys.exit(f"cannot open {a.iface}: {ex}")
    try:
        if home is None:
            home = read_pose(arm)
            if home is None:
                return 1
            print(f"home = the pose measured now: {deg(home)}")
        chain, legs = build_plan(home, ready, a.dx, a.dz, a.max_leg)
        print_plan(chain, home, legs, a.speed)
        if a.plan:
            return 0
        return run(arm, a, legs)
    finally:
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
