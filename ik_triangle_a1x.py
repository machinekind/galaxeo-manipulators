#!/usr/bin/env python3
"""Draw a closed triangle in the air with the real A1X, every vertex reached by IK.

    python ik_triangle_a1x.py --dry-run      # read the pose, print the plan, transmit nothing
    python ik_triangle_a1x.py                # live, Ctrl-C stops the stream

Three points A, B, C. The path is  start -> A -> B -> C -> A -> start.
Every vertex is an XYZ target solved to joint angles by kinematics.py on the
URDF, with the gripper held horizontal, pointing forward. The only leg that is
not IK is the last one: the arm returns to the joint angles measured at start,
so it ends exactly where it began and never pushes into a mechanical stop.

Points are given in the arm's frame, metres:  x forward along the arm's own
plane (wherever J1 points now), y to the left, z up, origin at the base.
Default:  A 0.10,0,0.45   B 0.20,0,0.45   C 0.15,0,0.53   (about 10 cm sides).

    --points "0.10,0,0.45;0.20,0,0.45;0.15,0,0.53"

Home is the pose measured over the bus when the script starts. The plan is
solved and printed before anything is sent; every solve must converge (2 mm,
1 deg) and stay inside the URDF limits, and no leg between vertices may move
a joint more than --max-leg. Legs are joint slews at --speed deg/s, so the
tip path between vertices is not a straight line, only its ends are.

Safety is jog_a1x.py's: setpoint seeded from the measured pose, moves only at
--speed, streams at --rate, aborts on stale feedback or a joint lagging by
--track-tol, Ctrl-C stops the stream and the arm holds. Gripper untouched.
"""
import argparse
import math
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ik_demo_a1x import JOINTS, URDF, deg, print_plan, read_pose, run      # noqa: E402
from jog_a1x import A1X, LIMITS, N, NAMES                                  # noqa: E402
from kinematics import Chain, ik, pose_error                               # noqa: E402

READY_J234 = [1.0, -1.6, 0.6]     # the sim's home keyframe: IK seed and null-space bias


def build_plan(home, points, max_leg_deg):
    """[(label, q)] : A, B, C, A by IK, then the measured start pose."""
    chain = Chain(URDF, JOINTS)
    home = np.asarray(home, float)
    seed = home.copy(); seed[1:4] = READY_J234
    R = chain.fk(seed)[:3, :3]                                # gripper horizontal, forward
    c, s = math.cos(home[0]), math.sin(home[0])
    to_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    legs = []
    q = seed.copy()
    for label, p in zip("ABCA", list(points) + [points[0]]):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = to_base @ np.asarray(p, float)
        q, T_got = ik(chain, T, q, iters=300, q_bias=seed)
        e = pose_error(T_got, T)
        ep, ew = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if ep > 2e-3 or ew > math.radians(1.0):
            raise SystemExit(f"plan rejected: IK for {label} {p} did not converge "
                             f"(tip error {ep * 1e3:.1f} mm, {math.degrees(ew):.1f} deg): out of reach")
        for j in range(N):
            lo, hi = LIMITS[j]
            if not (lo - 1e-6 <= q[j] <= hi + 1e-6):
                raise SystemExit(f"plan rejected: {label} puts {NAMES[j]} at {math.degrees(q[j]):.1f} deg, "
                                 f"outside [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]")
        legs.append((label, q.copy()))
    for (l0, q0), (l1, q1) in zip(legs, legs[1:]):
        step = math.degrees(np.max(np.abs(q1 - q0)))
        if step > max_leg_deg:
            raise SystemExit(f"plan rejected: {l0} -> {l1} moves a joint {step:.1f} deg "
                             f"(limit --max-leg {max_leg_deg:g}); bring the points closer")
    legs.append(("start", home))
    return chain, legs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="xcan" if platform.system() == "Darwin" else "can0")
    ap.add_argument("--plan", action="store_true", help="print the plan for --home and exit, no bus")
    ap.add_argument("--dry-run", action="store_true", help="read feedback, run the loop, transmit nothing")
    ap.add_argument("--home", default="here", help="'here' (default) or six joint angles in degrees")
    ap.add_argument("--points", default="0.10,0,0.45;0.20,0,0.45;0.15,0,0.53",
                    help="A;B;C as x,y,z in metres, arm frame (x forward, y left, z up, origin at base)")
    ap.add_argument("--speed", type=float, default=8.0, help="joint slew, deg/s")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--dwell", type=float, default=0.5, help="pause at each vertex, s")
    ap.add_argument("--start-tol", type=float, default=5.0, help="with an explicit --home: refuse unless this close, deg")
    ap.add_argument("--track-tol", type=float, default=15.0, help="abort if a joint lags its setpoint by this, deg")
    ap.add_argument("--max-leg", type=float, default=35.0, help="reject a vertex-to-vertex leg moving a joint more than this, deg")
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=1.0)
    a = ap.parse_args()
    try:
        home = None if a.home == "here" else [math.radians(float(x)) for x in a.home.split(",")]
        assert home is None or len(home) == N
        points = [[float(x) for x in p.split(",")] for p in a.points.split(";")]
        assert len(points) == 3 and all(len(p) == 3 for p in points)
    except (ValueError, AssertionError):
        sys.exit("--home needs 'here' or six angles in degrees; --points needs three x,y,z separated by ';'")
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
        chain, legs = build_plan(home, points, a.max_leg)
        print_plan(chain, home, legs, a.speed)
        if a.plan:
            return 0
        return run(arm, a, legs)
    finally:
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
