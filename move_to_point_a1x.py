#!/usr/bin/env python3
"""Move to point with the real A1X over CAN-FD, direct Python.

    python3 move_to_point_a1x.py --iface can0 --check               # read-only: is the arm streaming?
    python3 move_to_point_a1x.py --iface can0 --dry-run --dx 0.03   # the move loop, transmits nothing
    python3 move_to_point_a1x.py --iface can0 --tx --dx 0.05        # LIVE: move to point, +x 5 cm

The control algorithm is `RealRobot.move` in `a1x_arm.py`: read the 200 Hz
joint feedback on 0x052, solve the URDF IK (`kinematics.py`) for the desired
tip pose, then stream p_des on 0x050 along a rate-limited joint ramp while
watching for stale feedback and lagging joints. This script is that algorithm
as a standalone CLI experiment, the way `ik_demo_a1x.py` and `jog_a1x.py` are;
`a1x_arm.py` stays the library, and `calibrate_real.py` drives the same
`RealRobot` through the calibration wave of `sim/calib/calibrate.py`.

The goal is +dx/+dz in the arm base frame from the pose measured at start,
orientation held; the plan is rejected unless IK converges (2 mm / 1 deg),
so nothing is transmitted on a hopeless target.

Safety, same rules as jog_a1x.py / ik_demo_a1x.py (see docs/SAFETY.md):

  * Transmit is OFF unless --tx is passed. The first live frame is seeded from
    the MEASURED pose, so nothing jumps.
  * p_des only ever moves at --speed (deg/s), streamed at --rate.
  * Feedback older than 150 ms aborts; a joint more than --track-tol behind its
    setpoint aborts. The stream stops and the arm holds where it is.
  * The gripper is never commanded. Targets are clamped to the URDF limits.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ros2_ws", "src",
                                "galaxea_a1xy_driver", "galaxea_a1xy_driver"))

from a1x_arm import A1XArm, GripperFK, RealRobot, Webcam, deg     # noqa: E402
from kinematics import ik, pose_error                             # noqa: E402
from protocol import N_JOINTS                                     # noqa: E402


def read_pose(arm, secs):
    q, why = None, ""
    t0 = time.time()
    while time.time() - t0 < secs:
        arm.drain()
        time.sleep(0.005)
        q, why = arm.fresh()
        if q is not None:
            # keep listening to report a rate too
            pass
    if q is None:
        print(f"NO DATA: {why}")
        return None, 0.0
    with arm.lock:
        hz = arm.n / secs
    return q, hz


def run_check(a):
    arm = A1XArm(a.iface, dry_run=True)
    try:
        q, hz = read_pose(arm, a.secs)
    finally:
        arm.close()
    if q is None:
        return 1
    print(f"arm on {a.iface}: {hz:.0f} Hz")
    print(f"measured (deg): {deg(q)}")
    fk = GripperFK()
    T = fk.gripper2base(q)
    print(f"gripper xyz in base [m]: {np.round(T[:3, 3], 3)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="can0", help="SocketCAN interface (can_up.sh brings it up)")
    ap.add_argument("--check", action="store_true",
                    help="read-only: is the arm streaming? transmits nothing")
    ap.add_argument("--secs", type=float, default=2.0, help="listen seconds for --check")
    ap.add_argument("--dry-run", action="store_true",
                    help="read feedback, run the move loop, transmit nothing")
    ap.add_argument("--tx", action="store_true",
                    help="LIVE: stream p_des on 0x050. The arm may MOVE.")
    ap.add_argument("--enable", action="store_true",
                    help="also send the enable sequence (FF 1 -> 6) before streaming; "
                         "only if the arm reports but ignores 0x050")
    ap.add_argument("--home", default="here",
                    help="'here' (default): the pose measured at start is home. Or six "
                         "joint angles in degrees (what jog_a1x.py 'Set home = here' printed)")
    ap.add_argument("--dx", type=float, default=0.05, help="move +x in the base frame, m")
    ap.add_argument("--dz", type=float, default=0.0, help="move +z in the base frame, m")
    ap.add_argument("--speed", type=float, default=8.0, help="peak joint slew, deg/s")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--track-tol", type=float, default=15.0,
                    help="abort if a joint lags its setpoint by this, deg")
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV device index")
    ap.add_argument("--camera-width", type=int, default=0)
    ap.add_argument("--camera-height", type=int, default=0)
    a = ap.parse_args()

    home = None
    if a.home != "here":
        try:
            home = [math.radians(float(x)) for x in a.home.split(",")]
            assert len(home) == N_JOINTS
        except (ValueError, AssertionError):
            sys.exit("--home needs 'here' or six comma-separated angles in degrees")

    if a.check:
        sys.exit(run_check(a))

    if home is None and (a.tx or a.dry_run):
        # Read the pose first: home is where the arm is, which is the safe
        # contract (nothing jumps). A specific home comes from --home.
        arm = A1XArm(a.iface, dry_run=True)
        try:
            q, hz = read_pose(arm, a.secs)
        finally:
            arm.close()
        if q is None:
            sys.exit(1)
        print(f"arm on {a.iface}: {hz:.0f} Hz, measured (deg): {deg(q)}")
        home = q

    live = a.tx
    arm = A1XArm(a.iface, dry_run=not live, tx=live)
    cam = None
    try:
        try:
            import cv2
        except ImportError:
            if a.camera is not None:
                print("NOTE: opencv is missing; running without a camera")
            cam = None
        else:
            if a.camera is not None:
                cam = Webcam(a.camera, a.camera_width, a.camera_height)
        robot = RealRobot(arm, cam, speed=a.speed, rate=a.rate,
                          track_tol=a.track_tol, kp=a.kp, kd=a.kd, verbose=True)
        if live and a.enable:
            print("sending enable (FF 1 -> 6); the setpoint stays at the measured pose")
            arm.send_enable()
        # One minimal move: +dx in the base frame from the measured pose,
        # orientation held (the calib wave's "forward" leg, nothing more).
        q_start = robot.q()
        T0 = robot.chain.fk(q_start)
        fwd = np.array([math.cos(q_start[0]), math.sin(q_start[0]), 0.0])
        T_goal = T0.copy()
        T_goal[:3, 3] = T0[:3, 3] + a.dx * fwd + a.dz * np.array([0.0, 0.0, 1.0])
        q_goal, T_got = ik(robot.chain, T_goal, q_start, iters=200, q_bias=q_start)
        e = pose_error(T_got, T_goal)
        ep, ew = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if ep > 2e-3 or ew > math.radians(1.0):
            sys.exit(f"plan rejected: IK did not converge "
                     f"({ep * 1e3:.1f} mm, {math.degrees(ew):.1f} deg)")
        print(f"move goal: tip {np.round(T_goal[:3, 3], 3)} m  "
              f"(IK residual {ep * 1e3:.1f} mm)")
        robot.move(q_goal, 1.5)
        if robot.aborted:
            sys.exit(2)
        print("done: arm holds at the goal. Ctrl-C to stop streaming.")
        while True:
            time.sleep(1)
            arm.drain()
    except KeyboardInterrupt:
        print("\nCtrl-C: stream stopped; the arm re-latches where it is.")
    finally:
        arm.close()
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    main()
