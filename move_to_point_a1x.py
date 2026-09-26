#!/usr/bin/env python3
"""Move the real A1X's tool to a point, through galaxeo.arm.

    python3 move_to_point_a1x.py --check                          # read-only: streaming? where is the tool?
    python3 move_to_point_a1x.py --dry-run --point 0.35 0 0.12    # plan only, transmits nothing
    python3 move_to_point_a1x.py --tx --point 0.35 0 0.12         # LIVE: tool to (x, y, z) in the base frame
    python3 move_to_point_a1x.py --tx --dx 0.05                   # LIVE: +5 cm forward of home, orientation kept

Everything that makes the move safe is the package (galaxeo/arm, see README):
the plan is refused outside the reach box, when IK misses by more than 3 mm, on a
jump over 45 deg or on a collision with the table or the arm itself; the move
starts from the measured pose, ramps at --speed, aborts on stale or frozen
feedback, and on an impact holds the measured pose and backs off gently.

--point is absolute, in the arm base frame (x forward over the table, z up from the
table top). --dx/--dz are relative to the tool at --home ('here' = the pose measured
at start, or six joint angles in degrees), along the arm's heading, orientation kept.

Transmit is OFF unless --tx is passed. --enable sends function frames 1 -> 5 -> 6
first, with p_des = the measured pose streamed throughout (docs/SAFETY.md); use it
only if the arm reports but ignores 0x050. Ctrl-C mid-move holds the measured pose.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import numpy as np

from galaxeo.arm import Arm, BusTransport, enable
from galaxeo.arm.transport import deg
from galaxeo.bus import default_iface, open_bus


def wait_feedback(arm, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        q = arm.measured()
        if q is not None:
            return q
        time.sleep(0.01)
    _, why = arm.motion.fresh()
    print(f"NO DATA: {why}")
    return None


def run_check(arm, secs):
    t0 = time.monotonic()
    n0 = arm.transport.frames_rx
    while time.monotonic() - t0 < secs:
        arm.transport.read()
        time.sleep(0.005)
    r, why = arm.motion.fresh()
    if r is None:
        print(f"NO DATA: {why}")
        return 1
    tip = arm.fk(r.q)[:3, 3]
    print(f"feedback: {(arm.transport.frames_rx - n0) / secs:.0f} Hz")
    print(f"measured (deg): {deg(r.q)}")
    print(f"effort:         {', '.join(f'{e:6.2f}' for e in r.effort)}")
    print(f"tool xyz in base [m]: {np.round(tip, 3)}  "
          f"({'inside' if arm.reach.contains(tip) else 'outside'} the reach box "
          f"{arm.reach.lo.round(2).tolist()}..{arm.reach.hi.round(2).tolist()})")
    return 0


def goal_target(arm, a, home):
    if a.point is not None:
        return np.array(a.point, float)
    T = arm.fk(home)
    heading = np.array([math.cos(home[0]), math.sin(home[0]), 0.0])
    T[:3, 3] = T[:3, 3] + a.dx * heading + a.dz * np.array([0.0, 0.0, 1.0])
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=default_iface(),
                    help="can0 (Linux), xcan[:addr] (macOS), PCAN_USBBUSn (Windows), <python-can>:<channel>")
    ap.add_argument("--check", action="store_true", help="read-only: is the arm streaming? transmits nothing")
    ap.add_argument("--secs", type=float, default=2.0, help="listen seconds for --check")
    ap.add_argument("--dry-run", action="store_true", help="read feedback and plan; transmit nothing")
    ap.add_argument("--tx", action="store_true", help="LIVE: stream p_des on 0x050. The arm MOVES.")
    ap.add_argument("--enable", action="store_true",
                    help="send FF 1 -> 5 -> 6 first, p_des = measured streamed throughout; "
                         "only if the arm reports but ignores 0x050")
    ap.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="absolute tool target in the base frame, m")
    ap.add_argument("--home", default="here",
                    help="pose --dx/--dz are relative to: 'here' (measured at start) or six angles in degrees "
                         "(what jog_a1x.py 'Set home = here' printed)")
    ap.add_argument("--dx", type=float, default=0.05, help="relative: forward along the arm's heading, m")
    ap.add_argument("--dz", type=float, default=0.0, help="relative: up, m")
    ap.add_argument("--speed", type=float, default=8.0, help="average speed of the fastest joint, deg/s")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--camera", type=int, default=None, help="OpenCV index: save one frame at the goal")
    ap.add_argument("--snapshot", default="move_to_point.jpg", help="where --camera writes its frame")
    a = ap.parse_args()

    home = None
    if a.home != "here":
        try:
            home = np.radians([float(x) for x in a.home.split(",")])
            assert len(home) == 6
        except (ValueError, AssertionError):
            sys.exit("--home needs 'here' or six comma-separated angles in degrees")
    if a.enable and not a.tx:
        sys.exit("--enable transmits: it needs --tx")

    try:
        bus = open_bus(a.iface)
    except Exception as ex:
        sys.exit(f"cannot open {a.iface}: {ex}")
    transport = BusTransport(bus, tx=a.tx and not (a.check or a.dry_run))
    arm = Arm(transport, rate_hz=a.rate)
    cam = None
    try:
        if a.check:
            return run_check(arm, a.secs)
        q0 = wait_feedback(arm, a.secs)
        if q0 is None:
            return 1
        print(f"arm on {a.iface}, measured (deg): {deg(q0)}")
        target = goal_target(arm, a, q0 if home is None else home)
        plan = arm.plan(target, start=q0)
        point = target[:3, 3] if np.ndim(target) == 2 else target
        if not plan:
            print(f"plan rejected: {plan.reason}" + (f" ({plan.detail})" if plan.detail else ""))
            return 2
        print(f"plan: tool to {np.round(point, 3)} m, IK residual {plan.ik.pos_err * 1e3:.1f} mm, "
              f"joints (deg): {deg(plan.joints)}")
        if not transport.tx:
            print("dry run: nothing transmitted")
            return 0

        if a.camera is not None:
            from a1x_arm import Webcam
            cam = Webcam(a.camera)
        if a.enable:
            print("enable: FF 1 -> 5 -> 6, p_des = measured throughout")
            enable(transport)

        last = [-math.inf]

        def progress(now, cmd, reading, phase):
            if now - last[0] >= 0.5:
                last[0] = now
                left = math.degrees(float(np.max(np.abs(plan.joints - reading.q))))
                print(f"\r  {phase}: {left:5.1f} deg to go", end="", flush=True)

        arm.motion.on_tick = progress
        arm.set_armed(True)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", arm.move(plan.joints, a.speed)), daemon=True)
        th.start()
        try:
            while th.is_alive():
                th.join(0.1)
        except KeyboardInterrupt:
            arm.stop("ctrl-c")
            th.join(5.0)
        r = out.get("r")
        print()
        if r is None:
            print("move did not end cleanly; the arm holds the last p_des")
            return 2
        print(r.state + (f": {r.reason}" if r.reason else "") + (f" ({r.detail})" if r.detail else ""))
        if r.state == "reached" and cam is not None:
            import cv2
            cv2.imwrite(a.snapshot, cam.image())
            print(f"frame at the goal: {a.snapshot}")
        return 0 if r.state == "reached" else 2
    finally:
        bus.close()
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    sys.exit(main())
