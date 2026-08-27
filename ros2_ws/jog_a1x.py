#!/usr/bin/env python3
"""Bounded single-joint jog for the A1X, against GALAXEO's own ROS 2 driver.

`move_test.py` next to this file does the same thing against the VENDOR stack
(`hdas_msg` + the `/function_frame_arm` service). Neither exists when the arm is
driven by `galaxea_a1xy_driver`, so this is that test rewritten for
`galaxea_a1xy_msgs/MotorControl`.

The driver sends exactly ONE CAN frame per message it receives -- there is no
streaming loop on the command path -- so a publisher has to keep the frames
coming at 100-200 Hz itself. That is what this script is for; `ros2 topic pub
--once` puts a single frame on the wire and the arm ignores it.

Safety design, same as move_test.py:
  * ONE joint only; the default is arm_joint6 (wrist roll: lightest, lowest
    inertia).
  * p_des is seeded from the joint's MEASURED position, so the commanded motion
    starts at zero and there is nothing to snap to.
  * Ramps out by AMPLITUDE and back, ending where it started.
  * Aborts if the joint deviates more than ABORT_ERR from target.
  * Always releases (kp=kd=0) on exit, including on exception.

  ./jog_a1x.py                 # arm_joint6, 6 deg
  ./jog_a1x.py 6 10            # arm_joint6, 10 deg
  ./jog_a1x.py 1 15 --dry-run  # print targets, publish nothing
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from galaxea_a1xy_msgs.msg import MotorControl

N_JOINTS = 6
RAMP_S = 3.0
HOLD_S = 1.0
RATE = 100.0
ABORT_ERR = math.radians(25.0)
# kp does not set stiffness on this arm -- a 3 deg step tracks identically at
# 0.5 and at 20 (docs/STEERING.md). But it must be NON-ZERO or the arm discards
# the whole frame and keeps holding its own setpoint.
KP = 20.0
KD = 2.0
# The gravity-loaded shoulder and elbow carry the arm's weight; keep their
# default throw smaller than the wrist's.
HEAVY_JOINTS = (2, 3)
HEAVY_MAX_DEG = 10.0


class Jog(Node):
    def __init__(self, joint: int, amplitude: float, kp: float, kd: float,
                 dry_run: bool) -> None:
        super().__init__("jog_a1x")
        self.joint = joint
        self.amplitude = amplitude
        self.kp, self.kd = kp, kd
        self.dry_run = dry_run
        self.pos = None
        self.create_subscription(JointState, "/hdas/feedback_arm", self._cb, 50)
        self.pub = self.create_publisher(
            MotorControl, "/motion_control/control_arm", 10)

    def _cb(self, msg: JointState) -> None:
        self.pos = list(msg.position)

    def wait_pos(self, timeout: float = 5.0):
        end = time.time() + timeout
        while time.time() < end and self.pos is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.pos

    def send(self, targets, kp: float, kd: float) -> None:
        if self.dry_run:
            return
        m = MotorControl()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = "arm"
        # Every populated array must be exactly N_JOINTS long: the driver logs
        # a length mismatch and DROPS the message rather than sending a partial
        # command.
        m.p_des = [float(x) for x in targets]
        m.v_des = [0.0] * N_JOINTS
        m.kp = [float(kp)] * N_JOINTS
        m.kd = [float(kd)] * N_JOINTS
        m.t_ff = [0.0] * N_JOINTS
        m.mode = 0
        self.pub.publish(m)

    def run(self) -> int:
        if self.wait_pos() is None:
            print("no feedback on /hdas/feedback_arm -- is the driver running?")
            return 1
        start = list(self.pos[:N_JOINTS])
        j = self.joint
        print(f"  start pos (deg): {[round(math.degrees(x), 2) for x in start]}")
        print(f"  moving arm_joint{j + 1} by {math.degrees(self.amplitude):.1f} deg, "
              f"kp={self.kp}, kd={self.kd}"
              + ("   [DRY RUN -- publishing nothing]" if self.dry_run else ""))

        t0 = time.time()
        total = RAMP_S + HOLD_S + RAMP_S
        peak_err = 0.0
        aborted = False
        try:
            while True:
                t = time.time() - t0
                if t > total:
                    break
                if t < RAMP_S:
                    frac = t / RAMP_S
                elif t < RAMP_S + HOLD_S:
                    frac = 1.0
                else:
                    frac = max(0.0, 1.0 - (t - RAMP_S - HOLD_S) / RAMP_S)
                tgt = list(start)
                tgt[j] = start[j] + self.amplitude * frac
                self.send(tgt, self.kp, self.kd)
                rclpy.spin_once(self, timeout_sec=1.0 / RATE)
                if self.pos and not self.dry_run:
                    err = abs(self.pos[j] - tgt[j])
                    peak_err = max(peak_err, err)
                    if err > ABORT_ERR:
                        print(f"  ABORT: tracking error "
                              f"{math.degrees(err):.1f} deg")
                        aborted = True
                        break
                if abs(t % 1.0) < 0.012 and self.pos:
                    print(f"    t={t:4.1f}s  target={math.degrees(tgt[j]):7.2f}  "
                          f"actual={math.degrees(self.pos[j]):7.2f} deg")
        finally:
            # Release: kp=kd=t_ff=0 is the one command provably incapable of
            # moving the arm. An uncommanded arm holds where it is.
            for _ in range(20):
                self.send(start, 0.0, 0.0)
                rclpy.spin_once(self, timeout_sec=0.01)
            end = list(self.pos[:N_JOINTS]) if self.pos else []
            if end:
                print(f"  end pos   (deg): {[round(math.degrees(x), 2) for x in end]}")
                print(f"  peak tracking error: {math.degrees(peak_err):.2f} deg")
                print(f"  net displacement of arm_joint{j + 1}: "
                      f"{math.degrees(abs(end[j] - start[j])):.2f} deg")
        return 1 if aborted else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joint", nargs="?", type=int, default=6,
                    help="joint to move, 1-6 (default: 6, wrist roll)")
    ap.add_argument("degrees", nargs="?", type=float, default=6.0,
                    help="amplitude in degrees (default: 6)")
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the targets, publish nothing")
    args = ap.parse_args()

    if not 1 <= args.joint <= N_JOINTS:
        ap.error(f"joint must be 1..{N_JOINTS}")
    if args.joint in HEAVY_JOINTS and abs(args.degrees) > HEAVY_MAX_DEG:
        ap.error(f"arm_joint{args.joint} carries the arm's weight; keep the first "
                 f"moves under {HEAVY_MAX_DEG:g} deg")
    if abs(args.degrees) > 45.0:
        ap.error("amplitude over 45 deg; jog in smaller steps")
    if args.kp <= 0.0 and not args.dry_run:
        ap.error("kp must be non-zero or the arm discards the frame entirely")

    rclpy.init()
    node = Jog(args.joint - 1, math.radians(args.degrees),
               args.kp, args.kd, args.dry_run)
    rc = 0
    try:
        rc = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
