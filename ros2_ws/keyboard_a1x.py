#!/usr/bin/env python3
"""Keyboard jogging for the A1X, against GALAXEO's own ROS 2 driver.

    q/a  J1 base yaw        r/f  J4 wrist pitch      SPACE  stop here
    w/s  J2 shoulder pitch  t/g  J5 wrist yaw        0      back to start pose
    e/d  J3 elbow pitch     y/h  J6 wrist roll       [ ]    step size -/+
                                                     ESC    quit (releases)

Run it from a terminal with a TTY (`./ros2.sh shell`), with the driver up and
transmitting:

    ros2 launch galaxea_a1xy_driver driver.launch.py command_can_id:=80
    cd ~/ros2_ws && python3 keyboard_a1x.py

Design notes, all of them forced by the hardware:

  * The driver sends ONE CAN frame per message, so this streams at RATE Hz for
    as long as it runs -- not just when a key is pressed. An arm that stops
    receiving simply latches where it is, which is safe, but it will not track.

  * Keys ACCUMULATE INTO A TARGET rather than setting a velocity with a deadman
    timeout. A velocity+timeout scheme has to outlive the terminal's ~500 ms
    auto-repeat delay to feel smooth, which means it also keeps driving for
    that long after the key is released. Here each keypress adds one step and
    key-repeat does the rest, so releasing the key stops the motion inside one
    control cycle with no overshoot. The jog rate is step_size x your
    terminal's repeat rate.

  * The target is clamped to within MAX_LAG of the MEASURED position. If the
    arm stalls -- against its own limit, against an obstacle, against a hand --
    the target cannot keep integrating away and then snap when it comes free.
    This is the one guard that matters on an arm with no compliance.

  * Joint limits are the URDF's, plus LIMIT_MARGIN of slack: J2 and J3 rest a
    little outside their own limits on this hardware (docs/HARDWARE.md), and
    without the margin they would be unjoggable in one direction.

  * kp must be NON-ZERO or the arm discards the frame entirely. Its value is
    irrelevant -- kp does not set stiffness on this transport.

The gripper is not driven here: it lives on 0x051 and this driver's command
path only carries the 6 arm joints on 0x050.
"""
import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from galaxea_a1xy_msgs.msg import MotorControl

N_JOINTS = 6
RATE = 100.0
KP = 20.0
KD = 2.0

# URDF limits (docs/HARDWARE.md), radians.
LIMITS = [(-2.880, 2.880),   # J1 base yaw
          (0.000, 3.142),    # J2 shoulder pitch
          (-3.316, 0.000),   # J3 elbow pitch
          (-1.571, 1.571),   # J4 wrist pitch
          (-1.571, 1.571),   # J5 wrist yaw
          (-2.880, 2.880)]   # J6 wrist roll
LIMIT_MARGIN = 0.05          # rad; matches galaxea_a1xy_teleop/config/teleop.yaml

MAX_LAG = math.radians(5.0)  # target may never lead the measured pose by more
MAX_SPEED = math.radians(30.0)   # rad/s ceiling on target travel
STALE_TIMEOUT = 0.25         # s without feedback -> freeze
STALE_ABORT = 1.0            # s without feedback -> release and quit

STEP_MIN, STEP_MAX = 0.1, 2.0    # degrees per keypress

# key -> (joint index, direction)
KEYMAP = {
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
}
JOINT_LABELS = ["J1 yaw", "J2 shldr", "J3 elbow",
                "J4 wpitch", "J5 wyaw", "J6 wroll"]


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class KeyboardJog(Node):
    def __init__(self, step_deg: float, dry_run: bool) -> None:
        super().__init__("keyboard_a1x")
        self.step = math.radians(step_deg)
        self.dry_run = dry_run
        self.pos = None
        self.last_fb = 0.0
        self.target = None
        self.start = None
        self.homing = False
        self.msg = ""
        self.create_subscription(JointState, "/hdas/feedback_arm", self._cb, 50)
        self.pub = self.create_publisher(
            MotorControl, "/motion_control/control_arm", 10)

    def _cb(self, m: JointState) -> None:
        self.pos = list(m.position)
        self.last_fb = time.time()

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
        # Every populated array must be exactly N_JOINTS long or the driver
        # logs a length mismatch and drops the message.
        m.p_des = [float(x) for x in targets]
        m.v_des = [0.0] * N_JOINTS
        m.kp = [float(kp)] * N_JOINTS
        m.kd = [float(kd)] * N_JOINTS
        m.t_ff = [0.0] * N_JOINTS
        m.mode = 0
        self.pub.publish(m)

    # ---- keys -------------------------------------------------------------
    def read_keys(self):
        """Every byte waiting on stdin, ANSI escape sequences discarded."""
        out = []
        while select.select([sys.stdin], [], [], 0)[0]:
            try:
                chunk = os.read(sys.stdin.fileno(), 64)
            except OSError:
                break
            if not chunk:
                break
            i = 0
            while i < len(chunk):
                c = chunk[i:i + 1].decode("utf-8", "ignore")
                # Arrow keys and friends arrive as ESC [ X; drop the whole
                # sequence so a stray arrow cannot read as a bare-ESC quit.
                if c == "\x1b" and i + 1 < len(chunk) and chunk[i + 1:i + 2] == b"[":
                    i += 3
                    continue
                out.append(c)
                i += 1
        return out

    def apply(self, keys) -> bool:
        """False -> quit."""
        for c in keys:
            if c == "\x1b":
                return False
            if c == " ":
                # Stop where the arm actually is, not where the target drifted.
                self.target = list(self.pos[:N_JOINTS])
                self.homing = False
                self.msg = "stopped"
            elif c == "0":
                self.homing = True
                self.msg = "returning to start pose"
            elif c == "[":
                self.step = max(math.radians(STEP_MIN), self.step / 1.5)
                self.msg = f"step {math.degrees(self.step):.2f} deg"
            elif c == "]":
                self.step = min(math.radians(STEP_MAX), self.step * 1.5)
                self.msg = f"step {math.degrees(self.step):.2f} deg"
            elif c.lower() in KEYMAP:
                j, sign = KEYMAP[c.lower()]
                self.homing = False
                self.target[j] += sign * self.step
                self.msg = ""
        return True

    def constrain(self, pos) -> None:
        """Pull the target inside the joint limits and inside MAX_LAG of the arm."""
        for j in range(N_JOINTS):
            lo, hi = LIMITS[j]
            # Joint limits, with slack for the resting zero offsets.
            self.target[j] = clamp(self.target[j],
                                   lo - LIMIT_MARGIN, hi + LIMIT_MARGIN)
            # Never lead the arm by more than MAX_LAG: if it is stalled, the
            # target stops running away instead of storing up a snap for when
            # it comes free. Meaningless with nothing driving the arm, and it
            # would just pin the dry-run display 5 deg from the real pose.
            if not self.dry_run:
                self.target[j] = clamp(self.target[j],
                                       pos[j] - MAX_LAG, pos[j] + MAX_LAG)

    # ---- loop -------------------------------------------------------------
    def status(self) -> str:
        pos = self.pos[:N_JOINTS] if self.pos else [0.0] * N_JOINTS
        cells = []
        for j in range(N_JOINTS):
            drift = abs(self.target[j] - pos[j])
            mark = "*" if drift > math.radians(0.3) else " "
            cells.append(f"{JOINT_LABELS[j]}{mark}{math.degrees(pos[j]):7.2f}")
        return ("  ".join(cells)
                + f" | {math.degrees(self.step):.2f}d/key"
                + (f" | {self.msg}" if self.msg else ""))

    def run(self) -> int:
        if self.wait_pos() is None:
            print("no feedback on /hdas/feedback_arm -- is the driver running?")
            return 1
        self.start = list(self.pos[:N_JOINTS])
        self.target = list(self.start)
        print(__doc__.split("Design notes")[0].rstrip() + "\r")
        print(f"  start pos (deg): "
              f"{[round(math.degrees(x), 2) for x in self.start]}\r")
        if self.dry_run:
            print("  DRY RUN -- publishing nothing\r")
        print("\r")

        period = 1.0 / RATE
        last = time.time()
        next_paint = 0.0
        rc = 0
        try:
            while True:
                now = time.time()
                dt = min(now - last, 0.1)
                last = now

                if not self.apply(self.read_keys()):
                    break

                stale = now - self.last_fb
                if stale > STALE_ABORT:
                    self.msg = f"feedback lost for {stale:.1f}s -- releasing"
                    rc = 1
                    break
                if stale > STALE_TIMEOUT:
                    # Hold, but do not integrate against unknown state.
                    self.msg = "feedback stale -- holding"
                    self.send(self.target, KP, KD)
                    rclpy.spin_once(self, timeout_sec=period)
                    continue

                pos = self.pos[:N_JOINTS]
                if self.homing:
                    done = True
                    for j in range(N_JOINTS):
                        gap = self.start[j] - self.target[j]
                        if abs(gap) > math.radians(0.1):
                            done = False
                            self.target[j] += clamp(gap, -MAX_SPEED * dt,
                                                    MAX_SPEED * dt)
                    if done:
                        self.homing = False
                        self.msg = "at start pose"

                self.constrain(pos)
                self.send(self.target, KP, KD)

                if now >= next_paint:
                    sys.stdout.write("\r\x1b[2K" + self.status())
                    sys.stdout.flush()
                    next_paint = now + 0.1

                rclpy.spin_once(self, timeout_sec=period)
        except KeyboardInterrupt:
            pass
        finally:
            # kp=kd=t_ff=0 is the one command provably incapable of moving the
            # arm. An uncommanded arm holds where it is; it does not fall.
            for _ in range(20):
                self.send(self.pos[:N_JOINTS] if self.pos else self.target,
                          0.0, 0.0)
                rclpy.spin_once(self, timeout_sec=0.01)
            end = [round(math.degrees(x), 2)
                   for x in (self.pos[:N_JOINTS] if self.pos else [])]
            sys.stdout.write("\r\x1b[2K")
            print(f"released. end pos (deg): {end}\r")
            if self.msg:
                print(f"{self.msg}\r")
        return rc


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Keyboard jogging for the Galaxea A1X.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=0.5,
                    help="degrees added per keypress (default: 0.5). Jog rate "
                         "is this times your terminal's key-repeat rate.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the targets, publish nothing")
    args = ap.parse_args()
    if not STEP_MIN <= args.step <= STEP_MAX:
        ap.error(f"--step must be {STEP_MIN}..{STEP_MAX} degrees")
    if not sys.stdin.isatty():
        ap.error("stdin is not a TTY; run this from `./ros2.sh shell`")

    rclpy.init()
    node = KeyboardJog(args.step, args.dry_run)
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    rc = 0
    try:
        # cbreak, not raw: ISIG stays on so Ctrl-C still interrupts.
        tty.setcbreak(fd)
        rc = node.run()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
