#!/usr/bin/env python3
"""Cartesian (XYZ) web control for the A1X: say where the tip should go, the
joints work out how to get there.

    ros2 launch galaxea_a1xy_driver driver.launch.py command_can_id:=80
    cd ~/ros2_ws && python3 webui_a1x.py
    # then open http://127.0.0.1:8080

Why it plans once instead of servoing
-------------------------------------
The obvious design -- creep a Cartesian setpoint toward the goal and re-solve
IK every cycle -- is WRONG on this arm, and dangerously so. At the rest pose
the tip sits ~33 mm from the J1 rotation axis, which is a singularity: J1
barely changes tip position there, so the solver swings it enormously for a
tiny Cartesian move. Measured, creeping from rest to (0.30, 0.15, 0.35):

    J1 ended at -149.6 deg   -- the long way round, through everything
    peak joint rate 103 .. 1974 deg/s depending on damping
    and it still missed by up to 107 mm

Solving ONCE from the measured pose gives the natural answer instead
(J1 = +26.3 deg, vs atan2(y, x) = +26.6 deg) and every test goal converged to
0.00 mm. So: solve once, VALIDATE the answer, then interpolate in JOINT space
at a capped rate. The tip path is not a Cartesian straight line -- it bows --
but it is bounded, predictable, and cannot be thrown by conditioning.

Guards, in the order they reject a request
------------------------------------------
    reach          |p| outside [MIN_REACH, MAX_REACH]
    floor          z below Z_FLOOR (the arm will not aim under itself)
    singularity    sqrt(x^2+y^2) < AXIS_GUARD of the J1 axis
    solver         IK residual > POS_TOL, or a non-finite/out-of-limit answer
    travel         any joint asked to move more than MAX_TRAVEL in one go

and during motion:

    JOINT_SPEED    per-cycle slew cap
    MAX_LAG        target may never lead the MEASURED pose (a stalled arm
                   cannot bank up a snap)
    deadman        no HTTP contact for DEADMAN seconds -> stop

The arm has no brakes and no compliance. Stopping is safe; starting is where it
goes wrong. Read docs/SAFETY.md.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from galaxea_a1xy_msgs.msg import MotorControl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from kinematics import Chain, ik as solve_ik
except ImportError:
    sys.exit("kinematics.py not found next to this script.\n"
             "It is mounted by docker-compose.yml; if you just added that "
             "mount, restart the container:  ./ros2.sh down && ./ros2.sh up")

N_JOINTS = 6
JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 7)]
URDF = os.path.join(HERE, "install/galaxea_a1xy_description/share/"
                          "galaxea_a1xy_description/urdf/a1x.urdf")

RATE = 100.0
KP, KD = 20.0, 2.0

# Workspace guards (metres). Reach is 0.728 m maximum, measured over 40k random
# poses; MAX_REACH is deliberately inside that so the solver is never asked to
# work at full extension.
MIN_REACH, MAX_REACH = 0.15, 0.60
Z_FLOOR = 0.05
AXIS_GUARD = 0.12          # keep the tip this far off the J1 axis

POS_TOL = 0.005            # m; reject a solve that misses by more than this
MAX_TRAVEL = math.radians(150.0)   # per-joint, per request
JOINT_SPEED = math.radians(40.0)   # rad/s slew cap
MAX_LAG = math.radians(5.0)
LIMIT_MARGIN = 0.05
DEADMAN = 3.0              # s without a browser poll -> stop

LIMITS = [(-2.880, 2.880), (0.000, 3.142), (-3.316, 0.000),
          (-1.571, 1.571), (-1.571, 1.571), (-2.880, 2.880)]


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class Arm(Node):
    def __init__(self, dry_run: bool) -> None:
        super().__init__("webui_a1x")
        self.dry_run = dry_run
        self.chain = Chain(URDF, JOINT_NAMES)
        self.lock = threading.Lock()

        self.pos = None            # measured q
        self.last_fb = 0.0
        self.cmd = None            # commanded q (what we publish)
        self.goal_q = None         # joint-space goal, or None when idle
        self.goal_p = None         # the XYZ that produced it
        self.home_q = None
        self.state = "idle"
        self.note = "waiting for feedback"
        self.last_contact = time.time()
        self.engaged = False

        self.create_subscription(JointState, "/hdas/feedback_arm", self._cb, 50)
        self.pub = self.create_publisher(
            MotorControl, "/motion_control/control_arm", 10)

    def _cb(self, m: JointState) -> None:
        with self.lock:
            self.pos = list(m.position[:N_JOINTS])
            self.last_fb = time.time()

    # ---- kinematics -------------------------------------------------------
    def fk(self, q):
        return self.chain.fk(np.asarray(q, dtype=float))[:3, 3]

    def plan(self, goal):
        """Validate an XYZ request and solve it. Returns (q, note) or (None, why)."""
        g = np.asarray(goal, dtype=float)
        if not np.all(np.isfinite(g)):
            return None, "target is not a finite point"
        r = float(np.linalg.norm(g))
        if r > MAX_REACH:
            return None, f"out of reach: |p| {r:.3f} m > {MAX_REACH} m"
        if r < MIN_REACH:
            return None, f"too close to the base: |p| {r:.3f} m < {MIN_REACH} m"
        if g[2] < Z_FLOOR:
            return None, f"below the floor guard: z {g[2]:.3f} m < {Z_FLOOR} m"
        axis = float(math.hypot(g[0], g[1]))
        if axis < AXIS_GUARD:
            return None, (f"too near the J1 axis ({axis*1000:.0f} mm): that is a "
                          f"singularity, keep sqrt(x^2+y^2) > {AXIS_GUARD} m")

        with self.lock:
            seed = np.array(self.pos, dtype=float) if self.pos else None
        if seed is None:
            return None, "no feedback yet"

        T = np.eye(4)
        T[:3, :3] = self.chain.fk(seed)[:3, :3]   # orientation is free; this is
        T[:3, 3] = g                              # only a seed for the solver
        # w_rot=0 -> position-only. Solve ONCE, from the measured pose.
        q, Tg = solve_ik(self.chain, T, seed, iters=300, w_rot=0.0,
                         q_bias=seed, k_bias=0.02)
        if not np.all(np.isfinite(q)):
            return None, "solver returned a non-finite pose"
        err = float(np.linalg.norm(Tg[:3, 3] - g))
        if err > POS_TOL:
            return None, f"unreachable from here: best solve misses by {err*1000:.0f} mm"
        for j in range(N_JOINTS):
            lo, hi = LIMITS[j]
            if not (lo - LIMIT_MARGIN <= q[j] <= hi + LIMIT_MARGIN):
                return None, (f"solution puts {JOINT_NAMES[j]} at "
                              f"{math.degrees(q[j]):.1f} deg, outside its limit")
        travel = float(np.max(np.abs(q - seed)))
        if travel > MAX_TRAVEL:
            return None, (f"that needs {math.degrees(travel):.0f} deg of joint "
                          f"travel, over the {math.degrees(MAX_TRAVEL):.0f} deg "
                          f"cap -- go there in stages")
        secs = travel / JOINT_SPEED
        return q, (f"{math.degrees(travel):.0f} deg of travel, about {secs:.1f}s, "
                   f"solved to {err*1000:.1f} mm")

    # ---- commands from the browser ---------------------------------------
    def listeners(self) -> int:
        """How many nodes subscribe to the command topic.

        driver_node only subscribes when command_can_id >= 0, so zero here means
        precisely 'no driver is listening for commands' -- either none is
        running, or it was launched read-only. Without this check the UI happily
        reports a move while the frames go nowhere."""
        return self.pub.get_subscription_count()

    def request_goto(self, goal):
        q, note = self.plan(goal)
        with self.lock:
            if q is None:
                self.note = note
                return False, note
            if not self.engaged:
                self.note = "not engaged"
                return False, "not engaged -- press ENGAGE first"
        if self.listeners() == 0:
            why = ("nothing is listening on /motion_control/control_arm -- "
                   "start the driver with command_can_id:=80 (a read-only "
                   "driver does not subscribe)")
            with self.lock:
                self.note = why
            return False, why
            self.goal_q = [float(v) for v in q]
            self.goal_p = [float(v) for v in goal]
            self.state = "moving"
            if self.dry_run:
                note += "  --  DRY RUN: nothing is sent to the arm"
            self.note = note
        return True, note

    def request_stop(self):
        with self.lock:
            self.goal_q = None
            self.state = "idle"
            if self.pos:
                self.cmd = list(self.pos)      # latch where the arm IS
            self.note = "stopped"

    def request_engage(self, on: bool):
        with self.lock:
            self.engaged = bool(on)
            self.goal_q = None
            self.state = "idle"
            if self.pos:
                self.cmd = list(self.pos)
                if self.home_q is None:
                    self.home_q = list(self.pos)
            self.note = "engaged" if on else "disengaged (released)"

    def request_home(self):
        with self.lock:
            if self.home_q is None or not self.engaged:
                self.note = "no home pose yet" if self.home_q is None else "not engaged"
                return False, self.note
            self.goal_q = list(self.home_q)
            self.goal_p = None
            self.state = "moving"
            self.note = "returning to the pose at engage"
        return True, self.note

    def touch(self):
        """Any request from the browser counts as contact. Without this a
        deadman that predates a /goto cancels the move on the very next cycle,
        so pressing Go would twitch the arm and stop."""
        with self.lock:
            self.last_contact = time.time()

    def snapshot(self):
        with self.lock:
            self.last_contact = time.time()
            pos = list(self.pos) if self.pos else None
            cmd = list(self.cmd) if self.cmd else None
            out = {
                "ok": pos is not None,
                "state": self.state,
                "engaged": self.engaged,
                "note": self.note,
                "stale": round(time.time() - self.last_fb, 3) if self.last_fb else None,
                "dry_run": self.dry_run,
                "listeners": self.pub.get_subscription_count(),
                "q_deg": [round(math.degrees(v), 2) for v in pos] if pos else None,
                "cmd_deg": [round(math.degrees(v), 2) for v in cmd] if cmd else None,
                "goal_p": self.goal_p,
                "limits_deg": [[math.degrees(a), math.degrees(b)] for a, b in LIMITS],
                "guards": {"min_reach": MIN_REACH, "max_reach": MAX_REACH,
                           "z_floor": Z_FLOOR, "axis_guard": AXIS_GUARD},
            }
        if pos:
            out["tip"] = [round(float(v), 4) for v in self.fk(pos)]
        return out

    # ---- control loop -----------------------------------------------------
    def send(self, q, kp, kd):
        if self.dry_run:
            return
        m = MotorControl()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = "arm"
        m.p_des = [float(v) for v in q]
        m.v_des = [0.0] * N_JOINTS
        m.kp = [float(kp)] * N_JOINTS
        m.kd = [float(kd)] * N_JOINTS
        m.t_ff = [0.0] * N_JOINTS
        m.mode = 0
        self.pub.publish(m)

    def spin(self):
        period = 1.0 / RATE
        last = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)
            now = time.time()
            dt = min(now - last, 0.1)
            last = now

            with self.lock:
                pos = list(self.pos) if self.pos else None
                if pos is None:
                    continue
                if self.cmd is None:
                    self.cmd = list(pos)
                    self.home_q = list(pos)
                    self.note = "ready -- not engaged"

                if now - self.last_fb > 0.25:
                    self.state = "idle"
                    self.goal_q = None
                    self.note = "feedback stale -- holding"
                elif now - self.last_contact > DEADMAN and self.state == "moving":
                    self.state = "idle"
                    self.goal_q = None
                    self.cmd = list(pos)
                    self.note = f"no browser for {DEADMAN:.0f}s -- stopped"

                if not self.engaged:
                    self.cmd = list(pos)
                    engaged = False
                else:
                    engaged = True
                    if self.state == "moving" and self.goal_q is not None:
                        step = JOINT_SPEED * dt
                        done = True
                        for j in range(N_JOINTS):
                            d = self.goal_q[j] - self.cmd[j]
                            if abs(d) > 1e-4:
                                done = False
                                self.cmd[j] += clamp(d, -step, step)
                        if done:
                            self.state = "idle"
                            self.goal_q = None
                            self.note = "arrived"
                    for j in range(N_JOINTS):
                        lo, hi = LIMITS[j]
                        self.cmd[j] = clamp(self.cmd[j],
                                            lo - LIMIT_MARGIN, hi + LIMIT_MARGIN)
                        # Never lead the measured pose: a stalled arm cannot
                        # store up a snap for when it comes free.
                        self.cmd[j] = clamp(self.cmd[j],
                                            pos[j] - MAX_LAG, pos[j] + MAX_LAG)
                cmd = list(self.cmd)
            self.send(cmd, KP if engaged else 0.0, KD if engaged else 0.0)

    def release(self):
        for _ in range(20):
            with self.lock:
                q = list(self.pos) if self.pos else (self.cmd or [0.0] * N_JOINTS)
            self.send(q, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)


# ---- http -------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    arm: Arm = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):      # keep the console for ROS output
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "webui_a1x.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, b"webui_a1x.html is missing", "text/plain")
        if self.path.split("?")[0] == "/state":
            return self._send(200, json.dumps(self.arm.snapshot()))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        self.arm.touch()
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, json.dumps({"ok": False, "note": "bad JSON"}))
        path = self.path.split("?")[0]
        if path == "/goto":
            try:
                goal = [float(body["x"]), float(body["y"]), float(body["z"])]
            except (KeyError, TypeError, ValueError):
                return self._send(400, json.dumps({"ok": False,
                                                   "note": "need x, y, z"}))
            ok, note = self.arm.request_goto(goal)
            return self._send(200, json.dumps({"ok": ok, "note": note}))
        if path == "/stop":
            self.arm.request_stop()
            return self._send(200, json.dumps({"ok": True, "note": "stopped"}))
        if path == "/engage":
            self.arm.request_engage(bool(body.get("on")))
            return self._send(200, json.dumps({"ok": True}))
        if path == "/home":
            ok, note = self.arm.request_home()
            return self._send(200, json.dumps({"ok": ok, "note": note}))
        if path == "/preview":
            try:
                goal = [float(body["x"]), float(body["y"]), float(body["z"])]
            except (KeyError, TypeError, ValueError):
                return self._send(400, json.dumps({"ok": False, "note": "need x, y, z"}))
            q, note = self.arm.plan(goal)
            return self._send(200, json.dumps({"ok": q is not None, "note": note}))
        self._send(404, json.dumps({"ok": False, "note": "not found"}))


def main():
    ap = argparse.ArgumentParser(description="Cartesian web control for the A1X.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: loopback only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dry-run", action="store_true",
                    help="serve the UI and solve, but publish nothing")
    args = ap.parse_args()

    rclpy.init()
    arm = Arm(args.dry_run)
    Handler.arm = arm
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  UI on http://{args.host}:{args.port}"
          + ("   [DRY RUN -- publishing nothing]" if args.dry_run else ""))
    print("  Ctrl-C to stop (releases the arm)")
    try:
        arm.spin()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        arm.release()
        arm.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\nreleased.")


if __name__ == "__main__":
    main()
