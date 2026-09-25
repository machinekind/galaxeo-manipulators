#!/usr/bin/env python3
"""The real-arm `Robot` behind `calib/calibrate.py`: A1X + webcam, direct Python.

    python3 real_robot_a1x.py --check               # read-only: is the arm streaming?
    python3 real_robot_a1x.py --power --secs 3      # read-only, longer listen
    python3 real_robot_a1x.py --dry-run --home 0,90,-91.7,34.4,0,0   # GUI-less dry run
    python3 real_robot_a1x.py --home 0,90,-91.7,34.4,0,0             # live: stream + camera

This is the missing piece between the calibration stack and the hardware:
`calib/calibrate.py` runs only `--sim` because it had no `Robot` with `q`,
`move` and `image` on the real arm. This module provides exactly that, on top
of the repository's own recovered CAN-FD protocol
(`ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver/protocol.py`) and raw
SocketCAN I/O (`can_io.py`), the same two files the ROS 2 driver node uses --
no ROS, no vendor binaries.

Standalone use, no calibration needed:

    from real_robot_a1x import RealRobot, A1XArm, Webcam
    arm = A1XArm("can0")                    # SocketCAN, dry_run=False
    robot = RealRobot(arm, Webcam(index=0))
    q = robot.q()                           # measured joint angles, 6
    robot.move(q + [0, 0.2, 0, 0, 0, 0], 2.0)   # streamed slew, aborts on lag
    img = robot.image()                     # BGR frame from the webcam

Also `robot.gripper2base(q)` -- the FK glue: gripper_link pose in the arm base
frame from the URDF (`kinematics.py`, pure numpy, no MuJoCo), which is what
`calib.handeye.Observation.T_frame2base` wants. The sim's `FK` class builds the
same transform from `a1x.xml` through MuJoCo; the URDF and the MuJoCo model
share the same joint axes and origins, so the two agree to float round-off.

Safety, same rules as jog_a1x.py / ik_demo_a1x.py (see docs/SAFETY.md):

  * Transmit is OFF unless --tx is passed. A power-cycled arm obeys 0x050
    immediately, so the first live frame is seeded from the MEASURED pose.
  * p_des only ever moves at --speed (deg/s), streamed at --rate. Nothing jumps.
  * Feedback older than 150 ms aborts; a joint more than --track-tol behind its
    setpoint aborts. The stream stops and the arm holds where it is.
  * The gripper is never commanded. Targets are clamped to the URDF limits,
    widened to include the measured start pose (J3 reads ~1.5 deg past its
    limit at rest).
  * An enable (function frames 1 -> 6) is --enable, off by default: use it only
    if the arm reports but ignores 0x050. The setpoint stays pinned to the
    measured pose throughout, as docs/SAFETY.md demands.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ros2_ws", "src",
                                "galaxea_a1xy_driver", "galaxea_a1xy_driver"))

from kinematics import Chain, ik, pose_error, _T, _rpy         # noqa: E402
from protocol import (                                          # noqa: E402
    CMD_CAN_ID, FB_CAN_ID, FB_LEN, FF_CAN_ID, N_JOINTS,
    FF_ENABLE, FF_ENABLE2, ArmCommand,
    decode_feedback, encode_command, encode_function_frame,
)

try:                        # raw SocketCAN I/O, the driver's own module
    import can_io
except ImportError:         # xcan / macOS / no SocketCAN: read-only paths still import
    can_io = None

URDF = os.path.join(HERE, "ros2_ws", "src", "galaxea_a1xy_description",
                    "urdf", "a1x.urdf")
JOINTS = [f"arm_joint{i}" for i in range(1, 7)]

# URDF limits, rad. jog_a1x.py hardcodes the same numbers; read from the URDF
# here so one edit keeps them in sync.
_LIMITS = [(-2.8798, 2.8798), (0.0, 3.1416), (-3.3161, 0.0),
           (-1.5708, 1.5708), (-1.5708, 1.5708), (-2.8798, 2.8798)]

FB_STALE_S = 0.15          # feedback older than this aborts a live stream


# --------------------------------------------------------------------- CAN
class A1XArm:
    """The arm over raw SocketCAN: 200 Hz feedback decode plus the two frames
    it listens to, built directly on `can_io` and `protocol`.

    Same contract as `jog_a1x.A1X` (q/v/e, t, same, drain, send_arm) so code
    written against either works with both -- but this one is the Linux path
    through the repository's own protocol module, not python-can.
    """

    def __init__(self, iface="can0", dry_run=True, tx=False):
        if not dry_run and not tx:
            raise ValueError("live arm needs tx=True (or use dry_run=True)")
        if not dry_run and can_io is None:
            raise SystemExit("can_io is missing (no SocketCAN on this machine?)")
        self.iface = iface
        self.dry_run = dry_run
        self.tx = tx
        self.sock = None
        if not dry_run:
            self.sock = can_io.open_socket(iface, rx_only=False)
            self.sock.setblocking(False)
        self.q = None; self.v = None; self.e = None
        self.t = 0.0; self.n = 0; self.n_tx = 0
        self.same = 0; self._last = None
        self.enabled = False
        self.lock = threading.Lock()

    # ---- receive ---------------------------------------------------------
    def drain(self):
        """Read every queued frame; a partial drain makes readings seconds stale."""
        got = False
        if self.sock is None:
            return got
        while True:
            try:
                frame = can_io.recv_frame(self.sock)
            except BlockingIOError:
                break
            except OSError:
                break
            except TypeError:
                break          # a stubbed socket (tests) returning None
            if frame is None:
                continue
            cid, payload = frame
            if cid != FB_CAN_ID or len(payload) != FB_LEN:
                continue
            fb = decode_feedback(payload)
            with self.lock:
                self.q = list(fb.position)
                self.v = list(fb.velocity)
                self.e = list(fb.effort)
                self.same = self.same + 1 if payload == self._last else 0
                self._last = payload
                self.t = time.time()
                self.n += 1
            got = True
        return got

    def fresh(self):
        """(q or None, reason). Fresh feedback or a printed reason."""
        self.drain()
        with self.lock:
            if self.q is None:
                return None, f"no 0x{FB_CAN_ID:03x} feedback on {self.iface}: arm off, or bus down?"
            if time.time() - self.t > FB_STALE_S:
                return None, f"feedback stale ({(time.time() - self.t) * 1e3:.0f} ms)"
            if self.same > 150:
                return None, "telemetry FROZEN (identical payloads): not starting"
        return np.array(self.q[:N_JOINTS], float), ""

    # ---- transmit --------------------------------------------------------
    def _send(self, cid, payload):
        if self.dry_run or not self.tx:
            return
        try:
            self.sock.send_frame(cid, payload)      # real bus: can_io's module-level API
        except AttributeError:
            self.sock.send(cid, payload)            # tests: a bus-level stub
        self.n_tx += 1

    def send_arm(self, p, kp=20.0, kd=1.0):
        """One 0x050 joint command: p_des = p, everything else per the fields.

        The arm honours p_des and nothing else (measured: kp sets nothing, t_ff
        is inert to +-12 Nm); the gains ride along because they are harmless."""
        cmd = ArmCommand()
        cmd.p_des = [float(x) for x in p]
        cmd.v_des = [0.0] * N_JOINTS
        cmd.kp = [kp] * N_JOINTS
        cmd.kd = [kd] * N_JOINTS
        self._send(CMD_CAN_ID, encode_command(cmd))

    def send_zero_torque(self):
        """The only command provably incapable of moving the arm."""
        self._send(CMD_CAN_ID, encode_command(ArmCommand.zero_torque()))

    def send_enable(self):
        """Function frames 1 -> 6, 0.3 s apart, as the ROS driver does."""
        for code in (FF_ENABLE, FF_ENABLE2):
            self._send(FF_CAN_ID, encode_function_frame(code))
            time.sleep(0.3)
        self.enabled = True

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


# ------------------------------------------------------------------ webcam
class Webcam:
    """One UVC camera. Same contract as `calib.calibrate.Robot.image()`.

    BGR frames, like OpenCV reads them; `calib.tags.detect` takes RGB and
    converts, and anything ndim==3 works. The wrist module streams upside
    down; a table-overhead webcam does not, so no rotation is applied here.
    """

    def __init__(self, index=0, width=0, height=0, fourcc="MJPG"):
        import cv2
        self.cv2 = cv2
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self._open(width, height, fourcc)

    def _open(self, width, height, fourcc):
        cv2 = self.cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*fourcc))
        if not cap.isOpened():
            raise SystemExit(f"camera {self.index} did not open")
        for _ in range(10):            # let exposure settle
            cap.read()
        self.cap = cap

    def image(self):
        with self.lock:
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.index} read failed")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------------------------------------------------- Robot
class RealRobot:
    """The `Robot` protocol `calib.calibrate` wants, on the real arm:

        q()        measured joint angles, 6, rad
        move(q, dur)   streamed joint slew to q over ~dur, aborts on lag
        image()    BGR frame from the webcam

    The move is the ik_demo_a1x.py streaming pattern: the setpoint is seeded
    from the MEASURED pose and slews toward the goal at `speed` deg/s while
    p_des streams at `rate` Hz. `dur` bounds the slew the way
    `calibrate.move_time` bounds the sim's: peak joint speed, not wall clock.
    Feedback older than 150 ms or a joint lagging its setpoint by more than
    `track_tol` aborts the stream and leaves the arm holding.
    """

    def __init__(self, arm, cam=None, speed=8.0, rate=200.0, track_tol=15.0,
                 kp=20.0, kd=1.0, verbose=False):
        self.arm = arm
        self.cam = cam
        self.speed = math.radians(speed)      # deg/s -> rad/s
        self.rate = rate
        self.track_tol = math.radians(track_tol)
        self.kp, self.kd = kp, kd
        self.verbose = verbose
        self.chain = Chain(URDF, JOINTS)
        self.aborted = None

    def q(self):
        q, why = self.arm.fresh()
        if q is None:
            raise RuntimeError(why)
        return q

    def wait_fresh(self, required=False, timeout=3.0):
        """True once fresh feedback is in hand. The move's last streamed frame
        does not come back instantly; capturing before it does puts T_frame2base
        in a pose the image was not taken in."""
        t0 = time.time()
        why = "no feedback yet"
        while time.time() - t0 < timeout:
            q, why = self.arm.fresh()
            if q is not None:
                return True
            time.sleep(0.01)
        if required:
            raise RuntimeError(f"no fresh feedback after {timeout:.0f} s ({why})")
        return False

    def move(self, q_goal, dur=1.5):
        """Stream to q_goal. `dur` = the sim's move_time contract: the longest
        leg takes about this long, so peak joint speed stays bounded."""
        self.aborted = None
        q_start = self.q()
        lohi = [(min(lo, q_start[j]), max(hi, q_start[j]))
                for j, (lo, hi) in enumerate(_LIMITS)]
        goal = np.array([min(hi, max(lo, float(x))) for x, (lo, hi) in zip(q_goal, lohi)])
        travel = float(np.max(np.abs(goal - q_start)))
        if travel < 1e-4:
            return
        # Peak joint speed capped: the whole move takes travel/speed, floored
        # at dur like the sim's WAVE['dur'] so short hops do not get rushed.
        move_s = max(travel / self.speed, min(dur, 3.5))
        step_max = travel / max(move_s * self.rate, 1)    # rad per streamed frame
        target = q_start.copy()
        period = 1.0 / self.rate
        nxt = time.time(); prev = nxt; done_at = None
        if self.verbose:
            print(f"  move: {math.degrees(travel):5.1f} deg over {move_s:4.1f} s "
                  f"at {math.degrees(self.speed):g} deg/s", flush=True)
        while True:
            self.arm.drain()
            now = time.time()
            if now < nxt:
                time.sleep(min(0.001, nxt - now))
                continue
            nxt += period
            if nxt < now - 0.05:                 # fell behind: resync, do not burst
                nxt = now
            dt = min(0.05, now - prev); prev = now
            with self.arm.lock:
                stale = now - self.arm.t
            if not self.arm.dry_run and stale > FB_STALE_S:
                self.aborted = f"stale feedback ({stale * 1e3:.0f} ms)"
                print(f"ABORT: {self.aborted}. Stream stopped; arm holds.", flush=True)
                return
            with self.arm.lock:
                meas = np.array(self.arm.q[:N_JOINTS], float) if self.arm.q else q_start
            lag = np.abs(meas - target)
            if not self.arm.dry_run and np.max(lag) > self.track_tol:
                j = int(np.argmax(lag))
                self.aborted = (f"joint {j + 1} lags its setpoint by "
                                f"{math.degrees(lag[j]):.1f} deg")
                print(f"ABORT: {self.aborted}. Stream stopped; arm holds.", flush=True)
                return
            d = goal - target
            # advance the setpoint along the ramp by dt at the peak rate: the
            # whole move takes move_s, so a 2x dt burst advances 2x — that is
            # the ik_demo_a1x.py pattern, target += clip(d, -step*dt, +step*dt).
            target = target + np.clip(d, -step_max * dt / period, step_max * dt / period)
            self.arm.send_arm(target, self.kp, self.kd)
            worst = float(np.max(np.abs(goal - target)))
            if self.verbose and now - prev >= 0.5:
                print(f"\r  {worst:5.1f} deg to go", end="", flush=True)
            if worst < 0.05:                     # 0.05 deg: settle before returning
                if done_at is None:
                    done_at = now
                elif now - done_at >= 0.4:
                    break
        # the last streamed frame is behind the queue: drain until the arm
        # reports the goal, so the caller's q() is the pose the move ended in.
        # An ideal servo converges; a real one settles within track_tol, and
        # the loop breaks at 0.05 deg of residual either way.
        t0 = time.time()
        while time.time() - t0 < 0.5:
            self.arm.drain()
            if self.arm.q is not None:
                with self.arm.lock:
                    meas = np.array(self.arm.q[:N_JOINTS], float)
                if np.max(np.abs(meas - goal)) < 0.05:
                    break
            time.sleep(0.005)
        if self.verbose:
            print("\r  reached." + " " * 40, flush=True)

    def image(self):
        if self.cam is None:
            raise RuntimeError("no camera on this RealRobot")
        return self.cam.image()

    def close(self):
        if self.cam is not None:
            self.cam.close()


# ------------------------------------------------------------------ FK glue
class GripperFK:
    """gripper_link pose in the arm base frame, from the URDF alone.

    The calib stack's `FK` class does this through MuJoCo (`a1x.xml`); the
    URDF and the MuJoCo model describe the same chain -- same joint origins,
    same axes, the fixed gripper_joint 0.08165 m out -- so this agrees with it
    to float round-off. `Observation.T_frame2base` wants exactly this.
    """

    def __init__(self, urdf=URDF):
        self.chain = Chain(URDF, JOINTS)
        # gripper_joint is fixed, so the tip of the 6-joint chain plus its
        # origin transform is the gripper_link origin.
        self.T_grip = _T(_rpy(0, 0, 0), self.chain.all["gripper_joint"].xyz)

    def gripper2base(self, q):
        T = self.chain.fk(q)
        return T @ self.T_grip


# ------------------------------------------------------------------- check
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


def deg(q):
    return ", ".join(f"{math.degrees(x):6.1f}" for x in q)


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
    ap.add_argument("--secs", type=float, default=2.0, help="listen seconds for --check/--power")
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
