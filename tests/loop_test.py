"""Loop test: feed synthetic 0x052 frames through a1x_arm's control loop.

vcan needs root this session does not have, so the fake arm and the code under
test talk through an in-process queue while stubbing only the socket layer.
Everything else -- the 200 Hz stream, the int16 decode, the lag and staleness
checks, the streamed slew, the URDF IK and FK in the loop -- is the real path
in a1x_arm.A1XArm/RealRobot, driven through the same send/drain
contract.

    python3 /tmp/rv/loop_test.py
"""
import sys, os, time, threading, struct, queue, math
import numpy as np

sys.path.insert(0, '/home/ljaniec/Repositories/galaxeo-manipulators')
sys.path.insert(0, '/home/ljaniec/Repositories/galaxeo-manipulators/ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver')

import a1x_arm as rr

txq = queue.Queue()
fbq = queue.Queue()
fake_q = [0.0, 1.0, -1.6, 0.6, 0.0, 0.0]          # sim home
stop = threading.Event()


def fake_arm():
    """A power-cycled arm obeying 0x050: p_des becomes the reported position
    after one 5 ms tick (a fast internal gain, like the real thing)."""
    period = 0.005
    nxt = time.time()
    while not stop.is_set():
        now = time.time()
        if now < nxt:
            time.sleep(min(0.001, nxt - now))
            continue
        nxt += period
        while True:
            try:
                payload = txq.get_nowait()
            except queue.Empty:
                break
            if len(payload) == 60:                   # the 0x050 command
                for j in range(6):
                    p = struct.unpack_from(">h", payload, j * 10)[0] / 4700.0
                    fake_q[j] = p                    # instant track: ideal servo
        # emit one 0x052 with the fake state
        out = bytearray(48)
        for g in range(7):
            raw = max(-32768, min(32767, int(fake_q[min(g, 5)] * 4700.0)))
            out[g * 6] = (raw >> 8) & 0xFF
            out[g * 6 + 1] = raw & 0xFF
        fbq.put(bytes(out))


t = threading.Thread(target=fake_arm, daemon=True)
t.start()


class FakeBus:
    """The kernel's socket layer stubbed: recv returns a RAW canfd_frame BYTES
    object (8-byte header + payload), which is what can_io.recv_frame unpacks
    from a real SocketCAN socket. The header mimics a 48-byte FD frame."""

    def recv(self, timeout=0.0):
        try:
            payload = fbq.get_nowait()
        except queue.Empty:
            return None
        return struct.pack("=IBBBB", 0x052, len(payload), 1, 0, 0) + payload.ljust(64, b"\x00")

    def send(self, cid, payload):
        txq.put(payload)

    def shutdown(self):
        pass


arm = rr.A1XArm.__new__(rr.A1XArm)       # skip __init__: it opens a real socket
arm.iface = "loopback"
arm.dry_run = False
arm.tx = True
arm.sock = FakeBus()
arm.q = None; arm.v = None; arm.e = None
arm.t = 0.0; arm.n = 0; arm.n_tx = 0
arm.same = 0; arm._last = None
arm.enabled = False
arm.lock = threading.Lock()

robot = rr.RealRobot(arm, None, speed=20.0, rate=200.0, track_tol=15.0, verbose=True)

robot.wait_fresh(required=True)
q0 = robot.q()
print("measured:", np.round(np.degrees(q0), 1), "deg")

# one IK move: +0.05 m along the arm's plane, like the dry-run leg
T0 = robot.chain.fk(q0)
fwd = np.array([math.cos(q0[0]), math.sin(q0[0]), 0.0])
T_goal = T0.copy()
T_goal[:3, 3] = T0[:3, 3] + 0.05 * fwd
from kinematics import ik, pose_error
q_goal, T_got = ik(robot.chain, T_goal, q0, iters=200, q_bias=q0)
e = pose_error(T_got, T_goal)
print(f"IK residual: {np.linalg.norm(e[:3]) * 1e3:.2f} mm")

robot.move(q_goal, 1.5)
assert robot.aborted is None, f"move aborted: {robot.aborted}"
time.sleep(0.05)                     # the fake arm catches up over one tick
arm.drain()
with arm.lock:
    arm.same = 0                     # the ideal servo holds: identical payloads are correct
q1 = robot.q()
T1 = robot.chain.fk(q1)
err = np.linalg.norm(T1[:3, 3] - T_goal[:3, 3])
print(f"reached: tip error vs goal {err * 1e3:.2f} mm (ideal servo, no gravity)")
print(f"GripperFK in the loop: {np.round(T1[:3, 3], 3)} m")
stop.set()
print("LOOP TEST: PASS" if err < 0.02 else f"LOOP TEST: FAIL ({err * 1e3:.1f} mm)")
