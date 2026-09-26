"""Where the motion primitive reads q / dq / effort and writes p_des: the CAN bus, a fake bus, or MuJoCo.

    Reading   one feedback sample: receive time, q [rad], dq [rad/s], effort, frozen flag
    Transport now() / sleep(s) / read() / write(p_des) / effort_limits(descend)

BusTransport speaks the A1X protocol over any galaxeo.bus transport (SocketCAN,
XCAN, python-can) or over `fake.FakeA1XBus`. It transmits only with `tx=True`, and
the only frames it can put on the wire are 0x050 (p_des, kp 20, kd 1) and the
enable codes 1, 5, 6 on 0x053. Release codes (2, 3, 4) are refused.

SimTransport steps the installation model in MuJoCo on a virtual clock: p_des goes to
the position actuators, effort is the actuator force [N m].
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .. import protocol as P
from .model import GRIPPER, JOINTS, build, indices

#: Identical 0x052 payloads in a row that mean frozen telemetry (the released state after FF 2).
FROZEN_FRAMES = 50
KP = 20.0
KD = 1.0

# Effort thresholds in 0x052 effort units (raw / 600), the same scale as docs/STEERING.md:
# resting 1-1.7, gravity load ~3, a hand push 5-22, a stop contact ~25, saturation 50.
# FIRST-SESSION GUESSES: replace them with values measured on the arm.
EFFORT_MOVE = 20.0
EFFORT_DESCEND = 12.0
# Sim thresholds as a fraction of each actuator's forcerange (normal / descending).
SIM_EFFORT_FRAC = 0.8
SIM_EFFORT_FRAC_DESCEND = 0.5


@dataclass
class Reading:
    t: float              # receive time on the transport clock [s]
    q: np.ndarray         # arm_joint1..6 [rad]
    dq: np.ndarray        # [rad/s]
    effort: np.ndarray    # transport units (0x052 effort, or N m in sim)
    frozen: bool = False


class BusTransport:
    """The A1X on a CAN bus. `clock`/`sleep` default to wall time; a fake bus passes its virtual ones."""

    rate_hz = 200.0

    def __init__(self, bus, *, tx: bool = False, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, kp: float = KP, kd: float = KD):
        self.bus = bus
        self.tx = bool(tx)
        self.now = clock
        self.sleep = sleep
        self.kp, self.kd = kp, kd
        self._last: Optional[Reading] = None
        self._raw: Optional[bytes] = None
        self._same = 0
        self.frames_rx = 0
        self.frames_tx = 0

    def read(self) -> Optional[Reading]:
        """Drain every queued frame (a partial drain gives readings seconds old) and return the newest."""
        while True:
            f = self.bus.recv(0.0)
            if f is None:
                break
            if f.can_id != P.FB_ID or len(f.data) != P.FB_LEN:
                continue
            data = bytes(f.data)
            fb = P.decode_feedback(data)
            self._same = self._same + 1 if data == self._raw else 0
            self._raw = data
            self.frames_rx += 1
            self._last = Reading(self.now(), np.array(fb.pos[:6]), np.array(fb.vel[:6]),
                                 np.array(fb.eff[:6]), self._same > FROZEN_FRAMES)
        return self._last

    def write(self, p_des: Sequence[float]) -> None:
        if not self.tx:
            return
        self.bus.send(P.CMD_ID, P.encode_arm([float(x) for x in p_des], self.kp, self.kd))
        self.frames_tx += 1

    def send_ff(self, code: int) -> None:
        if int(code) in P.RELEASE_CODES:
            raise ValueError(f"FF {code} releases the motors (no brakes: the arm drops); never sent")
        if not self.tx:
            return
        self.bus.send(P.FF_ID, P.encode_ff(code), fd=False)

    def effort_limits(self, descend: bool) -> Optional[np.ndarray]:
        return np.full(6, EFFORT_DESCEND if descend else EFFORT_MOVE)


class SimTransport:
    """The installation model in MuJoCo, on a virtual clock that advances with `sleep`."""

    rate_hz = 50.0

    def __init__(self, q0: Optional[Sequence[float]] = None, *, obstacles: Iterable = (), tool=None):
        import mujoco

        self._mj = mujoco
        self.model = build(tool or GRIPPER, tuple(obstacles))
        self.data = mujoco.MjData(self.model)
        self.ix = indices(self.model)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        if q0 is not None:
            d.qpos[self.ix.qadr] = np.asarray(q0, float)
        self.act = np.array([int(np.flatnonzero(m.actuator_trnid[:, 0] == m.joint(j).id)[0]) for j in JOINTS])
        d.ctrl[self.act] = d.qpos[self.ix.qadr]
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.writes = 0
        #: Extra effort added to the reading per joint, to fake a spike in tests.
        self.effort_bias = np.zeros(6)

    def now(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        n = max(1, int(round(secs / self.model.opt.timestep)))
        for _ in range(n):
            self._mj.mj_step(self.model, self.data)
        self.t += n * self.model.opt.timestep

    def read(self) -> Reading:
        d = self.data
        return Reading(self.t, d.qpos[self.ix.qadr].copy(), d.qvel[self.ix.dadr].copy(),
                       d.actuator_force[self.act].copy() + self.effort_bias)

    def write(self, p_des: Sequence[float]) -> None:
        self.data.ctrl[self.act] = np.asarray(p_des, float)
        self.writes += 1

    def effort_limits(self, descend: bool) -> np.ndarray:
        frac = SIM_EFFORT_FRAC_DESCEND if descend else SIM_EFFORT_FRAC
        r = self.model.actuator_forcerange[self.act]
        return frac * np.maximum(np.abs(r[:, 0]), np.abs(r[:, 1]))


def deg(q: Sequence[float]) -> str:
    return ", ".join(f"{math.degrees(x):6.1f}" for x in q)
