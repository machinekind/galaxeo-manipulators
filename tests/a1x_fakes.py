"""Test doubles shared by the A1X driver's tests and the twin backend's: an A1X on a fake bus, a virtual clock.

`FakeA1XBus` plays the arm on a virtual clock: it holds 7 groups (6 joints + gripper), moves toward
p_des at a finite speed, puts one 0x052 frame out on every `tick()` and records every frame sent.
Knobs: `silent`, `frozen`, `deaf`, `grip_effort`, `grip_rest_reading`, `state`.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field

from galaxeo import protocol as P
from galaxeo.a1x import A1XConfig
from galaxeo.bus import Frame

START_DEG = (10.0, 30.0, -60.0, 5.0, 0.0, -20.0)
STEP = 0.005                                   # one stream tick (200 Hz)
LSB_DEG = math.degrees(1.0 / P.POS_SCALE)
LSB = 2 / P.POS_SCALE                          # p_des tolerance after encoding
#: The largest gripper p_des change per tick: 2 rad/s, plus encoding.
GRIP_SLEW = 2.0 * STEP + 2 / P.POS_SCALE
DEFAULTS = A1XConfig()
#: Gripper p_des endpoints of the default config (measured 2026-09-27: +0.08 closed, -1.95 open).
OPEN = DEFAULTS.gripper_open_pdes
CLOSED = DEFAULTS.gripper_closed_pdes
#: A gripper without feedback (open loop): the tests of a guessed opening (`gripper_start`, `assume_gripper`).
OPEN_LOOP = {"gripper_fb_closed": None, "gripper_fb_open": None}


@dataclass
class Clock:
    t: float = 100.0

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class Sent:
    can_id: int
    data: bytes
    fd: bool


def gripper_pdes(data: bytes) -> float:
    return struct.unpack_from(">h", data, 0)[0] / P.POS_SCALE


def largest_rise(values: Sequence[float]) -> float:
    return max(b - a for a, b in zip(values, values[1:]))


def largest_fall(values: Sequence[float]) -> float:
    return max(a - b for a, b in zip(values, values[1:]))


@dataclass
class FakeA1XBus:
    """An A1X on the bus: answers 0x052, obeys 0x050/0x051, records what is sent."""

    start_deg: InitVar[Sequence[float]] = START_DEG
    grip_q: InitVar[float] = 0.0
    #: Group 7 in the frames until any 0x051 arrives (as after power-on: an arbitrary value,
    #: 2026-09-27 -2.88 rad with the jaws at -1.49); None = the true jaw position.
    grip_rest_reading: float | None = None
    silent: bool = False
    frozen: bool = False
    deaf: bool = False
    grip_effort: float = 0.0
    state: str = "ok"
    #: Joints move toward p_des at this speed [rad/s].
    rate: float = math.radians(120.0)
    grip_rate: float = 2.0
    q: list[float] = field(init=False)
    v: list[float] = field(default_factory=lambda: [0.0] * 7)
    eff: list[float] = field(default_factory=lambda: [0.0] * 7)
    p_des: list[float] | None = None
    g_des: float | None = None
    rx: deque = field(default_factory=deque)
    tx: list[Sent] = field(default_factory=list)
    closed: bool = False
    ticks: int = 0
    frozen_payload: bytes | None = None

    def __post_init__(self, start_deg: Sequence[float], grip_q: float) -> None:
        self.q = [math.radians(x) for x in start_deg] + [grip_q]

    def recv(self, timeout: float = 0.0) -> Frame | None:
        return self.rx.popleft() if self.rx else None

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        data = bytes(data)
        self.tx.append(Sent(can_id=can_id, data=data, fd=fd))
        if can_id == P.CMD_ID and not self.deaf and not self.frozen:
            self.p_des = [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]
        elif can_id == P.GRIP_ID and not self.frozen:
            self.g_des = gripper_pdes(data)

    def close(self) -> None:
        self.closed = True

    def tick(self, dt: float = STEP, n: int = 1) -> None:
        for _ in range(n):
            if self.p_des is not None:
                for j in range(6):
                    d = max(-self.rate * dt, min(self.rate * dt, self.p_des[j] - self.q[j]))
                    self.q[j] += d
                    self.v[j] = d / dt
            else:
                self.v[:6] = [0.0] * 6
            if self.g_des is not None and not self.grip_effort:
                self.q[6] += max(-self.grip_rate * dt, min(self.grip_rate * dt, self.g_des - self.q[6]))
            self.eff[6] = self.grip_effort
            self.ticks += 1
            if self.silent:
                continue

            q = list(self.q)
            if self.g_des is None and self.grip_rest_reading is not None:
                q[6] = self.grip_rest_reading
            if self.frozen:
                if self.frozen_payload is None:
                    self.frozen_payload = P.encode_feedback(q, self.v, self.eff)
                payload = self.frozen_payload
            else:
                self.frozen_payload = None
                eff = list(self.eff)
                eff[0] += 0.002 * (self.ticks % 2)      # measurement noise: live telemetry never repeats
                payload = P.encode_feedback(q, self.v, eff)
            self.rx.append(Frame(P.FB_ID, payload))

    def frames(self, can_id: int) -> list[bytes]:
        return [s.data for s in self.tx if s.can_id == can_id]

    def pdes_deg(self, data: bytes) -> list[float]:
        return [math.degrees(struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE) for j in range(6)]

    def grip_pdes(self) -> list[float]:
        return [gripper_pdes(d) for d in self.frames(P.GRIP_ID)]


@dataclass
class TrickleBus(FakeA1XBus):
    """As on hardware: 0x052 frames arrive one by one - each drain of the queue sees at most one."""

    gave: bool = False

    def recv(self, timeout: float = 0.0) -> Frame | None:
        if self.gave:
            self.gave = False
            return None
        if not self.rx:
            self.tick()
        self.gave = True
        return self.rx.popleft() if self.rx else None


def ticking_sleep(clock: Clock, bus: FakeA1XBus) -> Callable[[float], None]:
    """A `sleep` that moves the virtual clock and the fake arm together, one stream tick at a time."""

    def sleep(secs: float) -> None:
        n = max(1, round(secs / STEP))
        for _ in range(n):
            clock.t += secs / n
            bus.tick(secs / n)

    return sleep
