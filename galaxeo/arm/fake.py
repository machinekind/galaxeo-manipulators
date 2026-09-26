"""A fake A1X on a fake CAN bus, on a virtual clock: for tests of anything that drives the arm.

    bus = FakeA1XBus(start_deg=(0, 30, -60, 5, 0, 0))
    t = BusTransport(bus, tx=True, clock=bus.now, sleep=bus.sleep)

It speaks the real protocol: answers 0x052 at 200 Hz, obeys 0x050 p_des (joints move
toward it at `rate`), records every frame sent (`tx`). Knobs: `silent` (no feedback),
`frozen` (identical payloads, deaf, as after FF 2), `deaf` (ignores 0x050),
`eff` (effort per group), `wall` (a joint that cannot pass a position).
"""

from __future__ import annotations

import math
import struct
from collections import deque
from typing import List, Optional, Sequence, Tuple

from .. import protocol as P
from ..bus import Frame

PERIOD = 0.005


class FakeA1XBus:
    def __init__(self, start_deg: Sequence[float] = (0.0, 30.0, -60.0, 5.0, 0.0, 0.0), grip_q: float = 0.0,
                 t0: float = 100.0):
        self.q = [math.radians(x) for x in start_deg] + [grip_q]
        self.v = [0.0] * 7
        self.eff = [0.0] * 7
        self.rate = math.radians(120.0)
        self.p_des: Optional[List[float]] = None
        self.silent = self.frozen = self.deaf = False
        #: (joint index, limit [rad], direction sign): the joint stops at `limit` going in `sign`.
        self.wall: Optional[Tuple[int, float, float]] = None
        self.rx: deque = deque()
        self.tx: List[Tuple[int, bytes, bool]] = []
        self.closed = False
        self.t = t0
        self._n = 0
        self._frozen_payload: Optional[bytes] = None
        self._acc = 0.0

    # CanBus
    def recv(self, timeout: float = 0.0):
        return self.rx.popleft() if self.rx else None

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        data = bytes(data)
        self.tx.append((can_id, data, fd))
        if can_id == P.CMD_ID and not self.deaf and not self.frozen:
            self.p_des = [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]

    def close(self) -> None:
        self.closed = True

    # virtual clock
    def now(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        """Advance the clock, emitting one 0x052 frame per 5 ms."""
        self._acc += secs
        while self._acc >= PERIOD - 1e-12:
            self._acc -= PERIOD
            self.t += PERIOD
            self.tick()

    def tick(self, dt: float = PERIOD, n: int = 1) -> None:
        for _ in range(n):
            if self.p_des is not None:
                for j in range(6):
                    d = max(-self.rate * dt, min(self.rate * dt, self.p_des[j] - self.q[j]))
                    new = self.q[j] + d
                    if self.wall is not None and self.wall[0] == j:
                        _, lim, sign = self.wall
                        if sign * (new - lim) > 0:
                            new = lim if sign * (self.q[j] - lim) <= 0 else self.q[j]
                    self.v[j] = (new - self.q[j]) / dt
                    self.q[j] = new
            else:
                self.v[:6] = [0.0] * 6
            self._n += 1
            if self.silent:
                continue
            if self.frozen:
                if self._frozen_payload is None:
                    self._frozen_payload = P.encode_feedback(self.q, self.v, self.eff)
                payload = self._frozen_payload
            else:
                self._frozen_payload = None
                eff = list(self.eff)
                eff[6] += 0.002 * (self._n % 2)        # live telemetry never repeats exactly
                payload = P.encode_feedback(self.q, self.v, eff)
            self.rx.append(Frame(P.FB_ID, payload))

    # what went out
    def frames(self, can_id: int) -> List[bytes]:
        return [d for i, d, _ in self.tx if i == can_id]

    @staticmethod
    def pdes(data: bytes) -> List[float]:
        return [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]
