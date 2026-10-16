"""Fakes for bench_a1x.py and its tools: the A1X on a fake CAN bus with its own virtual clock, and scripted
keys. Nothing here opens a link.

`BenchBus` overlaps `a1x_fakes.FakeA1XBus` (the same 5 ms step and the same jaw endpoints of the default
config). It differs where the bench needs it: `sleep` advances its clock, so an unthreaded station runs on
it, and it plays a jaw blocked by a cube, the adapter's echo and a hook on every tick.
"""

from __future__ import annotations

import math
import struct
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field

import bench_a1x as bench
from galaxeo import protocol as P
from galaxeo.a1x import A1XConfig
from galaxeo.bus import Frame as CanFrame

STEP = 0.005                                   # the arm answers 0x052 every 5 ms
#: Jaw p_des of the default `a1x` config: open and closed (more positive = tighter).
JAW_OPEN = A1XConfig().gripper_open_pdes
JAW_CLOSED = A1XConfig().gripper_closed_pdes


def jaw_pdes(opening: float) -> float:
    return JAW_CLOSED + opening / 100.0 * (JAW_OPEN - JAW_CLOSED)


@dataclass(frozen=True)
class Sent:
    can_id: int
    data: bytes
    fd: bool


@dataclass
class BenchBus:
    """An A1X on a CAN bus, on its own virtual clock that `sleep` advances in 5 ms ticks.

    A drain that finds the queue empty still gets one frame of the present state (the clock does not
    move for it), as a reader polling the hardware would: the backend's connect polls on wall time.
    Joints move toward p_des at `rate`. The jaw moves toward its p_des at `grip_rate` with
    `moving_effort` on group 7; with `cube_pdes` it stops there when closing and reads `loaded_effort`.
    `on_tick` runs after every tick, on the thread that sleeps. With `echo` every command frame comes
    back on the queue one tick later, as the XCAN adapter echoes it.
    """

    start_deg: InitVar[Sequence[float]]
    t: float = 100.0
    rate: float = math.radians(120.0)
    grip_rate: float = 2.0
    grip_q: float = JAW_OPEN
    cube_pdes: float | None = None
    moving_effort: float = 0.3
    loaded_effort: float = 5.0
    on_tick: Callable[[], None] | None = None
    echo: bool = False
    echoes: list[CanFrame] = field(default_factory=list)
    q: list[float] = field(init=False)
    v: list[float] = field(default_factory=lambda: [0.0] * 7)
    eff: list[float] = field(default_factory=lambda: [0.0] * 7)
    p_des: list[float] | None = None
    g_des: float | None = None
    rx: deque = field(default_factory=deque)
    tx: list[Sent] = field(default_factory=list)
    closed: bool = False
    frames_out: int = 0
    _acc: float = 0.0
    _gave: bool = False

    def __post_init__(self, start_deg: Sequence[float]) -> None:
        self.q = [math.radians(x) for x in start_deg] + [self.grip_q]

    # ---- the CAN bus
    def recv(self, timeout: float = 0.0) -> CanFrame | None:
        if self.rx:
            self._gave = True
            return self.rx.popleft()
        if self._gave:
            self._gave = False
            return None
        self._gave = True
        return self._frame()

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        data = bytes(data)
        self.tx.append(Sent(can_id=can_id, data=data, fd=fd))
        if self.echo:
            self.echoes.append(CanFrame(can_id, data))
        if can_id == P.CMD_ID:
            self.p_des = [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]
        elif can_id == P.GRIP_ID:
            self.g_des = struct.unpack_from(">h", data, 0)[0] / P.POS_SCALE

    def close(self) -> None:
        self.closed = True

    # ---- the clock
    def now(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self._acc += max(float(secs), 0.0)
        while self._acc >= STEP - 1e-12:
            self._acc -= STEP
            self.t += STEP
            self._tick()

    # ---- what went out
    def frames(self, can_id: int) -> list[bytes]:
        return [s.data for s in self.tx if s.can_id == can_id]

    # ---- the arm
    def _tick(self) -> None:
        self._move_arm()
        self._move_jaw()
        self.rx.extend(self.echoes)
        self.echoes.clear()
        self.rx.append(self._frame())
        if self.on_tick is not None:
            self.on_tick()

    def _move_arm(self) -> None:
        if self.p_des is None:
            self.v[:6] = [0.0] * 6
            return
        step = self.rate * STEP
        new = [self.q[j] + max(-step, min(step, self.p_des[j] - self.q[j])) for j in range(6)]
        self.v[:6] = [(new[j] - self.q[j]) / STEP for j in range(6)]
        self.q[:6] = new

    def _move_jaw(self) -> None:
        if self.g_des is None:
            self.eff[6] = 0.0
            return
        target = self.g_des
        blocked = self.cube_pdes is not None and self.g_des > self.cube_pdes
        if blocked:
            target = self.cube_pdes
        step = self.grip_rate * STEP
        moved = max(-step, min(step, target - self.q[6]))
        self.q[6] += moved
        if blocked and abs(target - self.q[6]) < 1e-9:
            self.eff[6] = self.loaded_effort
        else:
            self.eff[6] = self.moving_effort if abs(moved) > 1e-9 else 0.0

    def _frame(self) -> CanFrame:
        self.frames_out += 1
        eff = list(self.eff)
        eff[0] += 0.002 * (self.frames_out % 2)            # live telemetry never repeats exactly
        return CanFrame(P.FB_ID, P.encode_feedback(self.q, self.v, eff))


def bench_station(bus: BenchBus, settings: dict) -> bench.Station:
    """The bench's real station on the fake bus, with the backend's exchange ticks run by the bench."""
    return bench.real_station(bus, settings, port="fake", clock=bus.now, sleep=bus.sleep, threaded=False)


@dataclass
class ScriptedKeys:
    """`Keys` with scripted answers. A bare Enter once armed sets `stop`, as the real reader does.
    `on_proceed` runs at the prompt that goes on (the operator puts the cube between the jaws)."""

    lines: list[str] = field(default_factory=list)
    confirms: bool = True
    proceeds: bool = True
    on_proceed: Callable[[], None] | None = None
    interactive: bool = False
    armed: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    prompts: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        pass

    def arm(self) -> None:
        self.armed = True

    def note_end(self) -> None:
        pass

    def confirm(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.confirms

    def proceed(self, prompt: str, word: str = bench.CONTINUE_WORD) -> bool:
        self.prompts.append(prompt)
        if self.on_proceed is not None:
            self.on_proceed()
        if not self.proceeds:
            self.stop.set()
        return self.proceeds

    def readline(self, timeout_s: float | None = None) -> str | None:
        with self._lock:
            if not self.lines:
                return None
            text = self.lines.pop(0)
        if text == "" and self.armed:
            self.stop.set()
        return text
