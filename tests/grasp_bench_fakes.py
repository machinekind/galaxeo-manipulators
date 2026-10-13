"""Fakes for grasp_bench.py: the A1X on a fake CAN bus with its own virtual clock, a wrist camera on that
clock, scripted keys and a policy that only times. Nothing here opens a link, a camera or the network."""

from __future__ import annotations

import math
import struct
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field

import numpy as np

import grasp_bench as gb
from galaxeo import protocol as P
from galaxeo.arm import Kinematics
from galaxeo.bus import Frame as CanFrame

STEP = 0.005                                   # the arm answers 0x052 every 5 ms
#: Jaw p_des of the default `a1x` config: open and closed (more positive = tighter).
JAW_OPEN = -1.95
JAW_CLOSED = 0.08


def jaw_pdes(opening: float) -> float:
    return JAW_CLOSED + opening / 100.0 * (JAW_OPEN - JAW_CLOSED)


@dataclass(frozen=True)
class Sent:
    can_id: int
    data: bytes
    fd: bool


@dataclass(frozen=True)
class J6Sample:
    t: float
    q6: float                                  # rad


@dataclass
class BenchBus:
    """An A1X on a CAN bus, on its own virtual clock that `sleep` advances in 5 ms ticks.

    A drain that finds the queue empty still gets one frame of the present state (the clock does not
    move for it), as a reader polling the hardware would: the backend's connect polls on wall time.
    Joints move toward p_des at `rate`. With `floor_z` the arm stops where the TCP would go below it
    (a table under the fingertips) and joint 2 then reads `press_effort`. The jaw moves toward its
    p_des at `grip_rate` with `moving_effort` on group 7; with `cube_pdes` it stops there when closing
    and reads `loaded_effort`.
    """

    start_deg: InitVar[Sequence[float]]
    t: float = 100.0
    rate: float = math.radians(120.0)
    grip_rate: float = 2.0
    grip_q: float = JAW_OPEN
    floor_z: float | None = None
    press_effort: float = 0.0
    cube_pdes: float | None = None
    moving_effort: float = 0.3
    loaded_effort: float = 5.0
    on_tick: Callable[[], None] | None = None
    q: list[float] = field(init=False)
    v: list[float] = field(default_factory=lambda: [0.0] * 7)
    eff: list[float] = field(default_factory=lambda: [0.0] * 7)
    p_des: list[float] | None = None
    g_des: float | None = None
    rx: deque = field(default_factory=deque)
    tx: list[Sent] = field(default_factory=list)
    history: list[J6Sample] = field(default_factory=list)
    closed: bool = False
    frames_out: int = 0
    _acc: float = 0.0
    _gave: bool = False
    _kin: Kinematics | None = None

    def __post_init__(self, start_deg: Sequence[float]) -> None:
        self.q = [math.radians(x) for x in start_deg] + [self.grip_q]
        self.history.append(J6Sample(t=self.t, q6=self.q[5]))
        if self.floor_z is not None:
            self._kin = Kinematics()

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

    def j6_at(self, t: float) -> float:
        q6 = self.history[0].q6
        for sample in self.history:
            if sample.t > t:
                break
            q6 = sample.q6
        return q6

    def tcp_z(self) -> float:
        kin = self._kin or Kinematics()
        return float(kin.fk(self.q[:6])[2, 3])

    # ---- the arm
    def _tick(self) -> None:
        self._move_arm()
        self._move_jaw()
        self.history.append(J6Sample(t=self.t, q6=self.q[5]))
        self.rx.append(self._frame())
        if self.on_tick is not None:
            self.on_tick()

    def _move_arm(self) -> None:
        self.eff[1] = 0.0
        if self.p_des is None:
            self.v[:6] = [0.0] * 6
            return
        step = self.rate * STEP
        new = [self.q[j] + max(-step, min(step, self.p_des[j] - self.q[j])) for j in range(6)]
        if self._kin is not None and self.floor_z is not None:
            z_new = float(self._kin.fk(new)[2, 3])
            z_now = float(self._kin.fk(self.q[:6])[2, 3])
            if z_new < self.floor_z and z_new < z_now:
                self.v[:6] = [0.0] * 6
                self.eff[1] = self.press_effort
                return
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


def bench_station(bus: BenchBus, settings: dict) -> gb.Station:
    """The bench's real station on the fake bus, with the backend's exchange ticks run by the bench."""
    return gb.real_station(bus, settings, port="fake", clock=bus.now, sleep=bus.sleep, threaded=False)


@dataclass
class WristCamera:
    """Frames every `period` on the bus clock, each showing joint 6 as it was `delay` before the frame
    arrived: a stripe pattern shifted with the angle. The raw frame is plain grey at `size`."""

    bus: BenchBus
    delay: float = 0.06
    period: float = 1.0 / 30.0
    size: tuple[int, int] = (1920, 1080)
    grey: int = 128
    closed: bool = False

    def latest(self) -> gb.Frame | None:
        k = math.floor((self.bus.now() - self.delay) / self.period)
        exposed = k * self.period
        if exposed < self.bus.history[0].t:
            return None
        x = np.arange(gb.FRAME_SMALL[0])[None, :]
        y = np.arange(gb.FRAME_SMALL[1])[:, None]
        image = 128.0 + 60.0 * np.sin(x / 6.0 + y / 9.0 + 40.0 * self.bus.j6_at(exposed))
        return gb.Frame(image=image.astype(np.uint8), t=exposed + self.delay, seq=k, size=self.size)

    def raw(self) -> gb.RawFrame | None:
        frame = self.latest()
        if frame is None:
            return None
        image = np.full((self.size[1], self.size[0], 3), self.grey, np.uint8)
        return gb.RawFrame(image=image, t=frame.t)

    def close(self) -> None:
        self.closed = True


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

    def proceed(self, prompt: str, word: str = gb.CONTINUE_WORD) -> bool:
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


@dataclass(frozen=True)
class Timing:
    first_ms: float
    median_ms: float
    p95_ms: float


@dataclass
class TimedPolicy:
    """A policy that only reports how long it takes."""

    timing: Timing

    def time(self, n: int = 200) -> Timing:
        return self.timing


def deps(bus: BenchBus | None, keys: ScriptedKeys, *, camera: WristCamera | None = None,
         policy: TimedPolicy | None = None, opened: list[gb.Station] | None = None) -> gb.Deps:
    """The bench's outside world as fakes; without a bus the default station (the simulated arm) runs.
    `opened` collects the stations the run opens."""
    def station(a, settings):
        st = bench_station(bus, settings) if bus is not None else gb.open_station(a, settings)
        if opened is not None:
            opened.append(st)
        return st

    return gb.Deps(open_station=station, keys=lambda: keys, open_camera=lambda source, clock: camera,
                   load_policy=lambda path, sha256: policy)
