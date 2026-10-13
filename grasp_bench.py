#!/usr/bin/env python3
"""Bench measurements for the grasp student on the A1X: jaw, table, grasp point, latency, dot, summary.

    python3 grasp_bench.py gripper --i-know --rig rig.json             # jaw stroke and effort at >= 150 Hz
    python3 grasp_bench.py gripper --rig rig.json --set-freeze 100     # record the freeze decision
    python3 grasp_bench.py gripper --rig rig.json --gap-mm 0:3 100:103 # ruler readings of the tip gap
    python3 grasp_bench.py table --i-know --rig rig.json --xy 0 0.33   # fingertips onto the table
    python3 grasp_bench.py point --i-know --rig rig.json --xy 0 0.33 --yaw-deg 0
    python3 grasp_bench.py latency --i-know --rig rig.json --student student.pt --camera 0
    python3 grasp_bench.py dot --rig rig.json --error-px 3.5           # the dot error read from a strategy dump
    python3 grasp_bench.py show --rig rig.json                         # the rig file and what each mode misses
    python3 grasp_bench.py table --sim                                 # a rehearsal on the simulated arm

The rig file is `galaxeo.grasp.rig` format 1 at the path given with --rig, the file the strategy's
`rig_json` setting names. Nothing is written without --rig, so values measured on the simulated arm
reach a file only when --rig names one. Runbook: docs/GRASP.md.

One owner of the CAN link at a time. galaxeo.arm moves the arm: `Arm.move` on a BusTransport (measured
start, ramp, impact guard, backoff). galaxeo.arm has no jaw, so the `a1x` backend drives it. Both read
0x052 from the same queue, and the backend takes any 0x050 it did not send for another sender, so they
never run together: the backend is connected only for a jaw phase (`BackendLink.phase`: the whole
`gripper` tool, opening the jaw at the start pose, `g` while jogging, the latency steps) and the
BusTransport sends and reads nothing meanwhile. Between phases the arm holds its last p_des and the
jaw its last 0x051. The latency steps go through the backend on purpose: it is the path the
strategy's commands take, and galaxeo.arm's ramp starts too gently to time a camera against.

Safety (docs/SAFETY.md): anything that moves the real arm needs --i-know and an Enter; from then on a
bare Enter is STOP, and so is Ctrl-C. A STOP during a move holds the measured pose (`Motion`), a STOP
in a jaw phase stops the jaw where its stream is and holds the measured arm pose if the phase drives
the arm. Nothing sends a release frame. Teaching is jogging under position control: small top-down
Cartesian steps solved with `Kinematics.ik_step`, each one a guarded move. Three rules watch for the
table (measured height, joint lag, effort) and the first to fire holds the measured pose and backs
off 2 mm.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np

from galaxeo import protocol as P
from galaxeo.arm import JOINTS, Arm, BusTransport, Kinematics, MoveResult, Reading, SimTransport
from galaxeo.arm.guard import ImpactGuard
from galaxeo.bus import default_iface, open_bus
from galaxeo.grasp.contract import (
    ARRIVE_M,
    CHECKPOINT_SHA256,
    DOT_MAX_PX,
    FREEZE_DISABLED,
    GAP_CLOSED,
    GAP_PER_UNIT,
    HOVER,
    HOVER_J5_MAX_DEG,
    HOVER_J5_MEASURED_MAX_DEG,
    HOVER_J6_MARGIN_DEG,
    IK_MAX_JUMP_DEG,
    LATENCY_MAX_S,
    OBJECT_HEIGHT_DEFAULT,
    PREPOSITION_SPEED_DEG_S,
    RAW_H,
    RAW_W,
    RETREAT_BELOW,
    RETREAT_CLEAR,
    RETREAT_NEAR_XY,
    RETREAT_STEP,
    RETREAT_TILT_MAX_DEG,
    SAG_LIMIT,
    SAG_MAX,
    TABLE_TOUCH_FAR,
    TEACH_MAX_AGE_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TOUCH_TOLERANCE,
    TRAIN_HOVER_Z,
    TRAIN_J1_DEG,
    TRAIN_RADIUS,
    grasp_R,
    wrap,
    yaw_of,
)
from galaxeo.grasp.rig import MODES, OBJECT_HEIGHT_RANGE, Rig, RigError
from galaxeo.robot.backend import GRIP_START_DELTA, A1XArm, A1XConfig, GripperState

logger = logging.getLogger("grasp_bench")

GRIPPER = "gripper"
J6 = JOINTS[5]

# ---- keyboard
CONTINUE_WORD = "d"
ENTER_GUARD_S = 1.0                   # a line this soon after a move ended was typed as STOP
KEYS_POLL_S = 0.1                     # waits wake up this often, so Ctrl-C gets through

# ---- the link
FEEDBACK_WAIT_S = 1.0
FEEDBACK_PERIOD_S = 0.01              # two 0x052 frames
FOREIGN_LISTEN_S = 0.3
JAW_TICK_S = 0.005                    # one exchange tick of the backend, when the caller runs them
STOP_POLL_S = 0.05

# ---- moves
SEGMENT_DEG = 40.0                    # galaxeo.arm refuses 45 deg in one move; room for the measured start
MOVE_LAG_DEG = 4.0
SAG_AVERAGE_S = 0.3
POLL_S = 0.05

# ---- jaw
GRIP_OPEN = 100.0
GRIP_WAIT_S = 2.0
GRIP_ARRIVED = 1.0                    # units between stream and target that count as there
GRIP_MEASURED_TOL = 3.0               # units between the measured jaw and the target that count as there
GRIP_NUDGE_S = 0.1
#: 99 first: a first command within 0.5 of the opening the backend reports (100) sends nothing.
GRIPPER_FREE_MOVES = (99.0, 100.0, 0.0, 100.0, 0.0, 100.0)
GRIPPER_LOADED_MOVES = (0.0, 100.0)
GRIPPER_SETTLE_S = 2.0                # after each command; the A1X stream needs 1.0 to 1.3 s for a stroke
PRE_MOVE_S = 0.5                      # logged before each command: the noise band of its effort
GRIPPER_POLL_S = 0.001
MIN_RATE_HZ = 150.0
NOISE_SIGMAS = 3.0
EFFORT_NOISE_MIN = 0.05               # backend effort units; a flat trace would give a zero band
STROKE_MIN_UNITS = 50.0               # a move shorter than this is not a stroke
STROKE_EDGE_UNITS = 1.0               # start and end of a stroke: this far from its end values
SEPARABLE_RATIO = 1.5
GAP_WARN_MM = 3.0
OPERATOR_MIN_S = 0.05                 # an operator line sooner than this after a command was typed before it
SIM_JAW_EFFORT = 0.3                  # the simulated jaw's effort while its stream moves
SIM_JAW_NOISE = 0.01
SIM_FEED_S = 0.005                    # the simulated jaw reports at the A1X's 200 Hz

# ---- jog
JOG_STEP_M = 0.005
JOG_SLOW_STEP_M = 0.001               # `d` below JOG_SLOW_BELOW_Z
JOG_SLOW_BELOW_Z = 0.06               # m, measured TCP z; below it the height rule watches
JOG_STEP_YAW = math.radians(5.0)
JOG_SPEED_DEG_S = 10.0
#: ik_step tolerances of the jog. Training's 1 mm would take a 1 mm step as solved at its seed.
JOG_TOL_POS = 1e-4
JOG_TOL_ROT = 1e-3
JOG_GUARD_LAG_DEG = 3.0
JOG_LAG_DEG = 1.5                     # further behind than at the start of the step
JOG_CONTACT_MIN_DROP = 0.0004         # m over the last two 1 mm steps
JOG_BACKOFF = 0.002
JOG_SETTLE_S = 0.3
JOG_POLL_S = 0.01
#: Below this commanded TCP z the collision checker would refuse the touch pose itself.
JOG_COLLISION_ABOVE_Z = 0.04
AVERAGE_S = 0.5
START_Z_DEFAULT = 0.12
START_XY_DEFAULT = (0.0, 0.33)        # training's region: the arm's left, about 0.33 m out

# ---- latency
LATENCY_STEPS = 5
LATENCY_STEP_DEG = 3.0
LATENCY_MOVED_DEG = 0.2
LATENCY_SIGMAS = 4.0
LATENCY_STILL_FRAMES = 8
LATENCY_STILL_S = 1.0
LATENCY_STEP_TIMEOUT_S = 1.5
LATENCY_SETTLE_S = 0.6
LATENCY_POLL_S = 0.005
LATENCY_MIN_GOOD = 3
#: Grey levels. Identical still frames give sigma 0, and a larger floor hides a slow start of the motion.
MAD_SIGMA_MIN = 0.01
FIRST_FRAME_S = 3.0
FRAME_WAIT_S = 0.02
GUARD_LAG_S = 0.15
FRAME_SMALL = (160, 90)               # the frame difference runs on a grey frame this size


class Refusal(Exception):
    """Nothing moved: exit code 2."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class Ended(Exception):
    """A tool ended before its result: exit code 1."""

    def __init__(self, ending: str, detail: str = "") -> None:
        super().__init__(f"{ending}: {detail}" if detail else ending)
        self.ending = ending
        self.detail = detail


# ====================================================================== keyboard
def _isatty(stream: TextIO | None) -> bool:
    try:
        return stream is not None and bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


@dataclass(frozen=True)
class Line:
    """One line from stdin, stripped, with its arrival time."""

    t: float
    text: str


class Keys:
    """The one stdin reader of a run. Once armed, a bare Enter is always STOP.

    A jaw stroke ends by itself, so an Enter meant as STOP can arrive just after it. Going on therefore
    needs a typed word, typed after the prompt and more than ENTER_GUARD_S after the ending (`proceed`);
    everything else stops.
    """

    def __init__(self, stdin: TextIO | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        # None = `sys.stdin` as it is now, not as it was at import (callers replace it).
        stdin = sys.stdin if stdin is None else stdin
        self._stdin = stdin
        self._clock = clock
        self._tty = _isatty(stdin)
        self._cond = threading.Condition()
        self._lines: deque[Line] = deque()
        self._eof = stdin is None
        self._armed = False
        self._t_end: float | None = None
        self._thread: threading.Thread | None = None
        self.stop = threading.Event()

    @property
    def interactive(self) -> bool:
        with self._cond:
            return self._tty and not self._eof

    @property
    def armed(self) -> bool:
        with self._cond:
            return self._armed

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._eof:
                return
            self._thread = threading.Thread(target=self._read, name="chwyt-klawiatura", daemon=True)
        self._thread.start()

    def arm(self) -> None:
        """From now on a bare Enter sets `stop`; call right before the first motion."""
        with self._cond:
            self._armed = True
            # An Enter typed after the last prompt was answered is already a STOP.
            if any(line.text == "" for line in self._lines):
                self.stop.set()

    def note_end(self) -> None:
        """A move has just ended: input within ENTER_GUARD_S from now is a STOP."""
        with self._cond:
            self._t_end = self._clock()

    def confirm(self, prompt: str) -> bool:
        """Before the first motion only: True on a bare Enter typed after the prompt."""
        if self.armed:
            raise RuntimeError("Keys.confirm after arm(): only proceed() may continue once motion has started")
        if not self.interactive:
            return False
        self.start()
        with self._cond:
            self._lines.clear()
        self._print(prompt)
        t_prompt = self._clock()
        while True:
            line = self._next(None)
            if line is None:
                return False
            if line.t >= t_prompt:
                return line.text == ""

    def proceed(self, prompt: str, word: str = CONTINUE_WORD) -> bool:
        """True only for `word` + Enter typed after the prompt and outside the guard window; else STOP."""
        if self.stop.is_set():
            return self._refuse("STOP already given")
        if not self.interactive:
            return self._refuse("stdin is not interactive")
        self.start()
        with self._cond:
            if self._lines:
                self._lines.clear()
                return self._refuse("input typed before the prompt")
        self._print(prompt)
        t_prompt = self._clock()
        line = self._next(None)
        if line is None:
            return self._refuse("no answer")
        with self._cond:
            t_end = self._t_end
        if line.t < t_prompt:
            return self._refuse("input typed before the prompt")
        if t_end is not None and line.t - t_end <= ENTER_GUARD_S:
            return self._refuse("input within the guard window after an ending")
        if line.text != word:
            return self._refuse(f"answer {line.text!r} is not {word!r}")
        return True

    def readline(self, timeout_s: float | None = None) -> str | None:
        """The next line, stripped; None on timeout or end of file. An empty line is a STOP when armed."""
        self.start()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        line = self._next(deadline)
        if line is None:
            return None
        if line.text == "" and self.armed:
            self.stop.set()
        return line.text

    def _refuse(self, why: str) -> bool:
        logger.info("keys: STOP (%s)", why)
        self.stop.set()
        return False

    @staticmethod
    def _print(prompt: str) -> None:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    def _next(self, deadline: float | None) -> Line | None:
        with self._cond:
            while not self._lines:
                if self._eof:
                    return None
                wait = KEYS_POLL_S
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0.0:
                        return None
                    wait = min(wait, left)
                self._cond.wait(wait)
            return self._lines.popleft()

    def _read(self) -> None:
        while True:
            try:
                raw = self._stdin.readline()
            except (OSError, ValueError, UnicodeDecodeError):
                raw = ""
            if raw == "":
                with self._cond:
                    self._eof = True
                    self._cond.notify_all()
                return
            text = raw.strip()
            with self._cond:
                if text == "" and self._armed:
                    self.stop.set()
                self._lines.append(Line(t=self._clock(), text=text))
                self._cond.notify_all()


# ====================================================================== the link
class BorrowedBus:
    """The bench's CAN link lent to the backend for one jaw phase.

    The backend closes its link on disconnect; the bench closes the real one once, at the end.
    """

    def __init__(self, bus: Any) -> None:
        self._bus = bus

    @property
    def state(self) -> str:
        return getattr(self._bus, "state", "ok")

    def recv(self, timeout: float = 0.0) -> Any:
        return self._bus.recv(timeout)

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        self._bus.send(can_id, data, fd=fd)

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class BenchContext:
    """The fields of the twin's backend context that the `a1x` backend reads, filled by the bench."""

    joints: tuple[str, ...]
    gripper: str
    limits: Mapping[str, tuple[float, float]]
    port: str | None
    settings: Mapping[str, Any]


def bench_context(port: str | None, settings: Mapping[str, Any]) -> BenchContext:
    limits = {name: (math.degrees(lo_hi[0]), math.degrees(lo_hi[1]))
              for name, lo_hi in zip(JOINTS, P.URDF_LIMITS_RAD)}
    limits[GRIPPER] = (0.0, 100.0)
    return BenchContext(joints=(*JOINTS, GRIPPER), gripper=GRIPPER, limits=limits, port=port,
                        settings=dict(settings))


@dataclass(frozen=True)
class JointSample:
    """The arm joints as the backend last measured them."""

    t: float                          # station clock when the 0x052 frame was drained
    joints: dict[str, float]          # arm joints [deg, app units]
    fresh: bool


def _stream_on_target(gs: GripperState | None) -> bool:
    return gs is not None and gs.streamed is not None and gs.target is not None and abs(gs.streamed - gs.target) < 1e-6


def until_streamed(jaw: BackendLink | SimJaw, opening: float, stopped: threading.Event) -> None:
    """Waits until the jaw stream stands on its target (`opening`, or where a contact froze it), at most
    GRIP_WAIT_S: a phase that ends earlier leaves the jaw where the stream was."""
    t_end = jaw.clock() + GRIP_WAIT_S
    while jaw.clock() < t_end and not stopped.is_set():
        if _stream_on_target(jaw.gripper_state()):
            return
        jaw.wait(JOG_POLL_S)
    if not stopped.is_set():
        logger.warning("jaw stream did not reach %.0f within %.1f s", opening, GRIP_WAIT_S)


class BackendLink:
    """The `a1x` backend on the bench's link, connected only for a jaw phase (see the module docstring)."""

    def __init__(self, backend: A1XArm, *, clock: Callable[[], float], sleep: Callable[[float], None],
                 threaded: bool) -> None:
        self.backend = backend
        self.clock = clock
        self._sleep = sleep
        self.threaded = threaded
        #: Where the jaw stream stood when the last phase ended; the next phase starts the backend from it.
        self.opening: float | None = None
        self._active = threading.Event()
        self._drives_arm = False

    @contextmanager
    def phase(self) -> Iterator[BackendLink]:
        try:
            self.backend.connect()
        except (RuntimeError, OSError) as exc:
            raise Ended("link", str(exc)) from exc
        self._drives_arm = False
        self._active.set()
        try:
            if self.opening is not None:
                # A new connection guesses the jaw from `gripper_start`: its first-command gate would
                # drop a close from a wrong guess, or the stream would start from the wrong end.
                self.backend.assume_gripper(self.opening)
            yield self
        finally:
            self._active.clear()
            gs = self.backend.gripper_state()
            if gs is not None and gs.streamed is not None:
                self.opening = gs.streamed
            self.backend.disconnect()

    def wait(self, secs: float) -> None:
        """`secs` on the station clock; without the backend's IO thread its exchange ticks run here."""
        if self.threaded:
            self._sleep(secs)
            return
        end = self.clock() + secs
        while True:
            left = end - self.clock()
            if left <= 1e-12:
                return
            self._sleep(min(JAW_TICK_S, left))
            self.backend.io_step(self.clock())

    def gripper_state(self) -> GripperState | None:
        return self.backend.gripper_state()

    def foreign_frames(self) -> int:
        return self.backend.foreign_frames()

    def command(self, opening: float) -> None:
        """One jaw target through the backend's gate, as the twin sends it."""
        self.backend.send_joints({GRIPPER: float(opening)})

    def grip(self, opening: float, stopped: threading.Event) -> None:
        """The jaw to `opening`, waiting until the stream is there.

        Before the jaw is known (the first phase of a run) only GRIP_OPEN is asked: the backend opens
        a guessed jaw straight to the target, and closing from a guess could snap it shut.
        """
        if self.opening is None and opening != GRIP_OPEN:
            raise ValueError("the first jaw command of a run opens the jaw")
        if self._gate_drops(opening):
            if self.opening is not None:
                return
            self.command(GRIP_OPEN - 1.0)
            until = self.clock() + GRIP_NUDGE_S
            while self.clock() < until and not stopped.is_set():
                self.wait(JOG_POLL_S)
        self.command(opening)
        until_streamed(self, opening, stopped)
        self._until_jaw_there(opening, stopped)

    def _until_jaw_there(self, opening: float, stopped: threading.Event) -> None:
        """Keeps the stream on until the measured jaw is at `opening`, at most one full stroke: a first
        command jumps the stream to its target, and the jaw only gets there while 0x051 flows."""
        end = self.clock() + self.backend.gripper_travel_s()
        while self.clock() < end and not stopped.is_set():
            gs = self.gripper_state()
            if gs is not None and gs.position is not None and abs(gs.position - opening) <= GRIP_MEASURED_TOL:
                return
            self.wait(JOG_POLL_S)

    def _gate_drops(self, opening: float) -> bool:
        gs = self.backend.gripper_state()
        if gs is None or gs.commanded is not None:
            return False
        return abs(opening - self.backend.read_joints()[GRIPPER]) <= GRIP_START_DELTA

    def send(self, targets: Mapping[str, float]) -> None:
        """Arm and jaw targets [app units] through the backend, as the twin sends them."""
        if any(j in targets for j in JOINTS):
            self._drives_arm = True
        self.backend.send_joints(dict(targets))

    def app_degrees(self, q: Sequence[float]) -> dict[str, float]:
        """Wire radians -> the backend's app degrees (minus the zero offsets)."""
        offsets = self.backend.acfg.joint_offsets_deg
        return {j: math.degrees(float(q[i])) - offsets[i] for i, j in enumerate(JOINTS)}

    def sample(self) -> JointSample | None:
        gs = self.backend.gripper_state()
        if gs is None or gs.age_s is None:
            return None
        joints = self.backend.read_joints()
        fresh = gs.age_s * 1000.0 <= self.backend.acfg.stale_ms
        return JointSample(t=gs.t, joints={j: joints[j] for j in JOINTS}, fresh=fresh)

    def efforts(self) -> dict[str, float] | None:
        return self.backend.efforts()

    def velocities(self) -> dict[str, float] | None:
        return self.backend.velocities()

    def hold(self) -> None:
        """Stops the jaw where its stream is and, when this phase drives the arm, holds the measured arm pose."""
        if not self._active.is_set():
            return
        try:
            targets: dict[str, float] = {}
            if self._drives_arm:
                joints = self.backend.read_joints()
                targets.update({j: joints[j] for j in JOINTS})
            gs = self.backend.gripper_state()
            if (gs is not None and gs.streamed is not None and gs.commanded is not None
                    and abs(gs.streamed - gs.commanded) > GRIP_ARRIVED):
                targets[GRIPPER] = gs.streamed
            if targets:
                self.backend.send_joints(targets)
        except RuntimeError:
            logger.exception("holding through the backend failed")


class SimJaw:
    """The jaw of a simulated arm. galaxeo.arm's model drives no jaw, so this follows the commands at
    the A1X backend's default stream rate on the sim clock and reports a GripperState like it."""

    def __init__(self, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self.clock = clock
        self._sleep = sleep
        cfg = A1XConfig()
        self.rate = cfg.gripper_rate_pdes_s * 100.0 / abs(cfg.gripper_open_pdes - cfg.gripper_closed_pdes)
        self.opening: float | None = None
        self._commanded: float | None = None
        self._streamed: float | None = None
        self._t = clock()

    @contextmanager
    def phase(self) -> Iterator[SimJaw]:
        yield self

    def wait(self, secs: float) -> None:
        self._sleep(secs)

    def foreign_frames(self) -> int:
        return 0

    def command(self, opening: float) -> None:
        self._advance()
        self._commanded = float(opening)
        if self._streamed is None:
            # The backend sends a guessed jaw straight to its first target.
            self._streamed = float(opening)

    def grip(self, opening: float, stopped: threading.Event) -> None:
        self.command(opening)
        until_streamed(self, opening, stopped)

    def hold(self) -> None:
        self._advance()
        if self._streamed is not None:
            self._commanded = self._streamed

    def gripper_state(self) -> GripperState:
        self._advance()
        now = self.clock()
        feed = math.floor(now / SIM_FEED_S)
        moving = self._commanded is not None and abs(self._commanded - self._streamed) > 1e-9
        effort = (SIM_JAW_EFFORT if moving else 0.0) + SIM_JAW_NOISE * (feed % 2)
        return GripperState(commanded=self._commanded, target=self._commanded, streamed=self._streamed,
                            position=None, frozen=False, effort=effort, t=feed * SIM_FEED_S,
                            age_s=now - feed * SIM_FEED_S, rate=self.rate, force_threshold=FREEZE_DISABLED,
                            measured=False)

    def _advance(self) -> None:
        now = self.clock()
        dt = max(now - self._t, 0.0)
        self._t = now
        if self._commanded is None or self._streamed is None:
            return
        step = self.rate * dt
        self._streamed += max(-step, min(step, self._commanded - self._streamed))
        self.opening = self._streamed


@dataclass
class Station:
    """One run's arm: galaxeo.arm's guarded moves on a transport, the jaw, one clock for both."""

    arm: Arm
    jaw: BackendLink | SimJaw
    simulated: bool
    bus: Any = None                   # the real link, closed once at the end
    stopped: threading.Event = field(default_factory=threading.Event)

    def now(self) -> float:
        return self.arm.transport.now()

    def wait(self, secs: float) -> None:
        self.arm.transport.sleep(secs)

    def reading(self) -> Reading | None:
        """A fresh reading of the arm, or None."""
        r, _ = self.arm.motion.fresh()
        return r

    def settled_reading(self) -> Reading | None:
        """A reading drained twice one feedback period apart: after an idle gap the first drain can end
        on a frame the adapter queued long ago."""
        self.reading()
        self.wait(FEEDBACK_PERIOD_S)
        return self.reading()

    def wait_feedback(self, secs: float) -> Reading:
        end = self.now() + secs
        while True:
            r = self.reading()
            if r is not None:
                return r
            if self.now() >= end:
                raise Ended("no feedback", f"brak swiezych ramek 0x052 przez {secs:.0f} s - lacze CAN stoi "
                                           f"albo ramie bez zasilania")
            self.wait(FEEDBACK_PERIOD_S)

    def move(self, q: np.ndarray, speed: float, *, profile: str = "normal",
             lag_deg: float | None = None) -> MoveResult:
        """One guarded move on its own thread, so that Ctrl-C here reaches `Arm.stop` and the move holds
        the measured pose instead of leaving its last ramp command on the arm."""
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["result"] = self.arm.move(q, speed, profile=profile, lag_deg=lag_deg)
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=run, name="chwyt-ruch", daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                worker.join(STOP_POLL_S)
        except KeyboardInterrupt:
            self.arm.stop("stop")
            worker.join(5.0)
            raise
        if "error" in box:
            raise box["error"]
        return box["result"]

    def hold_measured(self) -> np.ndarray | None:
        """One p_des frame at the measured pose, between moves: a joint pressed into the table stops
        pushing. The A1X keeps the last p_des."""
        r = self.reading()
        if r is None:
            return None
        self.arm.transport.write(r.q)
        return r.q.copy()

    def stop(self) -> None:
        """Any thread: a move ends holding the measured pose, the jaw stops where its stream is."""
        self.stopped.set()
        self.arm.stop("stop")
        self.jaw.hold()

    def close(self) -> None:
        if self.bus is not None:
            self.bus.close()


def real_station(bus: Any, settings: Mapping[str, Any], *, port: str | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 threaded: bool = True) -> Station:
    """The A1X on `bus`: a BusTransport for galaxeo.arm and the `a1x` backend for the jaw, on one link."""
    backend = A1XArm(bench_context(port, settings), bus=BorrowedBus(bus), clock=clock, sleep=sleep,
                     threaded=threaded)
    transport = BusTransport(bus, tx=True, clock=clock, sleep=sleep, kp=backend.acfg.kp, kd=backend.acfg.kd)
    link = BackendLink(backend, clock=clock, sleep=sleep, threaded=threaded)
    return Station(arm=Arm(transport), jaw=link, simulated=False, bus=bus)


def sim_station() -> Station:
    transport = SimTransport()
    return Station(arm=Arm(transport), jaw=SimJaw(transport.now, transport.sleep), simulated=True)


def open_station(a: argparse.Namespace, settings: Mapping[str, Any]) -> Station:
    if a.sim:
        return sim_station()
    try:
        bus = open_bus(a.iface)
    except Exception as exc:
        raise Ended("link", f"nie otwarto lacza {a.iface}: {exc}") from exc
    try:
        return real_station(bus, settings, port=a.iface)
    except (TypeError, ValueError) as exc:
        bus.close()
        raise Refusal("a1x config", str(exc)) from exc


def a1x_settings(path: str | None) -> dict[str, Any]:
    """The `a1x:` section of the twin's YAML config, so the jaw runs as it does in the twin."""
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise Refusal("config", "--config potrzebuje PyYAML (pip install pyyaml)") from exc
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Refusal("config", f"{path}: {exc}") from exc
    section = data.get("a1x") if isinstance(data, dict) else None
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise Refusal("config", f"{path}: sekcja a1x musi byc mapowaniem")
    return dict(section)


class StopWatch:
    """Turns a STOP from the keyboard (a bare Enter once armed, a refused prompt) into `Station.stop`."""

    def __init__(self, keys: Keys, station: Station) -> None:
        self.keys = keys
        self.station = station
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._fired = False
        self._thread = threading.Thread(target=self._run, name="chwyt-stop", daemon=True)

    def start(self) -> StopWatch:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._done.is_set():
            if self.keys.stop.wait(STOP_POLL_S):
                self.fire()
                return

    def fire(self) -> None:
        with self._lock:
            if self._fired:
                return
            self._fired = True
        print("\nSTOP - ramie trzyma zmierzona poze.", flush=True)
        self.station.stop()

    def close(self) -> None:
        """A STOP typed just before the tool ended is still carried out."""
        self._done.set()
        if self.keys.stop.is_set():
            self.fire()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(1.0)


def require_consent(a: argparse.Namespace) -> None:
    if not a.sim and not a.i_know:
        raise Refusal("i-know", f"grasp_bench.py {a.tool} na {a.iface} rusza prawdziwym ramieniem - dodaj --i-know")


def confirm_motion(station: Station, keys: Keys, prompt: str) -> None:
    """The operator's Enter before the first motion on a real arm; then Enter means STOP."""
    if not station.simulated and not keys.confirm(prompt):
        raise Refusal("not confirmed", "bez potwierdzenia Enter (potrzebny terminal)")
    keys.arm()
    station.arm.set_armed(True)


@contextmanager
def operated(station: Station, keys: Keys) -> Iterator[None]:
    """The part of a tool in which the arm may move: Enter and Ctrl-C both STOP."""
    watch = StopWatch(keys, station).start()
    try:
        yield
    except KeyboardInterrupt:
        station.stop()
        raise Ended("stop", "Ctrl-C") from None
    finally:
        watch.close()


def listen_for_other_senders(station: Station) -> None:
    """On a real arm: listen FOREIGN_LISTEN_S for command frames of another sender before anything moves."""
    if station.simulated:
        return
    with station.jaw.phase():
        station.jaw.wait(FOREIGN_LISTEN_S)
        n = station.jaw.foreign_frames()
    if n:
        raise Refusal("busy", f"inny nadawca na laczu CAN ({n} ramek rozkazu): zamknij panel blizniaka, "
                              f"jog_a1x.py, so101_bridge.py, sterownik ROS")


# ====================================================================== poses and moves
@dataclass(frozen=True)
class Pose:
    """A top-down tool pose and the arm joints that hold it."""

    p: np.ndarray                     # TCP, base frame [m]
    yaw: float                        # closing axis about +z [rad]: the asked yaw or the asked yaw + pi
    q: np.ndarray                     # arm joints [rad]
    j6_margin_deg: float


@dataclass(frozen=True)
class _Candidate:
    yaw: float
    q: np.ndarray
    failed: list[str]
    j6_margin_deg: float


def _candidate(kin: Kinematics, p: np.ndarray, yaw: float) -> _Candidate:
    R = grasp_R(yaw)
    # The full IK from home keeps J5 at zero, as in every training hover; the warm step then meets
    # training's tolerances (1 mm, 5 mrad), so the policy starts from joints it knows.
    rough = kin.ik(p, R, seed=kin.home)
    step = kin.ik_step(p, R, rough.q)

    failed: list[str] = []
    if not step.converged:
        failed.append(f"pozycja {step.pos_err * 1000:.2f} mm, obrot {step.rot_err * 1000:.2f} mrad")
    j5 = math.degrees(step.q[4])
    if abs(j5) > HOVER_J5_MAX_DEG:
        failed.append(f"J5 {j5:.2f} st.")
    j6 = math.degrees(step.q[5])
    margin = min(j6 - math.degrees(kin.lo[5]), math.degrees(kin.hi[5]) - j6)
    if margin < HOVER_J6_MARGIN_DEG:
        failed.append(f"zapas J6 {margin:.1f} st.")
    return _Candidate(yaw=float(yaw), q=step.q.copy(), failed=failed, j6_margin_deg=float(margin))


def solve_pose(kin: Kinematics, p: Sequence[float], yaw: float) -> Pose:
    """The tool pointing down at `p` with the closing axis at `yaw` or `yaw + pi` (the pinch is
    symmetric, the joints are not: J6 differs by 180 deg); the one with more room on J6 wins."""
    point = np.asarray(p, float).reshape(3)
    if not (np.all(np.isfinite(point)) and math.isfinite(yaw)):
        raise Refusal("ik", f"nieskonczone: punkt {point.tolist()}, obrot {yaw}")
    candidates = [_candidate(kin, point, wrap(float(yaw))), _candidate(kin, point, wrap(float(yaw) + math.pi))]
    valid = [c for c in candidates if not c.failed]
    if not valid:
        raise Refusal("ik", "; ".join(f"obrot {math.degrees(c.yaw):.0f} st.: {', '.join(c.failed)}"
                                      for c in candidates))
    best = max(valid, key=lambda c: c.j6_margin_deg)
    return Pose(p=point.copy(), yaw=best.yaw, q=best.q, j6_margin_deg=best.j6_margin_deg)


def training_envelope(pose: Pose) -> list[str]:
    """Which checks of the training envelope a hover fails, in Polish; empty when inside."""
    out: list[str] = []
    j1 = math.degrees(pose.q[0])
    if not TRAIN_J1_DEG[0] <= j1 <= TRAIN_J1_DEG[1]:
        out.append(f"J1 {j1:.1f} st. poza {TRAIN_J1_DEG[0]:g}..{TRAIN_J1_DEG[1]:g}")
    radius = math.hypot(pose.p[0], pose.p[1])
    if not TRAIN_RADIUS[0] <= radius <= TRAIN_RADIUS[1]:
        out.append(f"promien {radius:.2f} m poza {TRAIN_RADIUS[0]:g}..{TRAIN_RADIUS[1]:g}")
    z = float(pose.p[2])
    if not TRAIN_HOVER_Z[0] <= z <= TRAIN_HOVER_Z[1]:
        out.append(f"wysokosc zawisu {z:.3f} m poza {TRAIN_HOVER_Z[0]:g}..{TRAIN_HOVER_Z[1]:g}")
    return out


@dataclass(frozen=True)
class Route:
    """The moves to a pose: 5 mm steps straight up when the tool is low near it, then one
    collision-checked joint-space line cut into segments of at most SEGMENT_DEG."""

    waypoints: list[np.ndarray]
    rise: int                         # how many waypoints are steps straight up
    travel_deg: float                 # the largest joint travel of the line


def rise_steps(kin: Kinematics, q0: np.ndarray, pose: Pose, clear_z: float) -> list[np.ndarray]:
    """Joints of 5 mm steps straight up to the pose's height, when the tool points down, low and near it.

    The tool keeps its measured orientation: squaring a few degrees of tilt in the first step needs more
    than IK_MAX_JUMP_DEG from one joint in 10 to 25 % of low poses and swings the fingertips sideways
    past the object. The joint-space line squares it at the pose's height.
    """
    T = kin.fk(q0)
    p = T[:3, 3].copy()
    tilt = math.degrees(math.acos(min(1.0, max(-1.0, -float(T[2, 0])))))
    near = math.hypot(p[0] - pose.p[0], p[1] - pose.p[1]) < RETREAT_NEAR_XY
    if not (tilt <= RETREAT_TILT_MAX_DEG and p[2] < pose.p[2] - RETREAT_BELOW and near):
        return []

    R = T[:3, :3].copy()
    steps: list[np.ndarray] = []
    q = q0
    z = float(p[2])
    while z < pose.p[2] - 1e-9:
        reached = z
        z = min(z + RETREAT_STEP, float(pose.p[2]))
        step = kin.ik_step([p[0], p[1], z], R, q, tol_pos=JOG_TOL_POS, tol_rot=JOG_TOL_ROT)
        if not step.converged or math.degrees(step.jump) > IK_MAX_JUMP_DEG:
            # At long reach a vertical line leaves the warm solver's branch before the pose's height;
            # well above the object the collision-checked line can take over.
            if reached >= clear_z:
                return steps
            raise Refusal("ik", f"krok w gore do z {z:.3f} m odrzucony (blad {step.pos_err * 1000:.2f} mm, "
                                f"{step.rot_err * 1000:.2f} mrad, skok {math.degrees(step.jump):.1f} st.)")
        steps.append(step.q.copy())
        q = step.q
    return steps


def clear_height(rig: Rig) -> float:
    """TCP z above which the jaws clear the object: a refused step up above it is no reason to stop."""
    touch = rig.touch_tcp_z()
    if touch is None:
        return math.inf
    height = rig.object_height()
    return touch + (OBJECT_HEIGHT_DEFAULT if height is None else height) + RETREAT_CLEAR


def plan_route(station: Station, pose: Pose, clear_z: float) -> Route:
    reading = station.settled_reading()
    if reading is None:
        raise Ended("stale feedback", "brak swiezych ramek 0x052")
    rise = rise_steps(station.arm.kin, reading.q, pose, clear_z)
    start = rise[-1] if rise else reading.q.copy()

    why = station.arm.collision.path_clear(start, pose.q)
    if why is not None:
        raise Refusal("collision", f"{why}; ustaw ramie blizej punktu (panel blizniaka albo jog_a1x.py)")

    travel = math.degrees(float(np.max(np.abs(pose.q - start))))
    n = max(1, math.ceil(travel / SEGMENT_DEG))
    segments = [start + (k / n) * (pose.q - start) for k in range(1, n + 1)]
    return Route(waypoints=[*rise, *segments], rise=len(rise), travel_deg=travel)


def describe(route: Route) -> str:
    rise = f"{route.rise} krokow w gore, potem " if route.rise else ""
    return f"{rise}{len(route.waypoints) - route.rise} ruchow, najwiekszy staw {route.travel_deg:.0f} st."


def stopped_by(result: MoveResult) -> bool:
    """The move ended on a STOP, or never started because a STOP disarmed the arm."""
    return result.reason == "stop" or (result.state == "rejected" and result.reason == "not armed")


def ended_by(result: MoveResult) -> Ended:
    if stopped_by(result):
        return Ended("stop", "ramie trzyma zmierzona poze")
    detail = ", ".join(x for x in (result.reason, result.detail) if x)
    return Ended(result.state, detail)


def arrival_misses(station: Station, pose: Pose) -> list[str]:
    r = station.reading()
    if r is None:
        return ["brak swiezych probek"]
    d = station.arm.kin.fk(r.q)[:3, 3] - pose.p
    misses: list[str] = []
    if abs(d[0]) > ARRIVE_M or abs(d[1]) > ARRIVE_M:
        misses.append(f"TCP xy obok o {d[0] * 1000:+.1f} {d[1] * 1000:+.1f} mm")
    if not -ARRIVE_M - SAG_MAX <= d[2] <= ARRIVE_M:
        misses.append(f"TCP z obok o {d[2] * 1000:+.1f} mm")
    return misses


def measure_sag(station: Station, pose: Pose) -> float:
    """Commanded minus measured TCP z at the pose, averaged [m]."""
    values: list[float] = []
    end = station.now() + SAG_AVERAGE_S
    while True:
        r = station.reading()
        if r is not None:
            z = float(station.arm.kin.fk(r.q)[2, 3])
            values.append(min(max(float(pose.p[2]) - z, 0.0), SAG_MAX))
        if station.now() >= end:
            break
        station.wait(POLL_S)
    return float(np.mean(values)) if values else 0.0


def follow(station: Station, route: Route, pose: Pose) -> float:
    """Drives the route at 20 deg/s and checks the arrival; returns the sag at the pose."""
    for q in route.waypoints:
        result = station.move(q, PREPOSITION_SPEED_DEG_S, lag_deg=MOVE_LAG_DEG)
        if not result:
            raise ended_by(result)

    misses = arrival_misses(station, pose)
    if misses:
        raise Ended("not arrived", "; ".join(misses))
    if not station.simulated:
        r = station.reading()
        j5 = math.degrees(r.q[4]) if r is not None else math.nan
        if not abs(j5) <= HOVER_J5_MEASURED_MAX_DEG:
            raise Ended("j5 offset", f"J5 {j5:.2f} st. w pozie startowej - sprawdz zero J5")
    return measure_sag(station, pose)


# ====================================================================== the rig file
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_rig(path: str | None) -> Rig:
    """The file named with --rig (a missing file is an empty rig), or an empty rig without a path."""
    if path is None:
        return Rig()
    try:
        return Rig.load(path)
    except RigError as exc:
        raise Refusal("rig rejected", str(exc)) from exc


def require_rig(a: argparse.Namespace) -> Path:
    if not a.rig:
        raise Refusal("rig", f"grasp_bench.py {a.tool} potrzebuje --rig PLIK (plik wartosci zmierzonych)")
    return Path(a.rig).expanduser()


def update_rig(path: Path, section: str, values: Mapping[str, Any], *,
               overrides: Mapping[str, str] | None = None) -> Rig:
    """Re-reads the file (the operator may have edited it meanwhile), merges one section, saves."""
    rig = Rig.load(path)
    getattr(rig, section).update(values)
    if overrides:
        rig.overrides.update(overrides)
    rig.save()
    return rig


# ====================================================================== jaw
@dataclass(frozen=True)
class JawSample:
    t: float                          # station clock at the poll that saw a new backend stamp
    effort: float
    streamed: float
    commanded: float
    frozen: bool


@dataclass
class EffortLog:
    """One jaw move: gripper states, `t` relative to the command (before it < 0)."""

    t: np.ndarray
    effort: np.ndarray
    streamed: np.ndarray
    commanded: np.ndarray
    frozen: np.ndarray
    rate_hz: float
    valid: bool                       # False when the freeze fired during the move
    target: float
    t0: float                         # station clock of the command
    operator_s: float | None = None   # when the operator saw the jaw stop, after the command


@dataclass(frozen=True)
class Stroke:
    """What one jaw move shows."""

    stream_s: float | None            # a full 0..100 stroke of the backend's stream, scaled
    effort_s: float | None            # the last time after the command the effort left its noise band
    peak: float | None
    mean: float | None
    noise: float
    delta: float | None               # units the stream moved


def _nan(v: float | None) -> float:
    return math.nan if v is None else float(v)


class JawRecorder:
    """Gripper states polled every GRIPPER_POLL_S, one row per new backend stamp."""

    def __init__(self, jaw: BackendLink | SimJaw, stopped: threading.Event) -> None:
        self.jaw = jaw
        self.stopped = stopped
        self._last_t: float | None = None

    def until(self, t_end: float) -> list[JawSample] | None:
        """The rows up to `t_end`; None when a STOP or a lost link cut them short."""
        rows: list[JawSample] = []
        while True:
            if self.stopped.is_set():
                return None
            gs = self.jaw.gripper_state()
            if gs is None:
                return None
            now = self.jaw.clock()
            if gs.t != self._last_t:
                self._last_t = gs.t
                rows.append(JawSample(t=now, effort=_nan(gs.effort), streamed=_nan(gs.streamed),
                                      commanded=_nan(gs.commanded), frozen=bool(gs.frozen)))
            if now >= t_end:
                return rows
            self.jaw.wait(GRIPPER_POLL_S)


def _froze(t: np.ndarray, frozen: np.ndarray) -> bool:
    """Whether the freeze fired during the move: `frozen` turned true after the command.

    A flag left over from the previous close, which this move's open clears, does not count.
    """
    before = frozen[t < 0.0]
    seq = np.concatenate([[bool(before[-1]) if len(before) else False], frozen[t >= 0.0]])
    return bool(np.any(seq[1:] & ~seq[:-1]))


def effort_log(rows: Sequence[JawSample], target: float, t0: float) -> EffortLog:
    t = np.array([r.t for r in rows], float) - t0
    frozen = np.array([r.frozen for r in rows], bool)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return EffortLog(
        t=t,
        effort=np.array([r.effort for r in rows], float),
        streamed=np.array([r.streamed for r in rows], float),
        commanded=np.array([r.commanded for r in rows], float),
        frozen=frozen,
        rate_hz=(len(t) - 1) / span if span > 0.0 else 0.0,
        valid=not _froze(t, frozen),
        target=float(target),
        t0=t0,
    )


def log_gripper(jaw: BackendLink | SimJaw, moves: Sequence[float], recorder: JawRecorder) -> list[EffortLog]:
    """Commands each opening in turn and samples the gripper state around it; a move cut short is dropped."""
    logs: list[EffortLog] = []
    for target in moves:
        before = recorder.until(jaw.clock() + PRE_MOVE_S)
        if before is None:
            break
        t0 = jaw.clock()
        jaw.command(target)
        after = recorder.until(t0 + GRIPPER_SETTLE_S)
        if after is None:
            break
        logs.append(effort_log([*before, *after], target, t0))
    return logs


def log_with_operator(jaw: BackendLink | SimJaw, keys: Keys, moves: Sequence[float],
                      stopped: threading.Event) -> list[EffortLog]:
    """`log_gripper`, while a reader stamps the operator's lines ("the jaw has stopped")."""
    stamps: list[float] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            text = keys.readline(timeout_s=KEYS_POLL_S)
            if text is None:
                done.wait(0.02)
            elif text:
                stamps.append(jaw.clock())

    reader = threading.Thread(target=read, name="chwyt-oko", daemon=True)
    reader.start()
    try:
        logs = log_gripper(jaw, moves, JawRecorder(jaw, stopped))
    finally:
        done.set()
        reader.join(1.0)
    for log in logs:
        seen = [t - log.t0 for t in stamps if OPERATOR_MIN_S <= t - log.t0 <= GRIPPER_SETTLE_S]
        if seen:
            log.operator_s = min(seen)
    return logs


def stroke_from_log(log: EffortLog, *, noise: float | None = None) -> Stroke:
    """Stream stroke, effort peak and mean, and the last effort above the noise band of one move.

    `stream_s` scales the part of the move between STROKE_EDGE_UNITS from its start and from its end
    to a full 0..100 stroke (None for a move under STROKE_MIN_UNITS). `effort_s` is the last time
    after the command at which the effort left the band of NOISE_SIGMAS sigma of the samples before
    it (or `noise` given).
    """
    t = np.asarray(log.t, float)
    post = t >= 0.0
    eff = np.asarray(log.effort, float)
    pre_eff = eff[(~post) & np.isfinite(eff)]
    centre = float(np.mean(pre_eff)) if len(pre_eff) else 0.0
    if noise is None:
        noise = NOISE_SIGMAS * float(np.std(pre_eff)) if len(pre_eff) > 1 else 0.0
        noise = max(noise, EFFORT_NOISE_MIN)

    post_eff = eff[post]
    finite = np.isfinite(post_eff)
    peak = float(np.max(np.abs(post_eff[finite]))) if finite.any() else None
    mean = float(np.mean(np.abs(post_eff[finite]))) if finite.any() else None
    tp = t[post]
    above = finite & (np.abs(post_eff - centre) > noise)
    effort_s = float(tp[above][-1]) if above.any() else None

    s = np.asarray(log.streamed, float)[post]
    ok = np.isfinite(s)
    stream_s: float | None = None
    delta: float | None = None
    if ok.sum() >= 2:
        ts = tp[ok]
        ss = s[ok]
        delta = float(ss[-1] - ss[0])
        if abs(delta) >= STROKE_MIN_UNITS:
            moved = np.abs(ss - ss[0]) > STROKE_EDGE_UNITS
            arrived = np.abs(ss - ss[-1]) <= STROKE_EDGE_UNITS
            i0 = int(np.argmax(moved))
            i1 = int(np.argmax(arrived))
            if moved.any() and arrived.any() and i1 > i0:
                stream_s = float(ts[i1] - ts[i0]) * 100.0 / (abs(delta) - 2.0 * STROKE_EDGE_UNITS)
    return Stroke(stream_s=stream_s, effort_s=effort_s, peak=peak, mean=mean, noise=float(noise), delta=delta)


def _is_stroke(stroke: Stroke, closing: bool) -> bool:
    d = stroke.delta
    if d is None:
        return False
    return d <= -STROKE_MIN_UNITS if closing else d >= STROKE_MIN_UNITS


def gripper_summary(free: Sequence[EffortLog], loaded: Sequence[EffortLog]) -> dict[str, Any]:
    """Rig `gripper` values from VALID moves only; a key without a valid move is left out.

    A freeze cuts a close short, and that short time would become the jaw speed, so nothing of a move
    in which the freeze fired counts. `freeze_threshold` is never here: the operator decides it and
    records it with --set-freeze.
    """
    out: dict[str, Any] = {}
    valid = [log for log in free if log.valid]
    strokes = [stroke_from_log(log) for log in valid]
    for closing in (True, False):
        name = "close" if closing else "open"
        moves = [k for k, s in enumerate(strokes) if _is_stroke(s, closing)]
        effort = [strokes[k].effort_s for k in moves if strokes[k].effort_s is not None]
        operator = [valid[k].operator_s for k in moves if valid[k].operator_s is not None]
        if effort:
            out[f"stroke_{name}_s_effort"] = round(max(effort), 4)
        if operator:
            out[f"stroke_{name}_s_operator"] = round(max(operator), 4)

    free_peaks = [s.peak for s in strokes if _is_stroke(s, True) and s.peak is not None]
    loaded_strokes = [stroke_from_log(log) for log in loaded if log.valid]
    loaded_peaks = [s.peak for s in loaded_strokes if _is_stroke(s, True) and s.peak is not None]
    if free_peaks:
        out["free_effort_peak"] = round(max(free_peaks), 4)
    if loaded_peaks:
        out["loaded_effort_peak"] = round(max(loaded_peaks), 4)
    if free_peaks and loaded_peaks:
        separable = max(loaded_peaks) > SEPARABLE_RATIO * max(free_peaks)
        out["separable"] = bool(separable)
        out["suggested_threshold"] = round(0.5 * (max(free_peaks) + max(loaded_peaks)), 4) if separable else None
    return out


def parse_gap_mm(items: Sequence[str] | None) -> dict[str, float]:
    """`O:MM` ruler readings -> {opening: mm}."""
    out: dict[str, float] = {}
    for item in items or ():
        parts = str(item).split(":")
        try:
            if len(parts) != 2:
                raise ValueError(item)
            opening = float(parts[0])
            mm = float(parts[1])
        except ValueError:
            raise Refusal("gap mm", f"--gap-mm {item!r}: oczekiwano ROZWARCIE:MM, np. 50:53") from None
        if not (math.isfinite(opening) and 0.0 <= opening <= 100.0 and math.isfinite(mm) and mm > 0.0):
            raise Refusal("gap mm", f"--gap-mm {item!r}: rozwarcie 0..100 i dodatnie mm")
        out[f"{opening:g}"] = mm
    return out


def record_gripper_notes(a: argparse.Namespace, gap: Mapping[str, float]) -> int:
    """--gap-mm and --set-freeze: written without moving anything."""
    path = require_rig(a)
    values: dict[str, Any] = {}
    if a.set_freeze is not None:
        if not (math.isfinite(a.set_freeze) and a.set_freeze > 0.0):
            raise Refusal("freeze", f"--set-freeze {a.set_freeze}: oczekiwano dodatniej liczby "
                                    f"({FREEZE_DISABLED:g} = zamrozenie wylaczone)")
        values["freeze_threshold"] = float(a.set_freeze)
    rig = load_rig(str(path))
    if gap:
        merged = dict(rig.gripper.get("gap_mm") or {})
        for key, mm in gap.items():
            model = (GAP_CLOSED + GAP_PER_UNIT * float(key)) * 1000.0
            if abs(mm - model) > GAP_WARN_MM:
                print(f"UWAGA: rozwarcie {key}: {mm:.1f} mm, model szczek zaklada {model:.1f} mm "
                      f"(roznica {mm - model:+.1f} mm, ponad {GAP_WARN_MM:g} mm)", flush=True)
            merged[key] = mm
        values["gap_mm"] = merged

    rig = update_rig(path, "gripper", values)
    if a.set_freeze is not None:
        off = " (zamrozenie wylaczone)" if a.set_freeze >= FREEZE_DISABLED else ""
        print(f"Zapisano gripper.freeze_threshold = {a.set_freeze:g}{off}. Ta sama wartosc musi byc "
              f"w a1x.gripper_force konfiguracji blizniaka, w ktorym jedzie strategia.", flush=True)
    if gap:
        print(f"Zapisano gripper.gap_mm = {values['gap_mm']}", flush=True)
    print(f"Plik: {rig.path}", flush=True)
    return 0


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None or not math.isfinite(v) else format(v, spec)


@dataclass(frozen=True)
class JawLogs:
    free: list[EffortLog]
    loaded: list[EffortLog]


@dataclass(frozen=True)
class JawMove:
    part: str                         # "wolny" | "kostka"
    log: EffortLog


def print_moves(moves: Sequence[JawMove]) -> None:
    print("\nruch  faza      cel  suw strumienia [s]  wysilek szczyt  srednio  ostatni ponad szum [s]  "
          "oko [s]  wazny", flush=True)
    for i, move in enumerate(moves, 1):
        s = stroke_from_log(move.log)
        valid = "tak" if move.log.valid else "NIE (zamrozenie)"
        print(f"{i:4d}  {move.part:<8s} {move.log.target:4.0f}  {_fmt(s.stream_s):>18s}  {_fmt(s.peak):>14s}  "
              f"{_fmt(s.mean):>7s}  {_fmt(s.effort_s):>22s}  {_fmt(move.log.operator_s):>7s}  {valid}", flush=True)


def save_logs(rig_path: Path, moves: Sequence[JawMove]) -> Path:
    """The raw traces next to the rig file, for a jaw model that is not a guess."""
    path = rig_path.with_name(f"{rig_path.stem}_gripper_{stamp()}.npz")
    arrays: dict[str, np.ndarray] = {
        "part": np.array([m.part for m in moves]),
        "target": np.array([m.log.target for m in moves], float),
        "t0": np.array([m.log.t0 for m in moves], float),
        "rate_hz": np.array([m.log.rate_hz for m in moves], float),
        "valid": np.array([m.log.valid for m in moves], bool),
        "operator_s": np.array([_nan(m.log.operator_s) for m in moves], float),
    }
    for i, m in enumerate(moves):
        for key in ("t", "effort", "streamed", "commanded", "frozen"):
            arrays[f"{key}_{i}"] = np.asarray(getattr(m.log, key))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def sampling_rate(moves: Sequence[JawMove]) -> float:
    samples = sum(len(m.log.t) for m in moves)
    span = sum(float(m.log.t[-1] - m.log.t[0]) for m in moves if len(m.log.t) > 1)
    return samples / span if span > 0 else 0.0


def advise_freeze(values: Mapping[str, Any]) -> None:
    if values.get("separable") is False:
        print("Wysilek z kostka nie odroznia sie od swobodnego ruchu - progu nie da sie dobrac; zostaw "
              f"zamrozenie wylaczone ({FREEZE_DISABLED:g}).", flush=True)
    elif values.get("suggested_threshold") is not None:
        print(f"Sugerowany prog zamrozenia: {values['suggested_threshold']:g} (swobodnie "
              f"{values['free_effort_peak']:g}, z kostka {values['loaded_effort_peak']:g}). Ustaw go jako "
              f"a1x.gripper_force w konfiguracji blizniaka i zapisz: grasp_bench.py gripper --set-freeze <wartosc>.",
              flush=True)


def report_jaw(rig: Rig, logs: JawLogs) -> int:
    complete = len(logs.free) == len(GRIPPER_FREE_MOVES) and len(logs.loaded) == len(GRIPPER_LOADED_MOVES)
    moves = [JawMove("wolny", log) for log in logs.free] + [JawMove("kostka", log) for log in logs.loaded]
    if not moves:
        print("Brak pomiaru (przerwany).", file=sys.stderr)
        return 1
    print_moves(moves)
    if rig.path is not None:
        print(f"Zapis przebiegow: {save_logs(Path(rig.path), moves)}", flush=True)

    rate = sampling_rate(moves)
    print(f"Probkowanie: {rate:.0f} probek/s (wymagane >= {MIN_RATE_HZ:.0f})", flush=True)
    if rate < MIN_RATE_HZ:
        print(f"Za wolne probkowanie stanu chwytaka ({rate:.0f} < {MIN_RATE_HZ:.0f}/s) - nic nie zapisano.",
              file=sys.stderr)
        return 1
    invalid = [i for i, m in enumerate(moves, 1) if not m.log.valid]
    if invalid:
        print(f"Ruchy {invalid}: zamrozenie zadzialalo - ich liczby NIE ida do pliku.", flush=True)

    values = gripper_summary(logs.free, logs.loaded)
    advise_freeze(values)
    if not values:
        print("Brak waznych ruchow - nic nie zapisano.", flush=True)
    elif rig.path is None:
        print(f"Bez --rig nic nie zapisano: {values}", flush=True)
    else:
        saved = update_rig(Path(rig.path), "gripper", values)
        print(f"Zapisano gripper do {saved.path}: {values}", flush=True)
    return 0 if complete else 1


def measure_jaw(station: Station, keys: Keys) -> JawLogs:
    """The free moves, then the loaded ones once the operator has put the cube between the plates."""
    jaw = station.jaw
    loaded: list[EffortLog] = []
    with jaw.phase():
        if jaw.gripper_state() is None:
            raise Refusal("not armed", "backend nie podaje stanu chwytaka")
        confirm_motion(station, keys, "Enter = pomiar szczek (99, 100, 0, 100, 0, 100, potem z kostka 0, 100); "
                                      "ramie stoi; Ctrl-C = koniec. ")
        with operated(station, keys):
            print("Szczeki bez niczego miedzy nimi. Gdy szczeki stana, wpisz s i Enter (pomiar na oko); "
                  "sam Enter = STOP.", flush=True)
            free = log_with_operator(jaw, keys, GRIPPER_FREE_MOVES, station.stopped)
            if len(free) == len(GRIPPER_FREE_MOVES):
                keys.note_end()
                if keys.proceed("Poloz kostke 3 cm miedzy szczekami, potem d i Enter; sam Enter = STOP. "):
                    loaded = log_with_operator(jaw, keys, GRIPPER_LOADED_MOVES, station.stopped)
    return JawLogs(free=free, loaded=loaded)


def bench_gripper(a: argparse.Namespace, deps: Deps) -> int:
    """Jaw stroke and effort: free open and close moves, then a close on a 3 cm cube."""
    gap = parse_gap_mm(a.gap_mm)
    if a.set_freeze is not None or gap:
        return record_gripper_notes(a, gap)
    require_consent(a)
    rig = load_rig(a.rig)

    keys = deps.keys()
    keys.start()
    # The freeze would cut a free stroke short, and its time would become the jaw speed.
    settings = {**a1x_settings(a.config), "gripper_force": FREEZE_DISABLED}
    station = deps.open_station(a, settings)
    try:
        listen_for_other_senders(station)
        logs = measure_jaw(station, keys)
    finally:
        station.close()
    return report_jaw(rig, logs)


# ====================================================================== jog and teaching
@dataclass(frozen=True)
class Measured:
    """The measured tool, averaged over fresh samples."""

    p: np.ndarray                     # TCP, base frame [m]
    yaw: float                        # closing axis about +z [rad]
    joints: dict[str, float]          # arm joints [deg], raw as on the wire


@dataclass(frozen=True)
class JogDelta:
    dx: float
    dy: float
    dz: float
    dyaw: float


@dataclass(frozen=True)
class Solution:
    """One warm IK step of the jog: the joints, or why the step is refused."""

    q: np.ndarray | None
    refusal: str = ""


def _clip(v: float, limit: float) -> float:
    return min(max(v, -limit), limit)


class Jog:
    """Small Cartesian steps of a top-down tool, by ik_step, each a guarded move of galaxeo.arm."""

    def __init__(self, station: Station, pose: Pose, sag: float) -> None:
        self.station = station
        self.kin = station.arm.kin
        self.p = pose.p.copy()                       # commanded TCP, base frame
        self.yaw = float(pose.yaw)                   # commanded closing-axis yaw
        self.q = pose.q.copy()                       # last commanded arm joints [rad]
        self.opening = GRIP_OPEN
        #: Commanded minus measured TCP z in free air (servo sag), from the start pose and every clean step.
        self.sag = float(sag)
        self.contact_z: float | None = None          # measured TCP z when the last contact fired
        self.backed_off = False                      # the last contact ended with the 2 mm backoff
        self.note = ""                               # what the last refusal or contact was
        self.clipped = False                         # the last step was cut to its limit
        self._heights: list[float] = []              # measured TCP z before and after each slow `d` step
        self._drops: list[float] = []                # commanded drop of each of those steps

    # ------------------------------------------------------------ public
    def step(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, dyaw: float = 0.0) -> str | None:
        """One step: None, "contact" or a refused step ("ik", "jump", "collision", "stale feedback",
        "non-finite"). A STOP or a move that ended otherwise raises Ended."""
        self.note = ""
        self.clipped = False
        if not all(math.isfinite(v) for v in (dx, dy, dz, dyaw)):
            return "non-finite"
        reading = self.station.settled_reading()
        if reading is None:
            return "stale feedback"

        z_before = float(self._tcp(reading.q)[2])
        slow = dz < 0.0 and dx == 0.0 and dy == 0.0 and dyaw == 0.0 and z_before < JOG_SLOW_BELOW_Z
        delta = JogDelta(dx=_clip(dx, JOG_STEP_M), dy=_clip(dy, JOG_STEP_M),
                         dz=max(min(dz, JOG_STEP_M), -(JOG_SLOW_STEP_M if slow else JOG_STEP_M)),
                         dyaw=_clip(dyaw, JOG_STEP_YAW))
        self.clipped = delta != JogDelta(dx=dx, dy=dy, dz=dz, dyaw=dyaw)

        p = self.p + np.array([delta.dx, delta.dy, delta.dz])
        yaw = float(wrap(self.yaw + delta.dyaw))
        # Below 4 cm the checker would refuse the touch pose itself: the IK rules only.
        solution = self._solve(p, yaw, collision=bool(p[2] > JOG_COLLISION_ABOVE_Z))
        if solution.q is None:
            return solution.refusal

        self._note_slow(slow, z_before, -delta.dz)
        lag_before = np.abs(self.q - reading.q)
        profile = "descend" if delta.dz < 0.0 else "normal"
        result = self.station.move(solution.q, JOG_SPEED_DEG_S, profile=profile, lag_deg=JOG_GUARD_LAG_DEG)
        self.p = p
        self.yaw = yaw
        self.q = solution.q.copy()

        if _tripped(result):
            self.note = f"straznik uderzen: {result.reason}" + (f" ({result.detail})" if result.detail else "")
            return self._contact()
        if not result:
            raise ended_by(result)
        hit = self._settle(solution.q, lag_before)
        if hit is None and slow:
            hit = self._height_rule()
        if hit is not None:
            return self._contact()
        if not slow:
            self._update_sag()
        return None

    def grip(self, opening: float) -> str | None:
        """The jaw to `opening` (0..100) in a jaw phase; the arm holds its last p_des meanwhile."""
        if not math.isfinite(opening):
            return "non-finite"
        v = min(max(float(opening), 0.0), 100.0)
        with self.station.jaw.phase():
            self.station.jaw.grip(v, self.station.stopped)
        if self.station.stopped.is_set():
            raise Ended("stop", "szczeki stoja tam, gdzie byl strumien")
        self.opening = v
        return None

    def measured(self, average_s: float = AVERAGE_S) -> Measured:
        """The measured tool, averaged over `average_s` of fresh samples."""
        end = self.station.now() + average_s
        samples: list[np.ndarray] = []
        last_t: float | None = None
        while True:
            r = self.station.reading()
            if r is not None and r.t != last_t:
                last_t = r.t
                samples.append(r.q.copy())
            if self.station.now() >= end:
                break
            self.station.wait(JOG_POLL_S)
        if not samples:
            raise Ended("stale feedback", "brak swiezych probek do usrednienia")
        q = np.mean(samples, axis=0)
        T = self.kin.fk(q)
        joints = {j: round(math.degrees(float(q[i])), 4) for i, j in enumerate(JOINTS)}
        return Measured(p=T[:3, 3].copy(), yaw=float(yaw_of(T[:3, :3])), joints=joints)

    def measured_tcp(self) -> np.ndarray | None:
        r = self.station.reading()
        return None if r is None else self._tcp(r.q)

    # ------------------------------------------------------------ internals
    def _tcp(self, q: np.ndarray) -> np.ndarray:
        return self.kin.fk(q)[:3, 3].copy()

    def _solve(self, p: np.ndarray, yaw: float, *, collision: bool) -> Solution:
        step = self.kin.ik_step(p, grasp_R(yaw), self.q, tol_pos=JOG_TOL_POS, tol_rot=JOG_TOL_ROT)
        jump = math.degrees(step.jump)
        self.note = (f"blad pozycji {step.pos_err * 1000:.2f} mm, obrotu {step.rot_err * 1000:.1f} mrad, "
                     f"skok {jump:.1f} st.")
        if not step.converged:
            return Solution(q=None, refusal="ik")
        if jump > IK_MAX_JUMP_DEG:
            return Solution(q=None, refusal="jump")
        if collision:
            why = self.station.arm.collision.config_clear(step.q)
            if why is not None:
                self.note = why
                return Solution(q=None, refusal="collision")
        self.note = ""
        return Solution(q=step.q.copy())

    def _note_slow(self, slow: bool, z_before: float, drop: float) -> None:
        if not slow:
            self._heights.clear()
            self._drops.clear()
            return
        if not self._heights:
            self._heights.append(z_before)
        self._drops.append(drop)

    def _settle(self, goal: np.ndarray, lag_before: np.ndarray) -> str | None:
        """JOG_SETTLE_S after the move: contact when a joint stays further behind than at the start of
        the step by more than JOG_LAG_DEG throughout."""
        end = self.station.now() + JOG_SETTLE_S
        excess = np.full(len(goal), math.inf)
        seen = 0
        last_t: float | None = None
        while True:
            r = self.station.reading()
            if r is not None and r.t != last_t:
                last_t = r.t
                seen += 1
                excess = np.minimum(excess, np.abs(goal - r.q) - lag_before)
            if self.station.now() >= end:
                break
            self.station.wait(JOG_POLL_S)
        if not seen:
            return None
        worst = int(np.argmax(excess))
        behind = math.degrees(float(excess[worst]))
        if behind <= JOG_LAG_DEG:
            return None
        self.note = f"opoznienie: J{worst + 1} o {behind:.1f} st. dalej za rozkazem niz na poczatku kroku"
        return "contact"

    def _height_rule(self) -> str | None:
        """After a 1 mm `d` step below JOG_SLOW_BELOW_Z: contact when two steps together dropped less
        than JOG_CONTACT_MIN_DROP, scaled by the commanded drop (shorter steps in free air are no contact)."""
        tcp = self.measured_tcp()
        if tcp is None:
            return None
        self._heights.append(float(tcp[2]))
        if len(self._heights) < 3:
            return None
        drop = self._heights[-3] - self._heights[-1]
        commanded = sum(self._drops[-2:])
        if drop >= JOG_CONTACT_MIN_DROP * commanded / (2.0 * JOG_SLOW_STEP_M):
            return None
        self.note = f"wysokosc: TCP opadl o {drop * 1000:.2f} mm w dwoch krokach (rozkaz {commanded * 1000:.1f} mm)"
        return "contact"

    def _update_sag(self) -> None:
        tcp = self.measured_tcp()
        if tcp is not None:
            self.sag = min(max(float(self.p[2]) - float(tcp[2]), 0.0), SAG_LIMIT)

    def _hold_measured(self) -> np.ndarray | None:
        """The measured pose held, and the jog's command moved onto it, the yaw too: a backoff at the
        commanded yaw would push a blocked joint 6 again."""
        q = self.station.hold_measured()
        if q is None:
            return None
        T = self.kin.fk(q)
        self.p = T[:3, 3].copy()
        self.yaw = float(yaw_of(T[:3, :3]))
        self.q = q.copy()
        return q

    def _contact(self) -> str:
        """Hold the measured pose, back off JOG_BACKOFF straight up, report "contact" (or the ending).

        The backoff starts from the measured height plus the free-air sag: a sagging servo would
        otherwise leave the tips on the table. It keeps the commanded reach, since the sag also pulls
        the tool towards the base and every contact would creep that way, but the measured bearing,
        so a blocked joint 1 is not pushed again.
        """
        reach = math.hypot(self.p[0], self.p[1])
        self._heights.clear()
        self._drops.clear()
        self.backed_off = False
        q = self._hold_measured()
        if q is None:
            raise Ended("stale feedback", f"po kontakcie ({self.note}) brak swiezego pomiaru ramienia")
        p_contact = self._tcp(q)
        self.contact_z = float(p_contact[2])

        bearing = math.atan2(p_contact[1], p_contact[0])
        p_back = np.array([reach * math.cos(bearing), reach * math.sin(bearing),
                           p_contact[2] + JOG_BACKOFF + self.sag])
        note = self.note
        solution = self._solve(p_back, self.yaw, collision=False)
        if solution.q is None:
            self.note = f"{note}; cofniecie odrzucone ({solution.refusal})"
            return "contact"
        result = self.station.move(solution.q, JOG_SPEED_DEG_S, lag_deg=JOG_GUARD_LAG_DEG)
        self.note = note
        if stopped_by(result):
            raise ended_by(result)
        if not result:
            # The backoff itself met something: stay where the arm is.
            self._hold_measured()
            self.note += "; cofniecie przerwane"
            return "contact"
        self.p = p_back
        self.q = solution.q.copy()
        self.backed_off = True
        return "contact"


def _tripped(result: MoveResult) -> bool:
    """The move met something: the impact guard tripped, or a joint never arrived."""
    return result.state == "backoff" or (result.state == "hold" and (result.reason.startswith("impact")
                                                                     or result.reason == "not arrived"))


def default_xy(rig: Rig, what: str) -> tuple[float, float]:
    other = "grasp" if what == "table" else "table"
    tcp = getattr(rig, other).get("tcp")
    if isinstance(tcp, (list, tuple)) and len(tcp) == 3:
        return float(tcp[0]), float(tcp[1])
    return START_XY_DEFAULT


def store_table(path: Path, m: Measured, *, force: bool) -> bool:
    """Writes `table`; False (nothing written) when the touch is more than TOUCH_TOLERANCE off without --force."""
    touch = float(m.p[2])
    diff = touch - TOUCH_EXPECTED_Z
    print(f"Dotyk stolu: TCP z {touch * 1000:.1f} mm, oczekiwane {TOUCH_EXPECTED_Z * 1000:.1f} mm "
          f"(roznica {diff * 1000:+.1f} mm)", flush=True)
    overrides = None
    if abs(diff) > TOUCH_TOLERANCE:
        if not force:
            print(f"Nie zapisano: roznica ponad {TOUCH_TOLERANCE * 1000:.0f} mm - problem z mocowaniem albo "
                  f"kinematyka. Sprawdz i naucz ponownie; --force zapisze mimo to (z wyjatkiem w pliku).",
                  file=sys.stderr, flush=True)
            return False
        overrides = {"table": f"--force {now_iso()}: dotyk {diff * 1000:+.1f} mm od oczekiwanego"}

    rig = update_rig(path, "table", {
        "touch_tcp_z": round(touch, 6),
        "tcp": [round(float(v), 6) for v in m.p],
        "joints": m.joints,
        "taught_at": now_iso(),
    }, overrides=overrides)
    grasp = rig.grasp.get("tcp")
    if isinstance(grasp, (list, tuple)) and len(grasp) == 3:
        far = math.hypot(float(m.p[0]) - float(grasp[0]), float(m.p[1]) - float(grasp[1]))
        verdict = "w porzadku" if far <= TABLE_TOUCH_FAR else "ZA DALEKO - strategia odmowi (table touch far)"
        print(f"Od punktu chwytu: {far * 1000:.0f} mm (najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm) - {verdict}",
              flush=True)
    else:
        print(f"Punkt chwytu naucz najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm (xy) od tego dotyku.", flush=True)
    print(f"Zapisano table do {rig.path} (wazne {TEACH_MAX_AGE_S / 3600:.0f} h).", flush=True)
    return True


def store_point(path: Path, m: Measured, *, object_height: float) -> Rig:
    """Writes `grasp`: the taught TCP and yaw, the object height, and zero joint offsets: the bench
    reads raw wire angles, the frame the wrist camera was calibrated in."""
    rig = update_rig(path, "grasp", {
        "tcp": [round(float(v), 6) for v in m.p],
        "yaw": round(float(m.yaw), 6),
        "joints": m.joints,
        "taught_at": now_iso(),
        "object_height": float(object_height),
        "joint_offsets_deg": [0.0] * len(JOINTS),
    })
    print(f"Zapisano grasp do {rig.path}: TCP ({m.p[0] * 1000:.1f}, {m.p[1] * 1000:.1f}, {m.p[2] * 1000:.1f}) mm, "
          f"obrot osi zamykania {math.degrees(m.yaw):.1f} st., obiekt {object_height * 1000:.0f} mm "
          f"(wazne {TEACH_MAX_AGE_S / 3600:.0f} h).", flush=True)
    return rig


def explain_point(rig: Rig, kin: Kinematics) -> None:
    believed = rig.believed()
    yaw = rig.grasp_yaw()
    if believed is None or yaw is None:
        print("Stol nie jest nauczony: chwyt, w ktory wierzy polityka (xy punktu, z = dotyk + 6 mm), bedzie "
              "znany po grasp_bench.py table.", flush=True)
        return
    print(f"Chwyt dla polityki: ({believed[0] * 1000:.1f}, {believed[1] * 1000:.1f}, {believed[2] * 1000:.1f}) mm "
          f"= xy punktu, z = dotyk stolu + {TIP_CLEAR * 1000:.0f} mm.", flush=True)
    table = rig.table.get("tcp")
    if isinstance(table, (list, tuple)) and len(table) == 3:
        far = math.hypot(float(table[0]) - believed[0], float(table[1]) - believed[1])
        if far > TABLE_TOUCH_FAR:
            print(f"UWAGA: dotyk stolu {far * 1000:.0f} mm od punktu (najwyzej {TABLE_TOUCH_FAR * 1000:.0f}) - "
                  f"strategia odmowi (table touch far).", flush=True)
    try:
        hover = solve_pose(kin, believed + np.array([0.0, 0.0, HOVER]), yaw)
    except Refusal as exc:
        print(f"Zawis nad punktem nie do osiagniecia: {exc.detail}", flush=True)
        return
    outside = training_envelope(hover)
    if outside:
        print("Poza obszarem treningu: " + "; ".join(outside), flush=True)
    else:
        print("Zawis w obszarze treningu.", flush=True)


JOG_HELP = ("Polecenia (Enter po kazdym): d MM w dol, u MM w gore, x MM, y MM (ze znakiem), r ST obrot, "
            "g 0..100 szczeki, ok = zapisz, q = koniec bez zapisu; sam Enter = STOP. "
            f"Kroki najwyzej {JOG_STEP_M * 1000:.0f} mm ({JOG_SLOW_STEP_M * 1000:.0f} mm w dol ponizej "
            f"{JOG_SLOW_BELOW_Z * 1000:.0f} mm), obrot {math.degrees(JOG_STEP_YAW):.0f} st.")


def print_contact(jog: Jog) -> None:
    z = "-" if jog.contact_z is None else f"{jog.contact_z * 1000:.1f}"
    if jog.backed_off:
        after = f"cofnelo sie o {JOG_BACKOFF * 1000:.0f} mm. Zejdz 'd 1' do dotyku i wpisz 'ok'."
    else:
        after = "stoi tam (bez cofniecia): podnies je 'u 2' i zejdz 'd 1'."
    print(f"Kontakt prawdopodobny ({jog.note}); TCP przy kontakcie z {z} mm. Ramie trzyma pomiar i {after}",
          flush=True)


def print_pose(jog: Jog) -> None:
    tcp = jog.measured_tcp()
    if tcp is None:
        return
    print(f"TCP zmierzony ({tcp[0] * 1000:.1f}, {tcp[1] * 1000:.1f}, {tcp[2] * 1000:.1f}) mm, rozkaz z "
          f"{jog.p[2] * 1000:.1f} mm, obrot {math.degrees(jog.yaw):.1f} st., szczeki {jog.opening:.0f}", flush=True)


def store(jog: Jog, what: str, rig_path: Path | None, *, force: bool, object_height: float) -> bool:
    """`ok`: the measured tool into the rig file; False when the table touch was refused."""
    m = jog.measured()
    if rig_path is None:
        print(f"Bez --rig nic nie zapisano: TCP ({m.p[0] * 1000:.1f}, {m.p[1] * 1000:.1f}, {m.p[2] * 1000:.1f}) mm, "
              f"obrot {math.degrees(m.yaw):.1f} st.", flush=True)
        return True
    if what == "table":
        return store_table(rig_path, m, force=force)
    rig = store_point(rig_path, m, object_height=object_height)
    explain_point(rig, jog.kin)
    return True


@dataclass(frozen=True)
class JogLine:
    """One jog command: `d 5` is JogLine("d", 5.0)."""

    cmd: str
    value: float


JOG_COMMANDS = ("d", "u", "x", "y", "r", "g")


def parse_jog_line(line: str) -> JogLine | None:
    parts = line.split()
    if len(parts) != 2 or parts[0].lower() not in JOG_COMMANDS:
        return None
    try:
        value = float(parts[1])
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return JogLine(cmd=parts[0].lower(), value=value)


def run_line(jog: Jog, line: JogLine) -> str | None:
    mm = line.value / 1000.0
    if line.cmd == "g":
        return jog.grip(line.value)
    if line.cmd == "d":
        return jog.step(dz=-abs(mm))
    if line.cmd == "u":
        return jog.step(dz=abs(mm))
    if line.cmd == "x":
        return jog.step(dx=mm)
    if line.cmd == "y":
        return jog.step(dy=mm)
    return jog.step(dyaw=math.radians(line.value))


def report_step(jog: Jog, why: str | None) -> None:
    if jog.clipped:
        print("Krok przyciety do limitu.", flush=True)
    if why == "contact":
        print_contact(jog)
    elif why is not None:
        print(f"Krok odrzucony ({why}): {jog.note}", flush=True)
    print_pose(jog)


def teach_lines(jog: Jog, keys: Keys, what: str, rig_path: Path | None, *, force: bool,
                object_height: float) -> int:
    """The operator's line commands: 0 when `ok` saved, 1 on `q` or a STOP, 2 on `q` after a refused save."""
    refused = False
    print(JOG_HELP, flush=True)
    while True:
        if keys.stop.is_set():
            return 1
        line = keys.readline(timeout_s=0.2 if keys.interactive else None)
        if line is None and keys.interactive:
            continue
        if line is None:
            print("Koniec wejscia - bez zapisu.", flush=True)
            return 2 if refused else 1
        if line == "":
            keys.stop.set()                    # the stop watch carries it out
            return 1

        word = line.split()[0].lower()
        if word == "q":
            print("Koniec bez zapisu.", flush=True)
            return 2 if refused else 1
        if word == "ok":
            if store(jog, what, rig_path, force=force, object_height=object_height):
                return 0
            refused = True
            continue

        jog_line = parse_jog_line(line)
        if jog_line is None:
            print(JOG_HELP, flush=True)
            continue
        report_step(jog, run_line(jog, jog_line))


def teach_object_height(a: argparse.Namespace) -> float:
    height = float(getattr(a, "object_height", OBJECT_HEIGHT_DEFAULT))
    if not OBJECT_HEIGHT_RANGE[0] <= height <= OBJECT_HEIGHT_RANGE[1]:
        raise Refusal("object height", f"--object-height {height}: oczekiwano "
                                       f"{OBJECT_HEIGHT_RANGE[0]}..{OBJECT_HEIGHT_RANGE[1]} m")
    if not (math.isfinite(a.start_z) and a.start_z > 0.0):
        raise Refusal("start z", f"--start-z {a.start_z}: oczekiwano wysokosci TCP nad stolem [m]")
    return height


def announce_start(what: str, pose: Pose, route: Route, object_height: float) -> None:
    print(f"Start: TCP ({pose.p[0] * 1000:.0f}, {pose.p[1] * 1000:.0f}, {pose.p[2] * 1000:.0f}) mm, narzedzie w dol, "
          f"obrot {math.degrees(pose.yaw):.0f} st.; {describe(route)}", flush=True)
    if what == "table":
        print(f"Opuszki na stol najwyzej {TABLE_TOUCH_FAR * 1000:.0f} mm od punktu chwytu.", flush=True)
    else:
        print(f"Plytki szczek po obu stronach obiektu ({object_height * 1000:.0f} mm), opuszki tuz nad stolem.",
              flush=True)


def start_jog(station: Station, route: Route, pose: Pose) -> Jog:
    sag = follow(station, route, pose)
    with station.jaw.phase():
        # At the start pose, never earlier: open jaws sweep wider on the way.
        station.jaw.grip(GRIP_OPEN, station.stopped)
    jog = Jog(station, pose, sag)
    print_pose(jog)
    return jog


def bench_teach(a: argparse.Namespace, deps: Deps) -> int:
    """Teach the table touch (`table`) or the grasp point (`point`) by jogging the fingertips onto it."""
    what: Literal["table", "point"] = a.tool
    object_height = teach_object_height(a)
    require_consent(a)
    rig = load_rig(a.rig)
    xy = (a.xy[0], a.xy[1]) if a.xy else default_xy(rig, what)
    rig_path = None if rig.path is None else Path(rig.path)

    keys = deps.keys()
    keys.start()
    station = deps.open_station(a, a1x_settings(a.config))
    try:
        station.wait_feedback(FEEDBACK_WAIT_S)
        pose = solve_pose(station.arm.kin, [xy[0], xy[1], a.start_z], math.radians(a.yaw_deg))
        listen_for_other_senders(station)
        route = plan_route(station, pose, clear_height(rig))
        announce_start(what, pose, route, object_height)
        confirm_motion(station, keys, "Enter = ramie jedzie nad punkt startowy (20 st./s); Ctrl-C = koniec. ")

        with operated(station, keys):
            jog = start_jog(station, route, pose)
            return teach_lines(jog, keys, what, rig_path, force=a.force, object_height=object_height)
    finally:
        station.close()


# ====================================================================== latency
@dataclass(frozen=True)
class Frame:
    image: np.ndarray                 # grey, FRAME_SMALL
    t: float                          # station clock on arrival
    seq: int
    size: tuple[int, int]             # (width, height) as delivered


class Camera:
    """The wrist camera through OpenCV, read on its own thread: `latest()` is the newest frame,
    stamped on arrival with the station clock."""

    def __init__(self, source: str, clock: Callable[[], float]) -> None:
        import cv2

        self._cv2 = cv2
        self._clock = clock
        device: int | str = int(source) if str(source).isdigit() else str(source)
        api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(device, int) else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(device, api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, RAW_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RAW_H)
        if not self._cap.isOpened():
            raise RuntimeError(f"kamera {source} sie nie otworzyla (macOS: uruchom z Terminal.app)")
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="chwyt-kamera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cv2 = self._cv2
        seq = 0
        while not self._closed.is_set():
            ok, frame = self._cap.read()
            t = self._clock()
            if not ok or frame is None:
                self._closed.wait(0.01)
                continue
            seq += 1
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, FRAME_SMALL, interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest = Frame(image=small, t=t, seq=seq, size=(int(frame.shape[1]), int(frame.shape[0])))

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._closed.set()
        self._thread.join(1.0)
        self._cap.release()


@dataclass(frozen=True)
class Still:
    """The camera at rest before a step: the reference image and the change that counts as motion."""

    ref: np.ndarray
    threshold: float
    seq: int


@dataclass(frozen=True)
class StepLatency:
    actuation_s: float | None         # command -> the first fresh joint sample that moved LATENCY_MOVED_DEG
    camera_s: float | None            # that sample -> the stamp of the first frame that shows the motion


@dataclass
class LatencyRun:
    steps: list[StepLatency] = field(default_factory=list)
    ending: str | None = None
    effort_peak: dict[str, float] = field(default_factory=dict)

    def good(self) -> list[StepLatency]:
        return [s for s in self.steps if s.actuation_s is not None and s.camera_s is not None]


class StepEnded(Exception):
    def __init__(self, ending: str) -> None:
        super().__init__(ending)
        self.ending = ending


def _mad(image: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(image.astype(np.float32) - ref)))


class StepGuard:
    """galaxeo.arm's impact guard on the backend's readings while the latency steps run."""

    def __init__(self, link: BackendLink, effort_abs: float) -> None:
        self.link = link
        self.guard = ImpactGuard(JOINTS, lag_deg=MOVE_LAG_DEG, lag_s=GUARD_LAG_S, effort_abs=effort_abs)
        self.sent: dict[str, float] = {}
        self.peak: dict[str, float] = {}

    def feed(self, sample: JointSample) -> None:
        efforts = self.link.efforts()
        if efforts:
            for j, v in efforts.items():
                self.peak[j] = max(self.peak.get(j, 0.0), abs(float(v)))
        hit = self.guard.feed(sample.t, self.sent, sample.joints, sample.fresh, efforts=efforts,
                              velocities=self.link.velocities())
        if hit is not None:
            self.link.hold()
            raise StepEnded(f"impact {hit.kind} {hit.joint}")


def _check_stop(stopped: threading.Event) -> None:
    if stopped.is_set():
        raise StepEnded("stop")


def still_frames(camera: Camera, link: BackendLink, stopped: threading.Event) -> Still:
    """New frames at rest: their spread is the noise the motion has to rise above."""
    images: list[np.ndarray] = []
    seq: int | None = None
    end = link.clock() + LATENCY_STILL_S
    while len(images) < LATENCY_STILL_FRAMES and link.clock() < end:
        _check_stop(stopped)
        frame = camera.latest()
        if frame is not None and frame.seq != seq:
            seq = frame.seq
            images.append(frame.image.astype(np.float32))
        link.wait(LATENCY_POLL_S)
    if len(images) < 2:
        raise StepEnded("camera")
    ref = np.mean(images, axis=0)
    mads = np.array([_mad(i, ref) for i in images])
    threshold = float(np.mean(mads)) + LATENCY_SIGMAS * max(float(np.std(mads)), MAD_SIGMA_MIN)
    return Still(ref=ref, threshold=threshold, seq=seq)


def settle(link: BackendLink, guard: StepGuard, stopped: threading.Event, secs: float) -> None:
    end = link.clock() + secs
    while link.clock() < end:
        _check_stop(stopped)
        sample = link.sample()
        if sample is not None:
            guard.feed(sample)
        link.wait(LATENCY_POLL_S)


def latency_step(link: BackendLink, camera: Camera, base: Mapping[str, float], up: bool, guard: StepGuard,
                 stopped: threading.Event) -> StepLatency:
    still = still_frames(camera, link, stopped)
    start = link.sample()
    if start is None or not start.fresh:
        raise StepEnded("stale feedback")

    target = dict(base)
    target[J6] = base[J6] + (LATENCY_STEP_DEG if up else 0.0)
    guard.sent = {j: target[j] for j in JOINTS}
    t_cmd = link.clock()
    link.send(target)

    t_act: float | None = None
    t_frame: float | None = None
    seq = still.seq
    deadline = t_cmd + LATENCY_STEP_TIMEOUT_S
    while link.clock() < deadline and (t_act is None or t_frame is None):
        _check_stop(stopped)
        sample = link.sample()
        if sample is not None:
            guard.feed(sample)
            moved = abs(sample.joints[J6] - start.joints[J6]) > LATENCY_MOVED_DEG
            if t_act is None and sample.t > t_cmd and moved:
                t_act = sample.t
        frame = camera.latest()
        if t_frame is None and frame is not None and frame.seq != seq:
            seq = frame.seq
            if frame.t > t_cmd and _mad(frame.image, still.ref) > still.threshold:
                t_frame = frame.t
        link.wait(LATENCY_POLL_S)

    settle(link, guard, stopped, LATENCY_SETTLE_S)
    return StepLatency(actuation_s=None if t_act is None else t_act - t_cmd,
                       camera_s=None if t_act is None or t_frame is None else t_frame - t_act)


def measure_latency(link: BackendLink, camera: Camera, base: Mapping[str, float], guard: StepGuard,
                    stopped: threading.Event) -> LatencyRun:
    """Steps of joint 6 by +3 deg and back at the hover, through the backend as the strategy's commands go."""
    run = LatencyRun()
    try:
        for k in range(LATENCY_STEPS):
            up = k % 2 == 0
            step = latency_step(link, camera, base, up, guard, stopped)
            run.steps.append(step)
            print(f"krok {k + 1}: J6 {'+' if up else '-'}{LATENCY_STEP_DEG:g} st.: ruch po "
                  f"{_fmt(step.actuation_s, '.3f')} s, kamera {_fmt(step.camera_s, '.3f')} s po nim", flush=True)
        if LATENCY_STEPS % 2 == 1:
            # Back to the hover, settled: the end of the phase would otherwise leave joint 6 stepped.
            guard.sent = {j: base[j] for j in JOINTS}
            link.send(base)
            settle(link, guard, stopped, LATENCY_SETTLE_S)
    except StepEnded as exc:
        run.ending = exc.ending
    run.effort_peak = {j: round(v, 2) for j, v in guard.peak.items()}
    return run


def first_frame(camera: Camera, station: Station) -> Frame:
    end = station.now() + FIRST_FRAME_S
    while True:
        frame = camera.latest()
        if frame is not None:
            return frame
        if station.now() >= end:
            raise Ended("wrist camera silent", f"brak klatki przez {FIRST_FRAME_S:.0f} s: zrodlo kamery, inni "
                                               f"uzytkownicy kamery, na macOS Terminal.app")
        station.wait(FRAME_WAIT_S)



def load_policy(path: Path, sha256: str) -> Any:
    from galaxeo.grasp.policy import GraspStudentPolicy

    return GraspStudentPolicy.load(path, sha256)


def inference_p95_s(policy: Any) -> float:
    """p95 of one policy call on this machine, timed with the camera and the backend running [s]."""
    t = policy.time()
    print(f"Inferencja: mediana {t.median_ms:.1f} ms, p95 {t.p95_ms:.1f} ms", flush=True)
    return t.p95_ms / 1000.0


def latency_values(run: LatencyRun, infer_s: float) -> dict[str, float]:
    good = run.good()
    actuation = float(np.median([s.actuation_s for s in good]))
    camera_raw = float(np.median([s.camera_s for s in good]))
    # A joint sample stamped after the frame that already shows the motion gives a negative value;
    # the strategy then pairs by the frame's own stamp.
    camera = max(0.0, camera_raw)
    total = camera + infer_s + actuation
    print(f"Opoznienie: kamera {camera * 1000:.0f} ms (surowo {camera_raw * 1000:.0f}) + inferencja p95 "
          f"{infer_s * 1000:.0f} ms + ramie {actuation * 1000:.0f} ms = {total * 1000:.0f} ms "
          f"({'w granicy' if total <= LATENCY_MAX_S else 'PONAD'} {LATENCY_MAX_S * 1000:.0f} ms)", flush=True)
    return {"camera_s": round(camera, 4), "actuation_s": round(actuation, 4), "infer_p95_s": round(infer_s, 4),
            "total_s": round(total, 4)}


@dataclass(frozen=True)
class LatencySetup:
    """What the latency tool needs, checked before anything is opened."""

    path: Path
    rig: Rig
    believed: np.ndarray
    yaw: float
    source: str
    policy: Any


def latency_setup(a: argparse.Namespace, deps: Deps) -> LatencySetup:
    from galaxeo.grasp.policy import CheckpointError

    if a.sim:
        raise Refusal("sim", "symulowane ramie nie ma kamery na nadgarstku: opoznienie mierzy sie na prawdziwym "
                             "ramieniu (w symulacji strategia paruje klatki po znaczniku czasu)")
    require_consent(a)
    path = require_rig(a)
    rig = load_rig(str(path))
    believed = rig.believed()
    yaw = rig.grasp_yaw()
    if believed is None or yaw is None:
        raise Refusal("grasp point not taught", "najpierw grasp_bench.py table i point")
    if not a.student:
        raise Refusal("model", "--student PLIK: czas inferencji wchodzi w sume opoznienia")
    source = a.camera or rig.wrist_source()
    if not source:
        raise Refusal("wrist camera not set", "--camera ZRODLO albo wrist.source w pliku --rig")
    try:
        policy = deps.load_policy(Path(a.student).expanduser(), a.sha256)
    except CheckpointError as exc:
        raise Refusal("model", str(exc)) from exc
    return LatencySetup(path=path, rig=rig, believed=believed, yaw=yaw, source=str(source), policy=policy)


def check_camera(camera: Camera, station: Station) -> None:
    """A silent camera or one that is not full HD is found before the arm moves."""
    frame = first_frame(camera, station)
    if frame.size != (RAW_W, RAW_H):
        raise Refusal("camera size", f"kamera daje {frame.size[0]}x{frame.size[1]}, strategia potrzebuje "
                                     f"{RAW_W}x{RAW_H}")


def step_joint_6(station: Station, camera: Camera, pose: Pose, rig: Rig) -> LatencyRun:
    """The latency steps in one jaw phase: the backend drives the arm, as under the strategy."""
    link = station.jaw
    with link.phase():
        link.grip(GRIP_OPEN, station.stopped)
        base = {**link.app_degrees(pose.q), GRIPPER: GRIP_OPEN}
        return measure_latency(link, camera, base, StepGuard(link, rig.effort_abs()), station.stopped)


def save_latency(path: Path, run: LatencyRun, infer_s: float) -> None:
    if run.effort_peak:
        print(f"Szczyt |wysilku| przegubow w krokach J6: {run.effort_peak}", flush=True)
    if run.ending is not None:
        raise Ended(run.ending, "pomiar przerwany")
    if len(run.good()) < LATENCY_MIN_GOOD:
        raise Ended("too few steps", f"udane {len(run.good())} z {LATENCY_STEPS} - nic nie zapisano")
    saved = update_rig(path, "latency", latency_values(run, infer_s))
    print(f"Zapisano latency do {saved.path}.", flush=True)


def bench_latency(a: argparse.Namespace, deps: Deps) -> int:
    """Arm and camera latency at the hover, and the total one policy tick adds up to."""
    setup = latency_setup(a, deps)
    keys = deps.keys()
    keys.start()
    station = deps.open_station(a, a1x_settings(a.config))
    camera: Camera | None = None
    try:
        station.wait_feedback(FEEDBACK_WAIT_S)
        pose = solve_pose(station.arm.kin, setup.believed + np.array([0.0, 0.0, HOVER]), setup.yaw)
        listen_for_other_senders(station)
        try:
            camera = deps.open_camera(setup.source, station.now)
        except RuntimeError as exc:
            raise Ended("camera", str(exc)) from exc
        check_camera(camera, station)
        infer_s = inference_p95_s(setup.policy)
        route = plan_route(station, pose, clear_height(setup.rig))
        confirm_motion(station, keys, f"Enter = ramie jedzie nad punkt chwytu ({describe(route)}, 20 st./s), potem "
                                      f"{LATENCY_STEPS} krokow J6 o {LATENCY_STEP_DEG:g} st.; Ctrl-C = koniec. ")

        with operated(station, keys):
            follow(station, route, pose)
            run = step_joint_6(station, camera, pose, setup.rig)
        save_latency(setup.path, run, infer_s)
        return 0
    finally:
        if camera is not None:
            camera.close()
        station.close()


# ====================================================================== dot and show
def bench_dot(a: argparse.Namespace, deps: Deps) -> int:
    """Records `wrist.dot_error_px`, the dot error the operator read from the strategy's dumps."""
    path = require_rig(a)
    if a.error_px is None:
        raise Refusal("error px", "Podaj --error-px V: odleglosc [px w obrazie 160x120] miedzy malym kwadratem "
                                  "(gdzie model kamery widzi srodek gornej sciany kostki) a prawdziwym srodkiem "
                                  f"gornej sciany na zrzucie strategii (dump_dir), najwiecej {DOT_MAX_PX:g} px")
    value = float(a.error_px)
    if not (math.isfinite(value) and value >= 0.0):
        raise Refusal("error px", f"--error-px {value}: oczekiwano liczby >= 0")
    load_rig(str(path))

    rig = update_rig(path, "wrist", {"dot_error_px": value})
    if value <= DOT_MAX_PX:
        print(f"Blad kropki {value:g} px - w granicy {DOT_MAX_PX:g} px. Zapisano do {rig.path}.", flush=True)
    else:
        print(f"Blad kropki {value:g} px - PONAD {DOT_MAX_PX:g} px (dot off): potrzebna ponowna kalibracja kamery "
              f"na nadgarstku; wyjatek tylko przez overrides.dot. Zapisano do {rig.path}.", flush=True)
    return 0


def bench_show(a: argparse.Namespace, deps: Deps) -> int:
    """Prints the rig file and what each mode still misses on a real arm."""
    path = require_rig(a)
    rig = load_rig(str(path))
    exists = path.is_file()
    print(f"Plik: {path}" + ("" if exists else " (brak - nic nie zmierzono)"))
    print(json.dumps(rig.to_dict(), indent=2, ensure_ascii=False))
    now = time.time()
    labels = {"table": "stol", "grasp": "punkt chwytu"}
    for name, age in rig.taught_age_s(now).items():
        if age is None:
            print(f"{labels[name]}: nie nauczony")
        else:
            state = "wazne" if age <= TEACH_MAX_AGE_S else "PRZETERMINOWANE"
            print(f"{labels[name]}: nauczony {age / 3600:.1f} h temu ({state}, limit {TEACH_MAX_AGE_S / 3600:.0f} h)")
    print("Czego brakuje strategii na prawdziwym ramieniu:")
    for mode in MODES:
        missing = rig.missing(mode, simulated=False, now=now)
        label = "shadow" if mode == "shadow" else "real  "
        print(f"  {label} " + (", ".join(missing) if missing else "nic - moze ruszac"))
    print("  sim    nic (rig_json pusty: wartosci symulacji z Rig.for_sim)")
    return 0


# ====================================================================== command line
@dataclass(frozen=True)
class Deps:
    """What a run opens outside this file; tests replace each with a fake."""

    open_station: Callable[[argparse.Namespace, Mapping[str, Any]], Station] = open_station
    keys: Callable[[], Keys] = Keys
    open_camera: Callable[[str, Callable[[], float]], Camera] = Camera
    load_policy: Callable[[Path, str], Any] = load_policy


def _arm_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    where = p.add_mutually_exclusive_group()
    where.add_argument("--iface", default=default_iface(),
                       help="lacze CAN ramienia: can0 (Linux), xcan[:adres] (macOS), PCAN_USBBUS1 (Windows)")
    where.add_argument("--sim", action="store_true", help="ramie w symulacji (MuJoCo, galaxeo.arm.SimTransport)")
    p.add_argument("--i-know", action="store_true", help="wymagane na prawdziwym ramieniu: rozumiem, ze sie ruszy")
    p.add_argument("--config", default=None,
                   help="konfiguracja YAML blizniaka: sekcja a1x (szczeki jak w blizniaku; potrzebny PyYAML)")
    return p


def _rig_option() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rig", default=None, help="plik wartosci zmierzonych (galaxeo.grasp.rig); bez niego nic nie "
                                               "jest zapisywane")
    return p


def _teach_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--xy", nargs=2, type=float, default=None, metavar=("X", "Y"),
                   help="punkt startowy nad stolem [m, uklad bazy] (domyslnie xy drugiego nauczonego punktu, "
                        "inaczej 0 0.33)")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="obrot osi zamykania szczek wokol pionu [st.]")
    p.add_argument("--start-z", type=float, default=START_Z_DEFAULT, help="wysokosc TCP w punkcie startowym [m]")
    return p


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="tool", required=True)
    arm = _arm_options()
    rig = _rig_option()
    teach = _teach_options()

    p = sub.add_parser("gripper", parents=[arm, rig], help="czas ruchu i wysilek szczek (>= 150 probek/s)")
    p.add_argument("--gap-mm", action="extend", nargs="+", default=None, metavar="O:MM",
                   help="rozwarcie zmierzone linijka przy otwarciu O (0..100) w mm, np. --gap-mm 0:3 50:53 100:103")
    p.add_argument("--set-freeze", type=float, default=None,
                   help=f"zapisz prog zamrozenia szczek z a1x.gripper_force blizniaka "
                        f"({FREEZE_DISABLED:g} = wylaczony)")
    p.set_defaults(run=bench_gripper)

    p = sub.add_parser("table", parents=[arm, rig, teach], help="opuszki na stol: wysokosc dotyku")
    p.add_argument("--force", action="store_true",
                   help=f"zapisz dotyk dalej niz {TOUCH_TOLERANCE * 1000:.0f} mm od oczekiwanych "
                        f"{TOUCH_EXPECTED_Z * 1000:.1f} mm (z wyjatkiem w pliku)")
    p.set_defaults(run=bench_teach)

    p = sub.add_parser("point", parents=[arm, rig, teach], help="punkt chwytu i obrot szczek dotykiem")
    p.add_argument("--object-height", type=float, default=OBJECT_HEIGHT_DEFAULT, help="wysokosc obiektu [m]")
    p.set_defaults(run=bench_teach, force=False)

    p = sub.add_parser("latency", parents=[arm, rig], help="opoznienie ramienia i kamery nad punktem chwytu")
    p.add_argument("--student", default=None, help="plik modelu (czas inferencji wchodzi w sume)")
    p.add_argument("--sha256", default=CHECKPOINT_SHA256, help="sha256 modelu (domyslnie opublikowany)")
    p.add_argument("--camera", default=None, help="kamera na nadgarstku: numer OpenCV albo sciezka urzadzenia "
                                                  "(domyslnie wrist.source z pliku --rig)")
    p.set_defaults(run=bench_latency)

    p = sub.add_parser("dot", parents=[rig], help="zapisz blad kropki odczytany ze zrzutu strategii")
    p.add_argument("--error-px", type=float, default=None, help="blad kropki [px w obrazie 160x120]")
    p.set_defaults(run=bench_dot)

    p = sub.add_parser("show", parents=[rig], help="plik --rig i czego brakuje strategii")
    p.set_defaults(run=bench_show)
    return ap


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return a.run(a, deps or Deps())
    except Refusal as exc:
        print(f"Odmowa: {exc.reason}" + (f" ({exc.detail})" if exc.detail else ""), file=sys.stderr)
        return 2
    except Ended as exc:
        print(f"Koniec: {exc.ending}" + (f" ({exc.detail})" if exc.detail else ""), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl-C).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
