"""The bench harness for terminal tools that move the real A1X without the twin.

One owner of the CAN link at a time. galaxeo.arm moves the arm: `Arm.move` on a BusTransport (measured
start, ramp, impact guard, backoff). galaxeo.arm has no jaw, so the `a1x` backend drives it. Both read
0x052 from the same queue, and the backend takes any 0x050 it did not send for another sender, so they
never run together. The backend is connected only for a jaw phase (`BackendLink.phase`), and the
BusTransport sends and reads nothing meanwhile. Between phases the arm holds its last p_des and the jaw
its last 0x051.

Safety (docs/SAFETY.md): a tool that moves the real arm needs --i-know and an Enter. From then on a bare
Enter is STOP, and so is Ctrl-C. A STOP during a move holds the measured pose (`Motion`). A STOP in a jaw
phase stops the jaw where its stream is, and holds the measured arm pose if the phase drives the arm.
Nothing sends a release frame.

A tool raises `Refusal` when nothing moved and `Ended` when it stopped before its result; `run` turns
them into exit codes 2 and 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from galaxeo import protocol as P
from galaxeo.a1x import GRIP_START_DELTA
from galaxeo.arm import JOINTS, Arm, BusTransport, MoveResult, Reading, SimTransport
from galaxeo.bus import default_iface, open_bus
from galaxeo.robot.backend import A1XArm, A1XConfig, GripperState

logger = logging.getLogger("bench_a1x")

GRIPPER = "gripper"

# ---- keyboard
CONTINUE_WORD = "d"
ENTER_GUARD_S = 1.0                   # a line this soon after a move ended was typed as STOP
KEYS_POLL_S = 0.1                     # waits wake up this often, so Ctrl-C gets through
STOP_POLL_S = 0.05

# ---- the link
FEEDBACK_WAIT_S = 1.0
FEEDBACK_PERIOD_S = 0.01              # two 0x052 frames
FOREIGN_LISTEN_S = 0.3
JAW_TICK_S = 0.005                    # one exchange tick of the backend, when the caller runs them
HOLD_STREAM_S = 0.05                  # a hold streamed this long before a phase ends: the stream slews to it

# ---- moves
SEGMENT_DEG = 40.0                    # galaxeo.arm refuses 45 deg in one move; room for the measured start

# ---- jaw
GRIP_OPEN = 100.0
GRIP_WAIT_S = 2.0
GRIP_ARRIVED = 1.0                    # units between stream and target that count as there
GRIP_MEASURED_TOL = 3.0               # units between the measured jaw and the target that count as there
GRIP_NUDGE_S = 0.1
JAW_POLL_S = 0.01
#: An `a1x.gripper_force` no jaw effort reaches: the freeze never fires.
FREEZE_OFF = 100.0
SIM_JAW_EFFORT = 0.3                  # the simulated jaw's effort while its stream moves
SIM_JAW_NOISE = 0.01
SIM_FEED_S = 0.005                    # the simulated jaw reports at the A1X's 200 Hz


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


def run(tool: Callable[[], int]) -> int:
    """Runs a tool and returns its exit code: `Refusal` 2, `Ended` and Ctrl-C 1."""
    try:
        return tool()
    except Refusal as exc:
        print(f"Refused: {exc.reason}" + (f" ({exc.detail})" if exc.detail else ""), file=sys.stderr)
        return 2
    except Ended as exc:
        print(f"Ended: {exc.ending}" + (f" ({exc.detail})" if exc.detail else ""), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C).", file=sys.stderr)
        return 1


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
            self._thread = threading.Thread(target=self._read, name="bench-keys", daemon=True)
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
    """The backend's context: the URDF joint limits in degrees, the gripper 0..100."""
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


def refuse_after_stop(stopped: threading.Event) -> None:
    """A jaw command after a STOP would move the jaw the operator just stopped."""
    if stopped.is_set():
        raise Ended("stop", "the jaw gets no command after a STOP")


def until_streamed(jaw: BackendLink | SimJaw, opening: float, stopped: threading.Event) -> None:
    """Waits until the jaw stream stands on its target (`opening`, or where a contact froze it), at most
    GRIP_WAIT_S: a phase that ends earlier leaves the jaw where the stream was."""
    t_end = jaw.clock() + GRIP_WAIT_S
    while jaw.clock() < t_end and not stopped.is_set():
        if _stream_on_target(jaw.gripper_state()):
            return
        jaw.wait(JAW_POLL_S)
    if not stopped.is_set():
        logger.warning("jaw stream did not reach %.0f within %.1f s", opening, GRIP_WAIT_S)


class BackendLink:
    """The `a1x` backend on the bench's link, connected only for a jaw phase (see the module docstring)."""

    def __init__(self, backend: A1XArm, *, clock: Callable[[], float], sleep: Callable[[float], None],
                 threaded: bool, drain: Callable[[], object]) -> None:
        self.backend = backend
        self.clock = clock
        self._sleep = sleep
        self.threaded = threaded
        #: Empties the link before the backend connects (the BusTransport's settled reading).
        self._drain = drain
        #: Where the jaw stream stood when the last phase ended; the next phase starts the backend from it.
        self.opening: float | None = None
        self._active = threading.Event()
        self._drives_arm = False
        self._held = False

    @contextmanager
    def phase(self) -> Iterator[BackendLink]:
        # The backend forgets what it sent when it connects: an echo of the BusTransport's last 0x050
        # still queued would count as another sender.
        self._drain()
        try:
            self.backend.connect()
        except (RuntimeError, OSError) as exc:
            raise Ended("link", str(exc)) from exc
        self._drives_arm = False
        self._held = False
        self._active.set()
        try:
            if self.opening is not None:
                # A new connection guesses the jaw from `gripper_start`: its first-command gate would
                # drop a close from a wrong guess, or the stream would start from the wrong end.
                self.backend.assume_gripper(self.opening)
            yield self
        except BaseException:
            # Ctrl-C or an ending unwinding the phase: once disconnected, the arm would stay on the
            # last streamed p_des, mid-slew, and a STOP after it could no longer reach the backend.
            self.hold()
            raise
        finally:
            if self._held:
                self.wait(HOLD_STREAM_S)
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
        refuse_after_stop(stopped)
        if self._gate_drops(opening):
            if self.opening is not None:
                return
            self.command(GRIP_OPEN - 1.0)
            until = self.clock() + GRIP_NUDGE_S
            while self.clock() < until and not stopped.is_set():
                self.wait(JAW_POLL_S)
            refuse_after_stop(stopped)
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
            self.wait(JAW_POLL_S)

    def _gate_drops(self, opening: float) -> bool:
        """Whether the backend's first-command gate would drop `opening`: no stream in this connection
        yet, and `opening` within GRIP_START_DELTA of the stroke (0.5 of 0..100) of the opening it reports
        (assumed or measured)."""
        gs = self.backend.gripper_state()
        if gs is None or gs.target is not None:
            return False
        return abs(opening - self.backend.read_joints()[GRIPPER]) <= GRIP_START_DELTA * 100.0

    def send(self, targets: Mapping[str, float]) -> None:
        """Arm and jaw targets [app units] through the backend, as the twin sends them."""
        if any(j in targets for j in JOINTS):
            self._drives_arm = True
        self.backend.send_joints(dict(targets))

    def app_degrees(self, q: Sequence[float]) -> dict[str, float]:
        """Wire radians -> the backend's app degrees (minus the zero offsets)."""
        offsets = self.backend.a1x.cfg.joint_offsets_deg
        return {j: math.degrees(float(q[i])) - offsets[i] for i, j in enumerate(JOINTS)}

    def sample(self) -> JointSample | None:
        gs = self.backend.gripper_state()
        if gs is None or gs.age_s is None:
            return None
        joints = self.backend.read_joints()
        fresh = gs.age_s * 1000.0 <= self.backend.a1x.cfg.stale_ms
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
                self._held = True
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
        refuse_after_stop(stopped)
        self.command(opening)
        until_streamed(self, opening, stopped)

    def hold(self) -> None:
        self._advance()
        if self._streamed is not None:
            self._commanded = self._streamed

    def gripper_state(self) -> GripperState:
        """The simulated jaw has no freeze: `force_threshold` is FREEZE_OFF."""
        self._advance()
        now = self.clock()
        feed = math.floor(now / SIM_FEED_S)
        moving = self._commanded is not None and abs(self._commanded - self._streamed) > 1e-9
        effort = (SIM_JAW_EFFORT if moving else 0.0) + SIM_JAW_NOISE * (feed % 2)
        return GripperState(commanded=self._commanded, target=self._commanded, streamed=self._streamed,
                            position=None, frozen=False, effort=effort, t=feed * SIM_FEED_S,
                            age_s=now - feed * SIM_FEED_S, rate=self.rate, force_threshold=FREEZE_OFF,
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


# ====================================================================== the station
@dataclass
class MoveOutcome:
    """What a move's worker thread hands back."""

    result: MoveResult | None = None
    error: Exception | None = None


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

    def wait_feedback(self, secs: float = FEEDBACK_WAIT_S) -> Reading:
        end = self.now() + secs
        while True:
            r = self.reading()
            if r is not None:
                return r
            if self.now() >= end:
                raise Ended("no feedback", f"no fresh 0x052 frames for {secs:.0f} s - the CAN link is down "
                                           f"or the arm has no power")
            self.wait(FEEDBACK_PERIOD_S)

    def move(self, q: np.ndarray, speed: float, *, profile: str = "normal",
             lag_deg: float | None = None) -> MoveResult:
        """One guarded move on its own thread, so that Ctrl-C here reaches `Arm.stop` and the move holds
        the measured pose instead of leaving its last ramp command on the arm."""
        outcome = MoveOutcome()

        def work() -> None:
            try:
                outcome.result = self.arm.move(q, speed, profile=profile, lag_deg=lag_deg)
            except Exception as exc:
                outcome.error = exc

        worker = threading.Thread(target=work, name="bench-move", daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                worker.join(STOP_POLL_S)
        except KeyboardInterrupt:
            self.arm.stop("stop")
            worker.join(5.0)
            raise
        if outcome.error is not None:
            raise outcome.error
        return outcome.result

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
    transport = BusTransport(bus, tx=True, clock=clock, sleep=sleep, kp=backend.a1x.cfg.kp, kd=backend.a1x.cfg.kd)

    def drain() -> None:
        transport.read()
        sleep(FEEDBACK_PERIOD_S)
        transport.read()

    link = BackendLink(backend, clock=clock, sleep=sleep, threaded=threaded, drain=drain)
    return Station(arm=Arm(transport), jaw=link, simulated=False, bus=bus)


def sim_station() -> Station:
    transport = SimTransport()
    return Station(arm=Arm(transport), jaw=SimJaw(transport.now, transport.sleep), simulated=True)


def open_station(a: argparse.Namespace, settings: Mapping[str, Any]) -> Station:
    """The station of the command line: --sim, or the A1X on --iface with the backend's `settings`."""
    if a.sim:
        return sim_station()
    try:
        bus = open_bus(a.iface)
    except Exception as exc:
        raise Ended("link", f"could not open the link {a.iface}: {exc}") from exc
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
        raise Refusal("config", "--config needs PyYAML (pip install pyyaml)") from exc
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Refusal("config", f"{path}: {exc}") from exc
    section = data.get("a1x") if isinstance(data, dict) else None
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise Refusal("config", f"{path}: the a1x section must be a mapping")
    # The bench connects the backend for every jaw phase, the first one before the operator's Enter,
    # and FF 5 of the enable sequence briefly releases the motors of an arm without brakes.
    if section.get("enable_on_connect") is True:
        raise Refusal("enable on connect", "a1x.enable_on_connect enables the motors on every connect (FF 5 "
                                           "briefly releases them); enable the arm from the panel and set "
                                           "enable_on_connect: false")
    return dict(section)


class StopWatch:
    """Turns a STOP from the keyboard (a bare Enter once armed, a refused prompt) into `Station.stop`."""

    def __init__(self, keys: Keys, station: Station) -> None:
        self.keys = keys
        self.station = station
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._fired = False
        self._thread = threading.Thread(target=self._run, name="bench-stop", daemon=True)

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
        print("\nSTOP - the arm holds the measured pose.", flush=True)
        self.station.stop()

    def close(self) -> None:
        """A STOP typed just before the tool ended is still carried out."""
        self._done.set()
        if self.keys.stop.is_set():
            self.fire()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(1.0)


def require_consent(a: argparse.Namespace, tool: str) -> None:
    """The real arm moves only with --i-know."""
    if not a.sim and not a.i_know:
        raise Refusal("i-know", f"{tool} on {a.iface} moves the real arm - add --i-know")


def confirm_motion(station: Station, keys: Keys, prompt: str) -> None:
    """The operator's Enter before the first motion on a real arm; then Enter means STOP."""
    if not station.simulated and not keys.confirm(prompt):
        raise Refusal("not confirmed", "no Enter to confirm (needs a terminal)")
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
        raise Refusal("busy", f"another sender on the CAN link ({n} command frames): close the twin's panel, "
                              f"jog_a1x.py, so101_bridge.py, the ROS driver")


# ====================================================================== moves
def segments(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
    """The straight joint-space move cut so that no joint travels more than SEGMENT_DEG in one move."""
    travel = math.degrees(float(np.max(np.abs(goal - start))))
    n = max(1, math.ceil(travel / SEGMENT_DEG))
    return [start + (k / n) * (goal - start) for k in range(1, n + 1)]


def stopped_by(result: MoveResult) -> bool:
    """The move ended on a STOP, or never started because a STOP disarmed the arm."""
    return result.reason == "stop" or (result.state == "rejected" and result.reason == "not armed")


def ended_by(result: MoveResult) -> Ended:
    """The ending a move that did not reach its goal gives the tool."""
    if stopped_by(result):
        return Ended("stop", "the arm holds the measured pose")
    detail = ", ".join(x for x in (result.reason, result.detail) if x)
    return Ended(result.state, detail)


def tripped(result: MoveResult) -> bool:
    """The move met something: the impact guard tripped, or a joint never arrived."""
    return result.state == "backoff" or (result.state == "hold" and (result.reason.startswith("impact")
                                                                     or result.reason == "not arrived"))


# ====================================================================== command line
def arm_options() -> argparse.ArgumentParser:
    """--iface or --sim, --i-know and --config, as a parent parser."""
    p = argparse.ArgumentParser(add_help=False)
    where = p.add_mutually_exclusive_group()
    where.add_argument("--iface", default=default_iface(),
                       help="the arm's CAN link: can0 (Linux), xcan[:address] (macOS), PCAN_USBBUS1 (Windows)")
    where.add_argument("--sim", action="store_true", help="the simulated arm (MuJoCo, galaxeo.arm.SimTransport)")
    p.add_argument("--i-know", action="store_true", help="required on the real arm: I understand that it moves")
    p.add_argument("--config", default=None,
                   help="the twin's YAML config: its a1x section (the jaw runs as in the twin; needs PyYAML)")
    return p
