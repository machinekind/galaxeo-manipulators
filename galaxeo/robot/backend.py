"""Galaxea A1X over CAN-FD: the digital twin's `a1x` backend.

A plugin of the twin's backend contract (`BackendPlugin`): a plain class built by `create(ctx)`,
registered under the `lerobot_mp.backends` entry point. It never imports the twin. Protocol and
CAN link are this package's own (`galaxeo.protocol`, `galaxeo.bus`).

Units as everywhere in the twin: arm joints in degrees, the gripper 0..100 (0 = closed). The
app <-> CAN conversion lives only here:

* joint: ``rad = radians(deg + a1x.joint_offsets_deg[i])`` (zero offset, J3 about +1.49 deg),
* gripper: p_des [rad] linear ``closed + v/100 * (open - closed)`` (more positive = tighter).
  Measured from group 7 of frame 0x052 (jaw position, rad like p_des; measured 2026-09-27: stops
  at +0.039 closed, -1.992 open) through the two points `gripper_fb_closed/open`, set by default.
  Group 7 is valid ONLY while the gripper is driven: at rest after power-on it reads an arbitrary
  value (e.g. -2.88 rad = -165 deg with the jaws at -85 deg) and follows p_des only while 0x051
  flows. So the measurement counts from `GRIP_LIVE_FRAMES` 0x052 frames after the first 0x051 of
  this connection (and again after a link loss); until then, and without `gripper_fb_*`, open
  loop (the last command; a warning in `calibration_warnings`). Before the first command open
  loop reports `gripper_start` (0 = closed), and the twin's "Set home = here" confirms it through
  `assume_gripper(0)`: the station convention is home = jaws closed. While the opening is only
  guessed, the first opening goes straight to the target and closing ramps.
* The gripper moves over XCAN (macOS): measured 2026-09-27 (`python -m galaxeo.xcan_usb
  --grip-test`), contrary to the caveat on `galaxeo.bus.XcanBus`. `a1x.gripper_over_xcan: false`
  restores the refusal (`gripper_drivable()` False); env A1X_GRIPPER_XCAN=1 forces driving.

Safety rules (docs/SAFETY.md, docs/PROTOCOL.md):

* Nothing goes on the bus before the first `send_joints`: an uncommanded A1X holds the pose it
  stands in, and the twin sends nothing until the first move.
* The p_des stream starts from the MEASURED pose and moves at most `max_step_deg_s`, so a bad
  first command does not jump. The stream runs at `stream_hz` and repeats the last target.
* Stale (> `stale_ms`) or frozen (FF 2) telemetry stops sending, sets `_link_silent` and puts an
  error in `faults()`: the twin stops and later starts from the measurement. Arm and gripper
  targets are dropped: when frames return, nothing goes out until the next `send_joints`.
* Never FF 2 (release: an arm without brakes drops), never kp 0 (the arm goes deaf), never the
  0x023 heartbeat. The FF 1 -> 5 -> 6 enable only on request (`enable`), with p_des = the
  measurement before, between and after each code.
* `disconnect` ends the stream and closes the link; it never releases the motors.
* One sender on the link: a 0x050 / 0x051 / FF frame this backend did not send is a fault in
  `faults()` for `FOREIGN_WINDOW_S`, and the twin STOPs.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import types
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from typing import Any, Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

from .. import bus as can
from .. import protocol as P

#: A1X_GRIPPER_XCAN=1 (or true / yes / tak) = the same as `a1x.gripper_over_xcan: true`.
GRIPPER_XCAN_ENV = "A1X_GRIPPER_XCAN"

logger = logging.getLogger(__name__)

#: A gripper command [app units] further than this from the reported value starts the 0x051
#: stream: holding the reported value after Connect sends nothing.
GRIP_START_DELTA = 0.5
#: A joint "stands" below this |velocity| [deg/s] (deaf arm detection).
STILL_DEG_S = 0.3
#: How long Connect waits for the first 0x052 frame [s].
CONNECT_TIMEOUT_S = 3.0
#: How long after the first frame to collect more to recognise frozen telemetry [s] (at least;
#: a large `frozen_frames` stretches the window to two 0x052 periods per frame).
SETTLE_S = 0.4
#: Period of the 0x052 frames [s]: the arm answers at 200 Hz.
FB_PERIOD_S = 0.005
#: Time between the codes of the enable sequence [s]; the p_des = measurement stream runs throughout.
ENABLE_STEP_S = 0.3
#: Gripper rest |effort|: the median of this many stream ticks at the start of each close (0.1 s at
#: 200 Hz); the target does not freeze within this window. Contact = |effort| > rest + `gripper_force`.
GRIP_REST_STEPS = 20
#: Opening [p_des] from the tightest point of a close after which the next close measures the rest
#: again. A smaller opening (jaws still on the cube) keeps the old rest.
GRIP_REOPEN_PDES = 0.3
#: This many 0x052 frames after the first 0x051 sent (20 ms) before group 7 counts as the gripper
#: measurement. At rest after power-on group 7 reads an arbitrary value (2026-09-27: -165 deg with
#: the jaws at -85 deg); it follows p_des only while the gripper is driven.
GRIP_LIVE_FRAMES = 4
#: Margin on the time of a full gripper stroke [s] (`gripper_travel_s`).
GRIP_TRAVEL_MARGIN_S = 0.2
#: How long `faults()` reports another sender after its last command frame [s].
FOREIGN_WINDOW_S = 0.5
#: The last payloads remembered per CAN id: the adapter's echo is not another sender.
ECHO_KEEP = 16
#: An echo arrives within this time of the send or not at all [s], counted to the exchange before
#: the drain that finds it (frames carry no receive stamp, and it came in after that exchange): a
#: CAN thread stalled by the GIL or a garbage collection drains late, and its own echo must not
#: look like another sender. Several loop periods; a later copy is another sender (e.g. a second
#: backend holding the same pose with the same gains), and so is a second copy of one send.
ECHO_WINDOW_S = 0.25
#: The fault while another sender is seen. No joint name: the twin takes it for an arm fault and STOPs.
FOREIGN_FAULT = "inny nadawca na laczu CAN (ramki rozkazu spoza blizniaka)"

_Settings = TypeVar("_Settings")


def open_bus(iface: str) -> Any:
    """The CAN link from `galaxeo.bus` (`can0`, `xcan`, `xcan:6`, `PCAN_USBBUS1`, `virtual:x`)."""
    try:
        return can.open_bus(iface)
    except ImportError as exc:
        extra = "xcan" if iface.lower().startswith("xcan") else "pcan"
        raise ImportError(f'lacze {iface!r} potrzebuje dodatku: pip install "galaxeo[{extra}]" ({exc})') from exc


def load_settings(cls: type[_Settings], raw: Any, section: str) -> _Settings:
    """The dataclass `cls` from the raw config section `section`, strictly.

    An unknown key, a missing required key or a value of the wrong type fails, naming
    `section.key`. Field types: float (int or float, never bool), bool, int (not bool),
    `tuple[float, ...]` (a list of numbers) and `float | None`.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section}: oczekiwano mapowania, dostalem {type(raw).__name__}")

    declared = {f.name: f for f in fields(cls)}
    unknown = sorted(str(key) for key in raw if key not in declared)
    if unknown:
        raise ValueError("nieznane klucze: " + ", ".join(f"{section}.{key}" for key in unknown))

    missing = [name for name, f in declared.items()
               if name not in raw and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError("brak wymaganych kluczy: " + ", ".join(f"{section}.{name}" for name in missing))

    # `from __future__ import annotations` leaves the annotations as strings.
    hints = get_type_hints(cls)
    values = {name: _setting(raw[name], hints[name], f"{section}.{name}") for name in declared if name in raw}
    return cls(**values)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_optional_float(hint: Any) -> bool:
    return get_origin(hint) in (Union, types.UnionType) and set(get_args(hint)) == {float, type(None)}


def _setting(value: Any, hint: Any, where: str) -> Any:
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{where}: oczekiwano true albo false, dostalem {value!r}")

    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise TypeError(f"{where}: oczekiwano liczby calkowitej, dostalem {value!r}")

    if hint is float:
        if _is_number(value):
            return float(value)
        raise TypeError(f"{where}: oczekiwano liczby, dostalem {value!r}")

    if hint == tuple[float, ...]:
        if isinstance(value, (list, tuple)) and all(_is_number(v) for v in value):
            return tuple(float(v) for v in value)
        raise TypeError(f"{where}: oczekiwano listy liczb, dostalem {value!r}")

    if _is_optional_float(hint):
        if value is None:
            return None
        if _is_number(value):
            return float(value)
        raise TypeError(f"{where}: oczekiwano liczby albo null, dostalem {value!r}")

    raise TypeError(f"{where}: typ pola {hint!r} nie jest obslugiwany")


@dataclass
class A1XConfig:
    """The `a1x:` section of the twin's config: Galaxea A1X over CAN-FD.

    The arm is purely position-controlled: only p_des acts and kp/kd/v_des/t_ff have no effect,
    but kp 0 leaves it deaf to commands (diag/REPORT.md), hence kp > 0.
    The arm has no brakes: disconnecting never releases the motors (FF 2), and an uncontrolled
    A1X holds the pose it stands in (docs/SAFETY.md).
    """

    #: Gains sent in 0x050 (never 0; only p_des has authority).
    kp: float = 20.0
    kd: float = 1.0
    gripper_kp: float = 25.0
    #: Gripper p_des [rad, same units as group 7 of frame 0x052] at 100 (open) and at 0
    #: (closed); more negative = wider. Measured 2026-09-27 (`python -m galaxeo.xcan_usb
    #: --grip-test`, kp 20): closed stop about +0.039 rad (+2.2 deg), open about -1.99 rad
    #: (-114 deg). The old +0.6 pushed the jaws 0.56 rad past the stop: the stall current
    #: (|effort| 7.8 instead of 0.2) overheated the gripper motor. Closed = 0.04 rad of pressure
    #: past the stop (objects are held by the `gripper_force` contact stop anyway), open = a hair
    #: before the stop so it does not lean on it.
    gripper_open_pdes: float = -1.95
    gripper_closed_pdes: float = 0.08
    #: Group 7 position from 0x052 [rad] at 0 and at 100: the stops measured 2026-09-27 (above).
    #: Only valid with a driven gripper (from the first 0x051 in the connection): at rest after
    #: power-on group 7 reads an arbitrary value (-165 deg with the jaws at -85 deg).
    #: None (both) = feedback not measured: the backend reports the last command (open loop).
    gripper_fb_closed: float | None = 0.039
    gripper_fb_open: float | None = -1.992
    #: What the backend reports as the gripper before the first command when feedback is not
    #: measured. 0 = closed: the station convention (home = jaws closed, "Set home = here"
    #: confirms it).
    gripper_start: float = 0.0
    #: Gripper p_des slew [rad/s]; the full 2.03 travel takes about 1.0 s.
    gripper_rate_pdes_s: float = 2.0
    #: |effort| of group 7 this far above rest (median of the first 0.1 s of closing) = contact:
    #: the target freezes (it grips, does not crush).
    gripper_force: float = 1.2
    #: Arm joint zero offset [deg]: subtracted on read, added on send
    #: (J3 reads about +1.49 deg at rest - measure yours).
    joint_offsets_deg: tuple[float, ...] = (0.0,) * 6
    #: Rate of the 0x050 / 0x051 stream [Hz] (the arm answers 0x052 every 5 ms).
    stream_hz: float = 200.0
    #: The backend's own p_des speed limit [deg/s]: a second layer below the twin's supervisor.
    max_step_deg_s: float = 90.0
    #: No 0x052 for longer than this [ms] = the link is silent: sending stops, the twin gets an error.
    stale_ms: float = 150.0
    #: This many identical 0x052 frames in a row = frozen telemetry (arm released, FF 2).
    frozen_frames: int = 50
    #: A command-measurement gap [deg] lasting `deaf_s` on a stationary joint = arm deaf or blocked.
    deaf_err_deg: float = 3.0
    deaf_s: float = 1.0
    #: FF 1 -> 5 -> 6 enable sequence right after connecting. Off by default: a freshly powered
    #: arm listens right away, and FF 5 briefly releases the motors. The panel has a button for it.
    enable_on_connect: bool = False
    #: Gripper over XCAN (macOS): on by default. Measured 2026-09-27 (`python -m galaxeo.xcan_usb
    #: --grip-test`): 0x051 gets through, the jaws travelled 85-116 degrees. The earlier refusal
    #: rested on an unverified comment. False = refuse ("no gripper"); env A1X_GRIPPER_XCAN=1 forces True.
    gripper_over_xcan: bool = True

    def __post_init__(self) -> None:
        if self.kp <= 0 or self.gripper_kp <= 0:
            raise ValueError("a1x.kp i a1x.gripper_kp musza byc > 0 - przy kp 0 ramie A1X przestaje "
                             "sluchac rozkazow")
        if len(self.joint_offsets_deg) != 6:
            raise ValueError("a1x.joint_offsets_deg: potrzeba 6 wartosci (arm_joint1..6)")
        self.joint_offsets_deg = tuple(float(x) for x in self.joint_offsets_deg)
        if (self.gripper_fb_closed is None) != (self.gripper_fb_open is None):
            raise ValueError("a1x.gripper_fb_closed i gripper_fb_open: podaj oba albo zadne")
        if self.gripper_fb_closed is not None and abs(self.gripper_fb_open - self.gripper_fb_closed) < 1e-6:
            raise ValueError("a1x.gripper_fb_closed i gripper_fb_open nie moga byc rowne")
        if self.stream_hz <= 0 or self.max_step_deg_s <= 0 or self.gripper_rate_pdes_s <= 0:
            raise ValueError("a1x.stream_hz, max_step_deg_s i gripper_rate_pdes_s musza byc > 0")


class BackendContext(Protocol):
    """The fields of the twin's `BackendContext` this backend reads. Duck-typed plain data."""

    @property
    def joints(self) -> Sequence[str]: ...

    @property
    def gripper(self) -> str | None: ...

    @property
    def limits(self) -> Mapping[str, tuple[float, float]]: ...

    @property
    def port(self) -> str | None: ...

    @property
    def settings(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BackendInfo:
    """What the twin shows about the connected arm (HUD, logs)."""

    name: str
    description: str
    simulated: bool


@dataclass(frozen=True)
class GripperState:
    """Read-only snapshot of the gripper as the backend drives and measures it (`gripper_state()`).

    Openings in app units: 0 = closed, 100 = open.
    """

    commanded: float | None        # last command the backend let through; None before the first
    target: float | None           # where the stream goes (differs from `commanded` once frozen)
    streamed: float | None         # opening really sent last; the best guess of the jaw without a measurement
    position: float | None         # measured opening; None while the backend does not measure it
    frozen: bool                   # contact while closing: the target stands where the stream was
    effort: float | None           # raw A1X scale; None without fresh feedback
    t: float                       # backend clock when the reading arrived (the queue drain)
    age_s: float | None            # backend clock now minus `t`; None before any reading
    rate: float                    # stream speed [units/s]
    force_threshold: float         # effort above rest at which a closing jaw stops
    measured: bool                 # whether `position` is a measurement


@dataclass(frozen=True)
class _Sent:
    """A payload this backend sent, for telling its echo from another sender."""

    data: bytes
    t: float                       # backend clock at the send


def create(ctx: BackendContext) -> A1XArm:
    """Entry point of the `a1x` backend plugin (`lerobot_mp.backends`)."""
    return A1XArm(ctx)


class A1XArm:
    """Galaxea A1X: 6 arm joints + gripper over CAN-FD.

    `threaded=False` (tests): no IO thread - `io_step(now)` runs the exchange ticks, and `enable()`
    measures time through `sleep` and calls `io_step` itself.
    """

    def __init__(self, ctx: BackendContext, *, bus: Any = None, clock: Callable[[], float] = time.monotonic,
                 threaded: bool = True, sleep: Callable[[float], None] = time.sleep,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S):
        self.acfg: A1XConfig = load_settings(A1XConfig, ctx.settings, "a1x")
        self.gripper = ctx.gripper
        self.arm = tuple(n for n in ctx.joints if n != ctx.gripper)
        if len(self.arm) != P.N_JOINTS or ctx.gripper is None:
            raise ValueError("backend a1x potrzebuje 6 stawow ramienia i chwytaka - stanowisko robot: a1x")

        #: Limits `send_joints` clamps to; the twin moves them through `set_joint_limits`.
        self.limits: dict[str, tuple[float, float]] = dict(ctx.limits)
        self.iface = ctx.port or None
        self._bus = bus
        self._own_bus = bus is None
        self._clock = clock
        self._sleep = sleep
        self._threaded = threaded
        self._connect_timeout = connect_timeout_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._off = tuple(math.radians(x) for x in self.acfg.joint_offsets_deg)
        self._reset_state()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(name="Galaxea A1X (CAN)", description=f"lacze {self.iface or 'CAN'}", simulated=False)

    def _reset_state(self) -> None:
        self._fb = None                              # latest decoded 0x052 (galaxeo Feedback)
        self._fb_t = -math.inf                       # when it arrived, on the backend's clock
        self._fb_raw: bytes | None = None
        self._identical = 0                          # identical frames in a row (frozen telemetry)
        self._n_fb = 0
        self._start_rad: tuple[float, ...] | None = None
        self._bounds: tuple[tuple[float, float], ...] = tuple(P.URDF_LIMITS_RAD)
        self._p_target: list[float] | None = None    # arm target [CAN rad]; None = nothing is sent
        self._p_stream: list[float] | None = None    # what really goes in 0x050 (slewed from the measurement)
        self._g_target: float | None = None          # gripper p_des target; None = no 0x051
        self._g_stream: float | None = None
        self._g_frozen = False                       # contact while closing: the target stands
        self._g_cmd: float | None = None             # last gripper command [0..100] or `assume_gripper`
        self._g_assumed = False                      # `_g_cmd` from `assume_gripper`, stream not started yet
        self._g_rest_eff: list[float] = []           # |effort| at the start of a close (rest)
        self._g_rest: float | None = None            # their median; None = not closing, or still collecting
        self._g_hi: float | None = None              # tightest p_des of this close (reference for reopening)
        self._g_live_n: int | None = None            # `_n_fb` at the first 0x051 (since connect / link loss)
        self._goal: dict[str, float] = {}            # latest targets in app units
        self._pinned = False                         # enable sequence running: p_des = measurement
        self._last_io: float | None = None
        self._deaf_since: dict[int, float] = {}
        self._deaf: str = ""
        self._bus_error: str = ""
        self._link_silent = False
        self._stale: list[str] = []
        self.tx_frames = 0
        self._sent: dict[int, deque[_Sent]] = {}
        self._foreign = 0
        self._foreign_t = -math.inf
        self._gripper_state_failed = False

    def _forget_grip_rest(self) -> None:
        self._g_rest_eff = []
        self._g_rest = None
        self._g_hi = None

    def _pdes(self, v: float) -> float:
        """Gripper 0..100 -> p_des (linear between closed and open)."""
        a = self.acfg
        return a.gripper_closed_pdes + v / 100.0 * (a.gripper_open_pdes - a.gripper_closed_pdes)

    def _pdes_app(self, p: float | None) -> float | None:
        """Gripper p_des -> 0..100, the inverse of `_pdes`; None stays None."""
        if p is None:
            return None
        a = self.acfg
        return (p - a.gripper_closed_pdes) * 100.0 / (a.gripper_open_pdes - a.gripper_closed_pdes)

    def _grip_app(self) -> float:
        """Gripper 0..100: measured through the two points, or (without them) the last command / assumption."""
        a = self.acfg
        if self.gripper_measured():
            span = a.gripper_fb_open - a.gripper_fb_closed
            return (self._fb.pos[6] - a.gripper_fb_closed) * 100.0 / span
        return a.gripper_start if self._g_cmd is None else self._g_cmd

    def _measured_rad(self) -> list[float]:
        assert self._fb is not None
        return list(self._fb.pos[:6])

    def _bound_deg(self, i: int) -> tuple[float, float]:
        """URDF limits of arm joint `i` widened to the start pose, in app degrees."""
        lo, hi = self._bounds[i]
        return (math.degrees(lo - self._off[i]), math.degrees(hi - self._off[i]))

    def connect(self) -> None:
        if self._connected:
            return
        self._reset_state()
        if self._bus is None:
            self.iface = self.iface or can.default_iface()
            self._bus = open_bus(self.iface)
        try:
            self._await_feedback()
        except Exception:
            self._close_bus()
            raise

        self._start_rad = tuple(self._fb.pos[:6])
        self._bounds = tuple((min(lo, s), max(hi, s))
                             for (lo, hi), s in zip(P.URDF_LIMITS_RAD, self._start_rad))
        self._link_silent = False
        self._stale = []
        self._connected = True

        g7 = self._fb.pos[6]
        logger.info("A1X chwytak: grupa 7 w spoczynku %.3f rad (%.1f st.) - pomiar dopiero po pierwszej 0x051 "
                    "(w spoczynku po wlaczeniu zasilania grupa 7 bywa dowolna)", g7, math.degrees(g7))
        logger.info("A1X na %s: poza startowa %s", self.iface or "podanym laczu",
                    ", ".join(f"{n}={v:.1f}" for n, v in self.read_joints().items()))

        self._start_thread()
        if self.acfg.enable_on_connect:
            self.enable()

    def _await_feedback(self) -> None:
        """The first 0x052 frame within `connect_timeout_s`, then a moment of collecting for the frozen test."""
        where = self.iface or "laczu"
        t0 = time.monotonic()
        while self._n_fb == 0:
            self._drain(self._clock())
            if self._n_fb:
                break
            if time.monotonic() - t0 > self._connect_timeout:
                raise RuntimeError(
                    f"brak ramek 0x052 na {where} - lacze CAN stoi (Linux: ./can_up.sh albo "
                    + can.IP_LINK_UP.format(iface=where) + ") albo ramie bez zasilania")
            time.sleep(0.002)

        # `_identical` counts REPEATS, so n frames give at most n - 1: the "frozen" verdict
        # (> frozen_frames) needs frozen_frames + 2 frames. On hardware they arrive one by one (every 5 ms).
        frozen = self.acfg.frozen_frames
        settle = max(SETTLE_S, 2.0 * (frozen + 2) * FB_PERIOD_S)
        t1 = time.monotonic()
        while self._n_fb < frozen + 2 and time.monotonic() - t1 < settle:
            if not self._drain(self._clock()):
                time.sleep(0.002)
        if self._identical > frozen:
            raise RuntimeError("telemetria zamrozona - ramie zwolnione (FF 2), wylacz i wlacz zasilanie")

    def _start_thread(self) -> None:
        if not self._threaded:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="a1x-can", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        """Ends the stream and closes the link. Never releases the motors (FF 2): the A1X has no brakes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._p_target = None
            self._g_target = None
            self._connected = False
        self._close_bus()

    def _close_bus(self) -> None:
        bus = self._bus
        if bus is None:
            return
        if self._own_bus:
            self._bus = None                   # the next connect opens the link again
        try:
            bus.close()
        except Exception:                      # pragma: no cover - disconnecting must not crash the app
            logger.exception("Blad przy zamykaniu lacza CAN")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _loop(self) -> None:
        period = 1.0 / self.acfg.stream_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.io_step(self._clock())
                self._bus_error = ""
            except Exception as exc:           # adapter pulled, link down: an error for the twin, not a dead thread
                if not self._bus_error:
                    logger.error("A1X: blad lacza CAN: %s", exc)
                self._bus_error = str(exc)
                self._stop.wait(0.1)
                continue
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -0.05:
                nxt = time.monotonic()         # running late: no catch-up burst
            elif delay > 0:
                self._stop.wait(delay)

    def _drain(self, now: float) -> bool:
        """Takes EVERY frame from the queue (reading only part of it gives readings seconds old)."""
        got = False
        while True:
            f = self._bus.recv(0.0)
            if f is None:
                return got
            if f.can_id in (P.CMD_ID, P.GRIP_ID, P.FF_ID):
                self._check_sender(f.can_id, bytes(f.data), now)
                continue
            if f.can_id != P.FB_ID or len(f.data) != P.FB_LEN:
                continue
            data = bytes(f.data)
            self._fb = P.decode_feedback(data)
            self._identical = self._identical + 1 if data == self._fb_raw else 0
            self._fb_raw = data
            self._fb_t = now
            self._n_fb += 1
            got = True

    def _stale_now(self, now: float) -> bool:
        return (now - self._fb_t) * 1000.0 > self.acfg.stale_ms

    def _frozen(self) -> bool:
        return self._identical > self.acfg.frozen_frames

    def io_step(self, now: float) -> None:
        """One exchange tick: receive every frame, then (when allowed) 0x050 and 0x051."""
        with self._lock:
            if self._bus is None:
                return
            self._drain(now)
            dt = (1.0 / self.acfg.stream_hz if self._last_io is None
                  else min(max(now - self._last_io, 0.0), 2.0 / self.acfg.stream_hz))
            self._last_io = now

            if self._fb is None or self._stale_now(now) or self._frozen():
                # Stale or frozen telemetry: send nothing and drop the targets. The twin stops and
                # moves again only from a confirmed measurement - the backend must not drive on its
                # own to a target from before the gap meanwhile (an arm power-cycled or blocked stands
                # somewhere else, and the A1X has no brakes). When frames return nothing goes out until
                # the next `send_joints`: the arm holds the pose it stands in, the gripper its last p_des.
                self._p_target = None
                self._p_stream = None
                self._g_target = None
                self._g_stream = None
                self._g_frozen = False
                self._forget_grip_rest()
                self._g_live_n = None                  # the arm may have been off: group 7 is uncertain again
                self._deaf_since = {}
                self._deaf = ""
                return

            if self._pinned:
                self._p_target = self._measured_rad()
                self._p_stream = list(self._p_target)

            if self._p_target is not None:
                if self._p_stream is None:
                    self._p_stream = self._measured_rad()     # start from the MEASUREMENT, not the target
                step = math.radians(self.acfg.max_step_deg_s) * dt
                self._p_stream = [s + max(-step, min(step, t - s)) for s, t in zip(self._p_stream, self._p_target)]
                self._send(P.CMD_ID, P.encode_arm(self._p_stream, self.acfg.kp, self.acfg.kd))
                self._watch_deaf(now)

            if self._g_target is not None:
                if self._g_stream is None:
                    self._g_stream = self._pdes(self._grip_app())
                closing = self._g_target > self._g_stream
                if closing:
                    self._g_hi = self._g_stream if self._g_hi is None else max(self._g_hi, self._g_stream)
                if self._g_target < self._g_stream:
                    # Opening: the rest is measured again only once the jaws moved GRIP_REOPEN_PDES away
                    # from the tightest point of the close. Opened by one slider notch the jaws still
                    # press the cube, and a new "rest" would be that pressure - the threshold grew with
                    # every cycle. A reached (or frozen) target does not reset the rest either, or a
                    # slider closed in small steps would take the pressure for rest too.
                    hi = self._g_hi
                    if self._g_rest is None or hi is None or self._g_stream < hi - GRIP_REOPEN_PDES:
                        self._forget_grip_rest()
                elif closing and self._g_rest is None:
                    # Start of a close: group 7's rest effort (unknown, sometimes a steady ~3) from the
                    # first ticks. A fixed 1.2 threshold froze the target on the very first tick and the
                    # gripper stopped listening.
                    self._g_rest_eff.append(abs(self._fb.eff[6]))
                    if len(self._g_rest_eff) >= GRIP_REST_STEPS:
                        e = sorted(self._g_rest_eff)
                        self._g_rest = e[len(e) // 2]
                elif closing and self._grip_contact():
                    # Contact: the target stops where the stream is - it grips, does not crush.
                    self._g_target = self._g_stream
                    self._g_frozen = True
                step = self.acfg.gripper_rate_pdes_s * dt
                self._g_stream += max(-step, min(step, self._g_target - self._g_stream))
                self._send(P.GRIP_ID, P.encode_gripper(self._g_stream, self.acfg.gripper_kp, self.acfg.kd))
                if self._g_live_n is None:
                    self._g_live_n = self._n_fb

    def _grip_contact(self) -> bool:
        """Gripper |effort| above the rest measured at the start of this close by more than `gripper_force`."""
        return self._g_rest is not None and abs(self._fb.eff[6]) > self._g_rest + self.acfg.gripper_force

    def _send(self, can_id: int, data: bytes, fd: bool = True) -> None:
        self._sent.setdefault(can_id, deque(maxlen=ECHO_KEEP)).append(_Sent(data=bytes(data), t=self._clock()))
        self._bus.send(can_id, data, fd=fd)
        self.tx_frames += 1

    def _check_sender(self, can_id: int, data: bytes, now: float) -> None:
        """A command frame on the link: our own echo, or another sender (counted, a fault for a while)."""
        received_after = now if self._last_io is None else min(self._last_io, now)
        if self._own_payload(can_id, data, received_after):
            return

        if now - self._foreign_t >= FOREIGN_WINDOW_S:
            logger.warning("A1X: ramka 0x%03X spoza blizniaka - inny nadawca na laczu CAN", can_id)
        self._foreign += 1
        self._foreign_t = now

    def _own_payload(self, can_id: int, data: bytes, received_after: float) -> bool:
        """Whether `data`, received after the time `received_after`, is the echo of a frame of ours on
        `can_id` sent at most ECHO_WINDOW_S before; one send explains one echo."""
        sent = self._sent.get(can_id, ())
        for i, mine in enumerate(sent):
            if received_after - mine.t > ECHO_WINDOW_S:
                continue
            # XCAN pads FD frames with zeros to an allowed length (0x050: 60 -> 64, 0x051: 10 -> 12).
            padded = (len(data) > len(mine.data) and data.startswith(mine.data)
                      and not any(data[len(mine.data):]))
            if data == mine.data or padded:
                del sent[i]
                return True
        return False

    def _watch_deaf(self, now: float) -> None:
        """Command far from the measurement while the joint stands for `deaf_s` = arm deaf to 0x050 or blocked."""
        a = self.acfg
        fb = self._fb
        worst = ""
        for i, name in enumerate(self.arm):
            err = math.degrees(abs(self._p_stream[i] - fb.pos[i]))
            if err > a.deaf_err_deg and abs(math.degrees(fb.vel[i])) < STILL_DEG_S:
                t0 = self._deaf_since.setdefault(i, now)
                if now - t0 >= a.deaf_s and not worst:
                    worst = (f"{name}: {err:.0f} st. od rozkazu przez {now - t0:.1f} s i stoi - ramie gluche na "
                             f"0x050 (Wlacz silniki A1X, FF 1-5-6) albo zablokowane")
            else:
                self._deaf_since.pop(i, None)
        self._deaf = worst

    def read_joints(self) -> dict[str, float]:
        """Arm degrees (minus the zero offset) + gripper 0..100. On a silent link: the last values."""
        with self._lock:
            if self._fb is None:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            now = self._clock()
            self._link_silent = self._stale_now(now) or self._frozen()
            self._stale = list(self.arm) if self._link_silent else []
            out = {name: math.degrees(self._fb.pos[i] - self._off[i]) for i, name in enumerate(self.arm)}
            out[self.gripper] = self._grip_app()
            return out

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Arm and gripper target. Returns the clamped COMMAND in app units.

        Gripper: returns the command, not the p_des frozen on contact - the twin's supervisor must
        keep seeing a command tighter than the measurement, or it does not recognise a grip
        (`_grip_squeezing`, "in hand" in cube tracking). `{}` on a silent link and during the
        enable sequence.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            if self._link_silent or self._pinned or self._fb is None:
                return {}
            sent: dict[str, float] = {}

            arm = [n for n in self.arm if n in targets]
            if arm:
                p = self._p_target if self._p_target is not None else self._measured_rad()
                p = list(p)
                for i, name in enumerate(self.arm):
                    if name not in targets:
                        continue
                    lo, hi = self.limits[name]
                    bound_lo, bound_hi = self._bound_deg(i)
                    v = min(max(float(targets[name]), lo, bound_lo), hi, bound_hi)
                    p[i] = math.radians(v) + self._off[i]
                    sent[name] = v
                self._p_target = p

            if self.gripper in targets:
                lo, hi = self.limits[self.gripper]
                v = min(max(float(targets[self.gripper]), lo, 0.0), hi, 100.0)
                reported = self._grip_app()
                if self._g_target is not None or abs(v - reported) > GRIP_START_DELTA:
                    want = self._pdes(v)
                    if self._g_target is None:
                        logger.info("A1X chwytak: pierwszy rozkaz %.0f (zglaszane %.0f%s) -> p_des %.2f", v, reported,
                                    "" if self.gripper_measured() else ", bez pomiaru", want)
                        # First command: the stream starts from the REPORTED opening, not from the target.
                        stream = self._pdes(reported)
                        if self._g_assumed or (self._g_cmd is None and not self.gripper_measured()):
                            # The opening is only guessed: `assume_gripper` ("Set home = here": closed)
                            # or just `gripper_start` after Connect (the A1X keeps its last pose between
                            # sessions, the jaws may stand open). Start from the more open of {guess,
                            # target}: opening goes straight to the target (an empty closed gripper loses
                            # nothing, an open one does not twitch), and a wrong guess does not snap the
                            # jaws shut in the first frame (closed p_des at kp 25, no force threshold -
                            # that one works only while closing). `_g_cmd` from a command that really
                            # went out survives a link gap, and the next one ramps from it.
                            stream = min(stream, want)
                        self._g_target = want
                        self._g_stream = stream
                        self._g_assumed = False
                    elif want <= self._g_target:
                        # Opening (or the same target) always passes and lifts the freeze. With a strict `<`
                        # a target frozen at the open end (`gripper_open_pdes`) could not be moved any more.
                        self._g_target = want
                        self._g_frozen = False
                    elif not self._g_frozen:
                        if self._grip_contact() and self._g_stream is not None:
                            self._g_target = self._g_stream
                            self._g_frozen = True
                        else:
                            self._g_target = want
                    self._g_cmd = v
                sent[self.gripper] = v

            self._goal.update(sent)
            return sent

    def _fresh_fb(self) -> Any:
        """Latest 0x052 frame, or None when missing, stale or frozen."""
        if not self._connected or self._fb is None or self._stale_now(self._clock()) or self._frozen():
            return None
        return self._fb

    def efforts(self) -> dict[str, float] | None:
        """Raw A1X effort scale from frame 0x052; arm joints only, no gripper."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return {name: float(fb.eff[i]) for i, name in enumerate(self.arm)}

    def gripper_effort(self) -> float | None:
        """|effort| of the gripper group (frame 0x052, eff[6], raw A1X scale), or None without fresh feedback.

        Readable even where 0x051 cannot be sent: the load of jaws pressed against something.
        """
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return abs(float(fb.eff[6]))

    @property
    def gripper_force(self) -> float:
        """Gripper effort above rest that means contact (config `gripper_force`)."""
        return float(self.acfg.gripper_force)

    def velocities(self) -> dict[str, float] | None:
        """deg/s from frame 0x052; arm joints only, no gripper."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return {name: math.degrees(fb.vel[i]) for i, name in enumerate(self.arm)}

    def gripper_state(self) -> GripperState | None:
        """Command, stream, freeze and effort of the jaw, read under the lock; openings 0..100.

        Changes nothing and sends nothing. None without a connection and on any error (logged
        once): a grasp loop asks every tick and must not crash on it.
        """
        try:
            with self._lock:
                if not self._connected:
                    return None

                a = self.acfg
                span = a.gripper_open_pdes - a.gripper_closed_pdes
                fb = self._fresh_fb()
                measured = self.gripper_measured()
                return GripperState(
                    commanded=self._g_cmd,
                    target=self._pdes_app(self._g_target),
                    streamed=self._pdes_app(self._g_stream),
                    position=self._grip_app() if measured else None,
                    frozen=self._g_frozen,
                    effort=None if fb is None else float(fb.eff[6]),
                    t=self._fb_t,
                    age_s=None if self._fb is None else self._clock() - self._fb_t,
                    rate=a.gripper_rate_pdes_s * 100.0 / abs(span),
                    force_threshold=a.gripper_force,
                    measured=measured,
                )
        except Exception:
            if not self._gripper_state_failed:
                self._gripper_state_failed = True
                logger.exception("A1X: nie udalo sie odczytac stanu chwytaka")
            return None

    def foreign_frames(self) -> int:
        """Command frames (0x050, 0x051, FF) of another sender received since connecting."""
        with self._lock:
            return self._foreign

    def goal_positions(self) -> dict[str, float]:
        """Latest targets [app units]; `{}` before the first command (p_des cannot be read back)."""
        with self._lock:
            return dict(self._goal)

    def faults(self) -> list[str]:
        try:
            with self._lock:
                if not self._connected:
                    return []
                out: list[str] = []
                now = self._clock()
                if self._bus_error:
                    out.append(f"magistrala CAN: {self._bus_error}")
                if self._stale_now(now):
                    out.append(f"brak odpowiedzi ramienia od {(now - self._fb_t) * 1000.0:.0f} ms")
                elif self._frozen():
                    out.append(f"telemetria zamrozona ({self._identical} identycznych ramek) - ramie zwolnione "
                               f"(FF 2)? wylacz i wlacz zasilanie")
                elif self._deaf:
                    out.append(self._deaf)
                state = str(getattr(self._bus, "state", "ok") or "ok")
                if state in ("bus-off", "error-passive"):
                    out.append(f"magistrala CAN: {state}")
                if now - self._foreign_t < FOREIGN_WINDOW_S:
                    out.append(FOREIGN_FAULT)
                return out
        except Exception as exc:               # pragma: no cover - called every control cycle, must not crash it
            logger.exception("Nie udalo sie sprawdzic stanu A1X")
            return [f"nie udalo sie sprawdzic stanu A1X: {exc}"]

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """URDF limits [deg] widened to the start pose (J3 stands ~1.5 deg past its limit); no gripper."""
        return {name: self._bound_deg(i) for i, name in enumerate(self.arm)}

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        with self._lock:
            for name, (lo, hi) in limits.items():
                if name in self.limits:
                    self.limits[name] = (float(lo), float(hi))

    def gripper_known(self) -> bool:
        """Whether the gripper position is known: measured, a command that REALLY went out in 0x051,
        or `assume_gripper`. A target equal to the reported default (`GRIP_START_DELTA`) sends
        nothing and settles nothing."""
        with self._lock:
            return self.gripper_measured() or self._g_cmd is not None

    def assume_gripper(self, v: float) -> bool:
        """The operator confirms the gripper opening without a measurement ("Set home = here": jaws closed = 0).

        Only while the 0x051 stream has not started: after that the last command that really went
        out is the truth, and the assumption is refused (False, a warning in the log). After a
        successful one `gripper_known()` is True, so the next commands ramp from the assumption; the
        first stream starts from the more open of {assumption, target} (`send_joints`, the same as
        from `gripper_start` alone). A measured gripper assumes nothing (False).
        """
        with self._lock:
            if self.gripper_measured():
                return False
            if self._g_target is not None:
                logger.warning("A1X chwytak juz sterowany (p_des %.2f) - zalozenie %.0f odrzucone", self._g_target, v)
                return False
            self._g_cmd = min(max(float(v), 0.0), 100.0)
            self._g_assumed = True
            logger.info("A1X chwytak: zalozone rozwarcie %.0f (bez pomiaru)", self._g_cmd)
            return True

    def gripper_drivable(self) -> bool:
        """Whether the twin should send gripper commands. Over XCAN (macOS) yes, since it was measured
        on 2026-09-27 that 0x051 gets through (jaws moved 85-116 degrees, `python -m galaxeo.xcan_usb
        --grip-test`); the earlier refusal rested on an unverified comment. `a1x.gripper_over_xcan:
        false` restores the refusal."""
        return not self._on_xcan() or self._gripper_xcan_allowed()

    def gripper_xcan_trial(self) -> bool:
        """XCAN with the gripper on (`a1x.gripper_over_xcan`, the default, or A1X_GRIPPER_XCAN=1):
        commands go out; the panel's status line shows it (without feedback it reminds that it shows
        the command)."""
        return self._on_xcan() and self._gripper_xcan_allowed()

    def _on_xcan(self) -> bool:
        return str(self.iface or "").lower().startswith("xcan")

    def _gripper_xcan_allowed(self) -> bool:
        env = os.environ.get(GRIPPER_XCAN_ENV, "").strip().lower() in ("1", "true", "yes", "tak")
        return bool(self.acfg.gripper_over_xcan) or env

    def gripper_travel_s(self) -> float:
        """Time of a full gripper stroke [s]: |open - closed| / `gripper_rate_pdes_s` + margin (1.2 s with
        the defaults: 2.03 / 2.0 + 0.2). How long to wait after a command for the jaws to really get there."""
        a = self.acfg
        return abs(a.gripper_open_pdes - a.gripper_closed_pdes) / a.gripper_rate_pdes_s + GRIP_TRAVEL_MARGIN_S

    def gripper_measured(self) -> bool:
        """Whether the gripper reading is measured: both `gripper_fb_*` points are set AND the gripper is
        driven in this connection (at least `GRIP_LIVE_FRAMES` 0x052 frames after the first 0x051, again
        after a link loss). At rest after power-on group 7 reads an arbitrary value (2026-09-27: -165 deg
        with the jaws at -85 deg), so until then the backend reports the last command, and before the
        first one `gripper_start` (or `assume_gripper`), which need not match the jaws."""
        with self._lock:
            a = self.acfg
            return (a.gripper_fb_closed is not None and a.gripper_fb_open is not None and self._fb is not None
                    and self._g_live_n is not None and self._n_fb - self._g_live_n >= GRIP_LIVE_FRAMES)

    def calibration_warnings(self) -> list[str]:
        a = self.acfg
        if a.gripper_fb_closed is None or a.gripper_fb_open is None:
            return ["chwytak A1X: sprzezenie niezmierzone - panel pokazuje rozkaz, nie pomiar; "
                    "zmierz a1x.gripper_fb_closed/open"]
        return []

    def enable(self) -> str:
        """The FF 1 -> 5 -> 6 sequence, with p_des = the measurement before, between and after each code.

        Only on request (the panel or `a1x.enable_on_connect`): a freshly powered arm listens right
        away, and FF 5 briefly releases the motors. Targets from `send_joints` are refused meanwhile.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            if self._fb is None or self._stale_now(self._clock()):
                raise RuntimeError("A1X: brak swiezych ramek 0x052 - wlaczenie odrzucone")
            if self._frozen():
                raise RuntimeError("A1X: telemetria zamrozona (FF 2) - wylacz i wlacz zasilanie; FF 1-5-6 tego "
                                   "nie naprawia")
            self._pinned = True
            self._p_target = self._measured_rad()
            self._p_stream = list(self._p_target)
        try:
            for code in P.ENABLE_SEQUENCE:
                self._wait(ENABLE_STEP_S)
                with self._lock:
                    if self._stale_now(self._clock()):
                        raise RuntimeError(f"A1X: lacze zamilklo w trakcie wlaczania (przed FF {code})")
                    self._send(P.FF_ID, P.encode_ff(code), fd=False)
            self._wait(ENABLE_STEP_S)
        finally:
            with self._lock:
                # The link broke meanwhile: no target (the last measurement before the gap is no longer the arm's pose).
                fresh = self._fb is not None and not self._stale_now(self._clock()) and not self._frozen()
                self._p_target = self._measured_rad() if fresh else None
                if not fresh:
                    self._p_stream = None
                self._pinned = False
        return "wyslano FF 1 -> 5 -> 6; ramie trzyma zmierzona poze"

    def _wait(self, secs: float) -> None:
        """Time between the codes: the IO thread streams on its own; without it, the ticks run here."""
        if self._threaded:
            self._sleep(secs)
            return
        period = 1.0 / self.acfg.stream_hz
        for _ in range(max(1, round(secs / period))):
            self._sleep(period)
            self.io_step(self._clock())
