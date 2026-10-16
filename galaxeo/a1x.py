"""Galaxea A1X over CAN-FD: the arm as a driven device, standard library only.

`A1X` connects to the arm, streams p_des from the measured pose, watches the link and drives the
gripper. It speaks wire units: arm joints J1..J6 by index 0..5 in radians (minus
`joint_offsets_deg`), the gripper as p_des [rad, more positive = tighter]. Protocol and CAN link come
from `galaxeo.protocol` and `galaxeo.bus`. The digital twin's `a1x` backend (`galaxeo.robot.backend`)
is a thin adapter on top of it.

    a1x = A1X(A1XConfig(), "can0")
    a1x.connect()                                 # waits for 0x052; sends nothing
    q = a1x.read().joints
    a1x.send({0: q[0] + 0.2})                     # J1 streams from the measured pose
    a1x.send(gripper=a1x.gripper_pdes(1.0))       # open
    a1x.disconnect()                              # ends the stream; never releases the motors

The gripper:

* p_des is linear in the opening, from `gripper_closed_pdes` (closed) to `gripper_open_pdes` (open):
  `gripper_pdes`. Targets are clamped to that stroke.
* Group 7 of frame 0x052 is the jaw position, rad like p_des (measured 2026-09-27: stops at +0.039
  closed, -1.992 open); `gripper_fb_closed/open` map it onto the stroke. Group 7 is valid ONLY while
  the gripper is driven: at rest after power-on it reads an arbitrary value (e.g. -2.88 rad = -165 deg
  with the jaws at -85 deg) and follows p_des only while 0x051 flows. So the measurement counts from
  `GRIP_LIVE_FRAMES` 0x052 frames after the first 0x051 of this connection (and again after a link
  loss); until then, and without `gripper_fb_*`, open loop: the last command. Before the first command
  open loop reports `gripper_start`, or what `assume_gripper` set. While the opening is only guessed,
  the first opening goes straight to the target and closing ramps.
* Closing stops on contact: group 7's |effort| above the rest of the first 0.1 s of the close by more
  than `gripper_force` freezes the target (it grips, does not crush).
* The gripper moves over XCAN (macOS): measured 2026-09-27 (`python -m galaxeo.xcan_usb --grip-test`),
  contrary to the caveat on `galaxeo.bus.XcanBus`. `gripper_over_xcan: false` restores the refusal
  (`gripper_drivable()` False); env A1X_GRIPPER_XCAN=1 forces driving.

Safety rules (docs/SAFETY.md, docs/PROTOCOL.md):

* Nothing goes on the bus before the first `send`: an uncommanded A1X holds the pose it stands in.
* The p_des stream starts from the MEASURED pose and moves at most `max_step_deg_s`, so a bad first
  command does not jump. The stream runs at `stream_hz` and repeats the last target.
* Stale (> `stale_ms`) or frozen (FF 2) telemetry stops sending, sets `link_silent` and puts an error in
  `faults()`: the caller stops and later starts from the measurement. Arm and gripper targets are
  dropped: when frames return, nothing goes out until the next `send`.
* Never FF 2 (release: an arm without brakes drops), never kp 0 (the arm goes deaf), never the 0x023
  heartbeat. The FF 1 -> 5 -> 6 enable only on request (`enable`), with p_des = the measurement
  before, between and after each code.
* `disconnect` ends the stream and closes the link; it never releases the motors.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from . import bus as can
from . import protocol as P

#: A1X_GRIPPER_XCAN=1 (or true / yes / tak) = the same as `gripper_over_xcan: true`.
GRIPPER_XCAN_ENV = "A1X_GRIPPER_XCAN"

logger = logging.getLogger(__name__)

#: A gripper target further than this from the reported position [fraction of the stroke] starts the
#: 0x051 stream: holding the reported position after `connect` sends nothing.
GRIP_START_DELTA = 0.005
#: A joint "stands" below this |velocity| [deg/s] (deaf arm detection).
STILL_DEG_S = 0.3
#: How long `connect` waits for the first 0x052 frame [s].
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

_Settings = TypeVar("_Settings")


def open_bus(iface: str) -> Any:
    """The CAN link from `galaxeo.bus` (`can0`, `xcan`, `xcan:6`, `PCAN_USBBUS1`, `virtual:x`)."""
    try:
        return can.open_bus(iface)
    except ImportError as exc:
        extra = "xcan" if iface.lower().startswith("xcan") else "pcan"
        raise ImportError(f'link {iface!r} needs an extra: pip install "galaxeo[{extra}]" ({exc})') from exc


def load_settings(cls: type[_Settings], raw: Any, section: str) -> _Settings:
    """The dataclass `cls` from the raw config section `section`, strictly.

    An unknown key, a missing required key or a value of the wrong type fails, naming
    `section.key`. Field types: float (int or float, never bool), bool, int (not bool),
    `tuple[float, ...]` (a list of numbers) and `float | None`.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section}: expected a mapping, got {type(raw).__name__}")

    declared = {f.name: f for f in fields(cls)}
    unknown = sorted(str(key) for key in raw if key not in declared)
    if unknown:
        raise ValueError("unknown keys: " + ", ".join(f"{section}.{key}" for key in unknown))

    missing = [name for name, f in declared.items()
               if name not in raw and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError("missing required keys: " + ", ".join(f"{section}.{name}" for name in missing))

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
        raise TypeError(f"{where}: expected true or false, got {value!r}")

    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise TypeError(f"{where}: expected an integer, got {value!r}")

    if hint is float:
        if _is_number(value):
            return float(value)
        raise TypeError(f"{where}: expected a number, got {value!r}")

    if hint == tuple[float, ...]:
        if isinstance(value, (list, tuple)) and all(_is_number(v) for v in value):
            return tuple(float(v) for v in value)
        raise TypeError(f"{where}: expected a list of numbers, got {value!r}")

    if _is_optional_float(hint):
        if value is None:
            return None
        if _is_number(value):
            return float(value)
        raise TypeError(f"{where}: expected a number or null, got {value!r}")

    raise TypeError(f"{where}: field type {hint!r} is not supported")


@dataclass
class A1XConfig:
    """Settings of the A1X driver; in a config file, the `a1x:` section.

    The arm is purely position-controlled: only p_des acts and kp/kd/v_des/t_ff have no effect,
    but kp 0 leaves it deaf to commands (diag/REPORT.md), hence kp > 0.
    The arm has no brakes: disconnecting never releases the motors (FF 2), and an uncontrolled
    A1X holds the pose it stands in (docs/SAFETY.md).
    """

    #: Gains sent in 0x050 (never 0; only p_des has authority).
    kp: float = 20.0
    kd: float = 1.0
    gripper_kp: float = 25.0
    #: Gripper p_des [rad, same units as group 7 of frame 0x052] at the open and at the closed end;
    #: more negative = wider. Measured 2026-09-27 (`python -m galaxeo.xcan_usb --grip-test`, kp 20):
    #: closed stop about +0.039 rad (+2.2 deg), open about -1.99 rad (-114 deg). The old +0.6 pushed
    #: the jaws 0.56 rad past the stop: the stall current (|effort| 7.8 instead of 0.2) overheated the
    #: gripper motor. Closed = 0.04 rad of pressure past the stop (objects are held by the
    #: `gripper_force` contact stop anyway), open = a hair before the stop so it does not lean on it.
    gripper_open_pdes: float = -1.95
    gripper_closed_pdes: float = 0.08
    #: Group 7 position from 0x052 [rad] closed and open: the stops measured 2026-09-27 (above).
    #: Only valid with a driven gripper (from the first 0x051 in the connection): at rest after
    #: power-on group 7 reads an arbitrary value (-165 deg with the jaws at -85 deg).
    #: None (both) = feedback not measured: the driver reports the last command (open loop).
    gripper_fb_closed: float | None = 0.039
    gripper_fb_open: float | None = -1.992
    #: The opening the driver reports before the first command while the gripper is not measured
    #: [per cent of the stroke, 0 = closed, 100 = open]. 0: a station's home has the jaws closed.
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
    #: The driver's own p_des speed limit [deg/s]: a second layer below the caller's.
    max_step_deg_s: float = 90.0
    #: No 0x052 for longer than this [ms] = the link is silent: sending stops, `faults()` reports it.
    stale_ms: float = 150.0
    #: This many identical 0x052 frames in a row = frozen telemetry (arm released, FF 2).
    frozen_frames: int = 50
    #: A command-measurement gap [deg] lasting `deaf_s` on a stationary joint = arm deaf or blocked.
    deaf_err_deg: float = 3.0
    deaf_s: float = 1.0
    #: FF 1 -> 5 -> 6 enable sequence right after connecting. Off by default: a freshly powered
    #: arm listens right away, and FF 5 briefly releases the motors.
    enable_on_connect: bool = False
    #: Gripper over XCAN (macOS): on by default. Measured 2026-09-27 (`python -m galaxeo.xcan_usb
    #: --grip-test`): 0x051 gets through, the jaws travelled 85-116 degrees. The earlier refusal
    #: rested on an unverified comment. False = refuse ("no gripper"); env A1X_GRIPPER_XCAN=1 forces True.
    gripper_over_xcan: bool = True

    def __post_init__(self) -> None:
        if self.kp <= 0 or self.gripper_kp <= 0:
            raise ValueError("a1x.kp and a1x.gripper_kp must be > 0 - at kp 0 the A1X arm stops "
                             "obeying commands")
        if len(self.joint_offsets_deg) != 6:
            raise ValueError("a1x.joint_offsets_deg: 6 values needed (arm_joint1..6)")
        self.joint_offsets_deg = tuple(float(x) for x in self.joint_offsets_deg)
        if (self.gripper_fb_closed is None) != (self.gripper_fb_open is None):
            raise ValueError("a1x.gripper_fb_closed and gripper_fb_open: set both or neither")
        if self.gripper_fb_closed is not None and abs(self.gripper_fb_open - self.gripper_fb_closed) < 1e-6:
            raise ValueError("a1x.gripper_fb_closed and gripper_fb_open must not be equal")
        if self.stream_hz <= 0 or self.max_step_deg_s <= 0 or self.gripper_rate_pdes_s <= 0:
            raise ValueError("a1x.stream_hz, max_step_deg_s and gripper_rate_pdes_s must be > 0")


@dataclass(frozen=True)
class Reading:
    """One `A1X.read()`: the latest 0x052 frame in wire units."""

    #: J1..J6 [rad], minus `joint_offsets_deg`.
    joints: tuple[float, ...]
    #: The gripper [p_des]: group 7 mapped onto the stroke once measured, else the last command, the
    #: assumption (`assume_gripper`) or `gripper_start`.
    gripper: float
    gripper_measured: bool


@dataclass(frozen=True)
class Accepted:
    """What `A1X.send` took."""

    #: Arm targets {J index: rad} after clamping to `limits()` and `bounds()`.
    joints: dict[int, float]
    #: The gripper command [p_des] after clamping to the stroke. None = no gripper target, or held back:
    #: the 0x051 stream does not run yet and the target is within `GRIP_START_DELTA` of the reported position.
    gripper: float | None


@dataclass(frozen=True)
class Fault:
    """Something wrong with the arm: on arm joint `joint` (index 0..5), or on the whole arm (None)."""

    joint: int | None
    text: str


class A1X:
    """Galaxea A1X: 6 arm joints + gripper over CAN-FD.

    `iface` names the CAN link for `galaxeo.bus.open_bus` (None = `galaxeo.bus.default_iface()`), or `bus`
    is an open link (tests). `threaded=False` (tests): no IO thread - `io_step(now)` runs the exchange
    ticks, and `enable()` measures time through `sleep` and calls `io_step` itself.
    """

    def __init__(self, cfg: A1XConfig | None = None, iface: str | None = None, *, bus: Any = None,
                 clock: Callable[[], float] = time.monotonic, threaded: bool = True,
                 sleep: Callable[[float], None] = time.sleep, connect_timeout_s: float = CONNECT_TIMEOUT_S):
        self.cfg = cfg if cfg is not None else A1XConfig()
        self.iface = iface or None
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
        self._off = tuple(math.radians(x) for x in self.cfg.joint_offsets_deg)
        self._limits: list[tuple[float, float]] = [(-math.inf, math.inf)] * P.N_JOINTS
        self._reset_state()

    def _reset_state(self) -> None:
        self._fb = None                              # latest decoded 0x052 (galaxeo Feedback)
        self._fb_t = -math.inf                       # when it arrived, on the driver's clock
        self._fb_raw: bytes | None = None
        self._identical = 0                          # identical frames in a row (frozen telemetry)
        self._n_fb = 0
        self._bounds: tuple[tuple[float, float], ...] = tuple(P.URDF_LIMITS_RAD)
        self._p_target: list[float] | None = None    # arm target [CAN rad]; None = nothing is sent
        self._p_stream: list[float] | None = None    # what really goes in 0x050 (slewed from the measurement)
        self._g_target: float | None = None          # gripper p_des target; None = no 0x051
        self._g_stream: float | None = None
        self._g_frozen = False                       # contact while closing: the target stands
        self._g_cmd: float | None = None             # last gripper command [p_des] or `assume_gripper`
        self._g_assumed = False                      # `_g_cmd` from `assume_gripper`, stream not started yet
        self._g_rest_eff: list[float] = []           # |effort| at the start of a close (rest)
        self._g_rest: float | None = None            # their median; None = not closing, or still collecting
        self._g_hi: float | None = None              # tightest p_des of this close (reference for reopening)
        self._g_live_n: int | None = None            # `_n_fb` at the first 0x051 (since connect / link loss)
        self._pinned = False                         # enable sequence running: p_des = measurement
        self._last_io: float | None = None
        self._deaf_since: dict[int, float] = {}
        self._deaf: Fault | None = None
        self._bus_error: str = ""
        self._link_silent = False
        self.tx_frames = 0

    def _forget_grip_rest(self) -> None:
        self._g_rest_eff = []
        self._g_rest = None
        self._g_hi = None

    def _grip_reported(self) -> float:
        """Gripper [p_des]: measured through the two points, else the last command / assumption / `gripper_start`."""
        a = self.cfg
        if self.gripper_measured():
            span = a.gripper_fb_open - a.gripper_fb_closed
            return self.gripper_pdes((self._fb.pos[6] - a.gripper_fb_closed) / span)
        return self.gripper_pdes(a.gripper_start / 100.0) if self._g_cmd is None else self._g_cmd

    def _stroke(self) -> tuple[float, float]:
        """The gripper p_des range: (open end, closed end), lower first."""
        a = self.cfg
        lo, hi = sorted((a.gripper_open_pdes, a.gripper_closed_pdes))
        return lo, hi

    def _measured_rad(self) -> list[float]:
        assert self._fb is not None
        return list(self._fb.pos[:6])

    def connect(self) -> None:
        """Opens the link and waits for 0x052; sends nothing. The bounds widen to the measured start pose.

        Fails when no 0x052 arrives within `connect_timeout_s` or the telemetry is frozen.
        """
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

        start = tuple(self._fb.pos[:6])
        self._bounds = tuple((min(lo, s), max(hi, s)) for (lo, hi), s in zip(P.URDF_LIMITS_RAD, start))
        self._link_silent = False
        self._connected = True

        g7 = self._fb.pos[6]
        logger.info("A1X gripper: group 7 at rest %.3f rad (%.1f deg) - measured only after the first 0x051 "
                    "(at rest after power-on group 7 reads an arbitrary value)", g7, math.degrees(g7))
        logger.info("A1X on %s: start pose %s deg", self.iface or "the given link",
                    ", ".join(f"J{i + 1}={math.degrees(start[i] - off):.1f}" for i, off in enumerate(self._off)))

        self._start_thread()
        if self.cfg.enable_on_connect:
            self.enable()

    def _await_feedback(self) -> None:
        """The first 0x052 frame within `connect_timeout_s`, then a moment of collecting for the frozen test."""
        where = self.iface or "the link"
        t0 = time.monotonic()
        while self._n_fb == 0:
            self._drain(self._clock())
            if self._n_fb:
                break
            if time.monotonic() - t0 > self._connect_timeout:
                raise RuntimeError(
                    f"no 0x052 frames on {where} - the CAN link is down (Linux: ./can_up.sh or "
                    + can.IP_LINK_UP.format(iface=where) + ") or the arm has no power")
            time.sleep(0.002)

        # `_identical` counts REPEATS, so n frames give at most n - 1: the "frozen" verdict
        # (> frozen_frames) needs frozen_frames + 2 frames. On hardware they arrive one by one (every 5 ms).
        frozen = self.cfg.frozen_frames
        settle = max(SETTLE_S, 2.0 * (frozen + 2) * FB_PERIOD_S)
        t1 = time.monotonic()
        while self._n_fb < frozen + 2 and time.monotonic() - t1 < settle:
            if not self._drain(self._clock()):
                time.sleep(0.002)
        if self._identical > frozen:
            raise RuntimeError("telemetry frozen - arm released (FF 2), switch the power off and on")

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
        except Exception:                      # pragma: no cover - disconnecting must not crash the caller
            logger.exception("Error while closing the CAN link")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def link_silent(self) -> bool:
        """The last `read()` found no fresh 0x052 (stale or frozen) and returned the last values."""
        return self._link_silent

    def _loop(self) -> None:
        period = 1.0 / self.cfg.stream_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.io_step(self._clock())
                self._bus_error = ""
            except Exception as exc:           # adapter pulled, link down: a fault, not a dead thread
                if not self._bus_error:
                    logger.error("A1X: CAN link error: %s", exc)
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
        return (now - self._fb_t) * 1000.0 > self.cfg.stale_ms

    def _frozen(self) -> bool:
        return self._identical > self.cfg.frozen_frames

    def io_step(self, now: float) -> None:
        """One exchange tick: receive every frame, then (when allowed) 0x050 and 0x051."""
        with self._lock:
            if self._bus is None:
                return
            self._drain(now)
            dt = (1.0 / self.cfg.stream_hz if self._last_io is None
                  else min(max(now - self._last_io, 0.0), 2.0 / self.cfg.stream_hz))
            self._last_io = now

            if self._fb is None or self._stale_now(now) or self._frozen():
                # Stale or frozen telemetry: send nothing and drop the targets. The caller stops and
                # moves again only from a confirmed measurement - the driver must not drive on its own
                # to a target from before the gap meanwhile (an arm power-cycled or blocked stands
                # somewhere else, and the A1X has no brakes). When frames return nothing goes out until
                # the next `send`: the arm holds the pose it stands in, the gripper its last p_des.
                self._p_target = None
                self._p_stream = None
                self._g_target = None
                self._g_stream = None
                self._g_frozen = False
                self._forget_grip_rest()
                self._g_live_n = None                  # the arm may have been off: group 7 is uncertain again
                self._deaf_since = {}
                self._deaf = None
                return

            if self._pinned:
                self._p_target = self._measured_rad()
                self._p_stream = list(self._p_target)

            if self._p_target is not None:
                if self._p_stream is None:
                    self._p_stream = self._measured_rad()     # start from the MEASUREMENT, not the target
                step = math.radians(self.cfg.max_step_deg_s) * dt
                self._p_stream = [s + max(-step, min(step, t - s)) for s, t in zip(self._p_stream, self._p_target)]
                self._send(P.CMD_ID, P.encode_arm(self._p_stream, self.cfg.kp, self.cfg.kd))
                self._watch_deaf(now)

            if self._g_target is not None:
                if self._g_stream is None:
                    self._g_stream = self._grip_reported()
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
                step = self.cfg.gripper_rate_pdes_s * dt
                self._g_stream += max(-step, min(step, self._g_target - self._g_stream))
                self._send(P.GRIP_ID, P.encode_gripper(self._g_stream, self.cfg.gripper_kp, self.cfg.kd))
                if self._g_live_n is None:
                    self._g_live_n = self._n_fb

    def _grip_contact(self) -> bool:
        """Gripper |effort| above the rest measured at the start of this close by more than `gripper_force`."""
        return self._g_rest is not None and abs(self._fb.eff[6]) > self._g_rest + self.cfg.gripper_force

    def _send(self, can_id: int, data: bytes, fd: bool = True) -> None:
        self._bus.send(can_id, data, fd=fd)
        self.tx_frames += 1

    def _watch_deaf(self, now: float) -> None:
        """Command far from the measurement while the joint stands for `deaf_s` = arm deaf to 0x050 or blocked."""
        a = self.cfg
        fb = self._fb
        worst: Fault | None = None
        for i in range(P.N_JOINTS):
            err = math.degrees(abs(self._p_stream[i] - fb.pos[i]))
            if err > a.deaf_err_deg and abs(math.degrees(fb.vel[i])) < STILL_DEG_S:
                t0 = self._deaf_since.setdefault(i, now)
                if now - t0 >= a.deaf_s and worst is None:
                    worst = Fault(i, f"{err:.0f} deg from the command for {now - t0:.1f} s and not moving - arm "
                                     f"deaf to 0x050 (enable the A1X motors, FF 1-5-6) or blocked")
            else:
                self._deaf_since.pop(i, None)
        self._deaf = worst

    def read(self) -> Reading:
        """The latest 0x052 frame. On a silent link: the last values, and `link_silent` turns True."""
        with self._lock:
            if self._fb is None:
                raise RuntimeError("The A1X arm is not connected")
            self._link_silent = self._stale_now(self._clock()) or self._frozen()
            return Reading(joints=tuple(self._fb.pos[i] - self._off[i] for i in range(P.N_JOINTS)),
                           gripper=self._grip_reported(), gripper_measured=self.gripper_measured())

    def send(self, joints: Mapping[int, float] | None = None, gripper: float | None = None) -> Accepted | None:
        """Arm targets {J index 0..5: rad} and a gripper target [p_des]. None = refused: the link is silent
        (`link_silent`), `enable()` runs, or no frame has arrived.

        Arm joints not given keep their target (before the first, the measurement). Gripper: a target frozen
        on contact stays, and `Accepted.gripper` is still the command.
        """
        joints = dict(joints or {})
        for i in joints:
            _check_joint(i)
        with self._lock:
            if not self._connected:
                raise RuntimeError("The A1X arm is not connected")
            if self._link_silent or self._pinned or self._fb is None:
                return None

            clamped: dict[int, float] = {}
            if joints:
                p = list(self._p_target if self._p_target is not None else self._measured_rad())
                for i, target in joints.items():
                    lo, hi = self._limits[i]
                    bound_lo, bound_hi = self._bounds[i]
                    off = self._off[i]
                    q = min(max(float(target), lo, bound_lo - off), hi, bound_hi - off)
                    p[i] = q + off
                    clamped[i] = q
                self._p_target = p

            took = None if gripper is None else self._send_gripper(float(gripper))
            return Accepted(joints=clamped, gripper=took)

    def _send_gripper(self, want: float) -> float | None:
        """The gripper target [p_des]; the command if it became the last one, None if held back."""
        lo, hi = self._stroke()
        want = min(max(want, lo), hi)
        reported = self._grip_reported()
        if self._g_target is None and abs(want - reported) <= GRIP_START_DELTA * (hi - lo):
            return None
        if self._g_target is None:
            measured = self.gripper_measured()
            logger.info("A1X gripper: first command p_des %.2f (reported %.2f%s)", want, reported,
                        "" if measured else ", not measured")
            # First command: the stream starts from the REPORTED opening, not from the target.
            stream = reported
            if self._g_assumed or (self._g_cmd is None and not measured):
                # The opening is only guessed: `assume_gripper` (e.g. "jaws closed" confirmed by an
                # operator) or just `gripper_start` after `connect` (the A1X keeps its last pose between
                # sessions, the jaws may stand open). Start from the more open of {guess, target}:
                # opening goes straight to the target (an empty closed gripper loses nothing, an open
                # one does not twitch), and a wrong guess does not snap the jaws shut in the first frame
                # (closed p_des at kp 25, no force threshold - that one works only while closing).
                # `_g_cmd` from a command that really went out survives a link gap, and the next one
                # ramps from it.
                stream = min(stream, want)
            self._g_target = want
            self._g_stream = stream
            self._g_assumed = False
        elif want <= self._g_target:
            # Opening (or the same target) always passes and lifts the freeze. With a strict `<` a
            # target frozen at the open end (`gripper_open_pdes`) could not be moved any more.
            self._g_target = want
            self._g_frozen = False
        elif not self._g_frozen:
            if self._grip_contact() and self._g_stream is not None:
                self._g_target = self._g_stream
                self._g_frozen = True
            else:
                self._g_target = want
        self._g_cmd = want
        return want

    def _fresh_fb(self) -> Any:
        """Latest 0x052 frame, or None when missing, stale or frozen."""
        if not self._connected or self._fb is None or self._stale_now(self._clock()) or self._frozen():
            return None
        return self._fb

    def efforts(self) -> list[float] | None:
        """J1..J6 effort, raw A1X scale from frame 0x052; None without fresh feedback."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return [float(fb.eff[i]) for i in range(P.N_JOINTS)]

    def velocities(self) -> list[float] | None:
        """J1..J6 [rad/s] from frame 0x052; None without fresh feedback."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return [float(fb.vel[i]) for i in range(P.N_JOINTS)]

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
        return float(self.cfg.gripper_force)

    def faults(self) -> list[Fault]:
        """What is wrong now: a CAN error, a silent link, frozen telemetry, a deaf joint. Never raises."""
        try:
            with self._lock:
                if not self._connected:
                    return []
                out: list[Fault] = []
                now = self._clock()
                if self._bus_error:
                    out.append(Fault(None, f"CAN bus: {self._bus_error}"))
                if self._stale_now(now):
                    out.append(Fault(None, f"no response from the arm for {(now - self._fb_t) * 1000.0:.0f} ms"))
                elif self._frozen():
                    out.append(Fault(None, f"telemetry frozen ({self._identical} identical frames) - arm released "
                                           f"(FF 2)? switch the power off and on"))
                elif self._deaf is not None:
                    out.append(self._deaf)
                state = str(getattr(self._bus, "state", "ok") or "ok")
                if state in ("bus-off", "error-passive"):
                    out.append(Fault(None, f"CAN bus: {state}"))
                return out
        except Exception as exc:               # pragma: no cover - called every control cycle, must not crash it
            logger.exception("Could not check the A1X state")
            return [Fault(None, f"could not check the A1X state: {exc}")]

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """The arm's own limits per joint [rad, minus the zero offsets]: the URDF limits widened to the pose at
        `connect` (J3 stands ~1.5 deg past its limit). `send` never goes past them."""
        return tuple((lo - off, hi - off) for (lo, hi), off in zip(self._bounds, self._off))

    def limits(self) -> tuple[tuple[float, float], ...]:
        """The caller's limits per joint [rad, minus the zero offsets]; unlimited until `set_limits`."""
        with self._lock:
            return tuple(self._limits)

    def set_limits(self, limits: Mapping[int, tuple[float, float]]) -> None:
        """The limits {J index: (lo, hi) [rad]} `send` clamps to from now on, within `bounds()`; joints not
        given keep theirs."""
        with self._lock:
            for i, (lo, hi) in limits.items():
                _check_joint(i)
                self._limits[i] = (float(lo), float(hi))

    def gripper_pdes(self, opening: float) -> float:
        """The gripper p_des at `opening` of the stroke: 0 = `gripper_closed_pdes`, 1 = `gripper_open_pdes`."""
        a = self.cfg
        return a.gripper_closed_pdes + opening * (a.gripper_open_pdes - a.gripper_closed_pdes)

    def gripper_opening(self, pdes: float) -> float:
        """The opening of the stroke at gripper p_des `pdes`: the inverse of `gripper_pdes`."""
        a = self.cfg
        return (pdes - a.gripper_closed_pdes) / (a.gripper_open_pdes - a.gripper_closed_pdes)

    def gripper_known(self) -> bool:
        """Whether the gripper position is known: measured, a command that REALLY went out in 0x051, or
        `assume_gripper`. A target within `GRIP_START_DELTA` of the reported position sends nothing and
        settles nothing."""
        with self._lock:
            return self.gripper_measured() or self._g_cmd is not None

    def assume_gripper(self, pdes: float) -> bool:
        """Takes the gripper to stand at `pdes` without a measurement (e.g. an operator confirms closed jaws).

        Only while the 0x051 stream has not started: after that the last command that really went out is
        the truth, and the assumption is refused (False, a warning in the log). After a successful one
        `gripper_known()` is True, so the next commands ramp from the assumption; the first stream starts
        from the more open of {assumption, target} (`send`, the same as from `gripper_start` alone). A
        measured gripper assumes nothing (False).
        """
        with self._lock:
            if self.gripper_measured():
                return False
            if self._g_target is not None:
                logger.warning("A1X gripper already driven (p_des %.2f) - assumption p_des %.2f refused",
                               self._g_target, pdes)
                return False
            lo, hi = self._stroke()
            self._g_cmd = min(max(float(pdes), lo), hi)
            self._g_assumed = True
            logger.info("A1X gripper: assumed p_des %.2f (not measured)", self._g_cmd)
            return True

    def gripper_drivable(self) -> bool:
        """Whether gripper commands reach the gripper. Over XCAN (macOS) yes, since it was measured on
        2026-09-27 that 0x051 gets through (jaws moved 85-116 degrees, `python -m galaxeo.xcan_usb
        --grip-test`); the earlier refusal rested on an unverified comment. `gripper_over_xcan: false`
        restores the refusal."""
        return not self._on_xcan() or self._gripper_xcan_allowed()

    def gripper_xcan_trial(self) -> bool:
        """XCAN with the gripper on (`gripper_over_xcan`, the default, or A1X_GRIPPER_XCAN=1): commands go out."""
        return self._on_xcan() and self._gripper_xcan_allowed()

    def _on_xcan(self) -> bool:
        return str(self.iface or "").lower().startswith("xcan")

    def _gripper_xcan_allowed(self) -> bool:
        env = os.environ.get(GRIPPER_XCAN_ENV, "").strip().lower() in ("1", "true", "yes", "tak")
        return bool(self.cfg.gripper_over_xcan) or env

    def gripper_travel_s(self) -> float:
        """Time of a full gripper stroke [s]: |open - closed| / `gripper_rate_pdes_s` + margin (1.2 s with
        the defaults: 2.03 / 2.0 + 0.2). How long to wait after a command for the jaws to really get there."""
        a = self.cfg
        return abs(a.gripper_open_pdes - a.gripper_closed_pdes) / a.gripper_rate_pdes_s + GRIP_TRAVEL_MARGIN_S

    def gripper_measured(self) -> bool:
        """Whether the gripper reading is measured: both `gripper_fb_*` points are set AND the gripper is
        driven in this connection (at least `GRIP_LIVE_FRAMES` 0x052 frames after the first 0x051, again
        after a link loss). At rest after power-on group 7 reads an arbitrary value (2026-09-27: -165 deg
        with the jaws at -85 deg), so until then the driver reports the last command, and before the
        first one `gripper_start` (or `assume_gripper`), which need not match the jaws."""
        with self._lock:
            a = self.cfg
            return (a.gripper_fb_closed is not None and a.gripper_fb_open is not None and self._fb is not None
                    and self._g_live_n is not None and self._n_fb - self._g_live_n >= GRIP_LIVE_FRAMES)

    def enable(self) -> None:
        """The FF 1 -> 5 -> 6 sequence, with p_des = the measurement before, between and after each code.

        Only on request (or `enable_on_connect`): a freshly powered arm listens right away, and FF 5
        briefly releases the motors. `send` is refused meanwhile; afterwards the arm holds the measured pose.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("The A1X arm is not connected")
            if self._fb is None or self._stale_now(self._clock()):
                raise RuntimeError("A1X: no fresh 0x052 frames - enable refused")
            if self._frozen():
                raise RuntimeError("A1X: telemetry frozen (FF 2) - switch the power off and on; FF 1-5-6 does "
                                   "not fix that")
            self._pinned = True
            self._p_target = self._measured_rad()
            self._p_stream = list(self._p_target)
        try:
            for code in P.ENABLE_SEQUENCE:
                self._wait(ENABLE_STEP_S)
                with self._lock:
                    if self._stale_now(self._clock()):
                        raise RuntimeError(f"A1X: the link went silent during enable (before FF {code})")
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

    def _wait(self, secs: float) -> None:
        """Time between the codes: the IO thread streams on its own; without it, the ticks run here."""
        if self._threaded:
            self._sleep(secs)
            return
        period = 1.0 / self.cfg.stream_hz
        for _ in range(max(1, round(secs / period))):
            self._sleep(period)
            self.io_step(self._clock())


def _check_joint(i: int) -> None:
    if not 0 <= i < P.N_JOINTS:
        raise ValueError(f"arm joint index {i!r}: 0..{P.N_JOINTS - 1} (the gripper is separate)")
