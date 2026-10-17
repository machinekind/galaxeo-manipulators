"""The digital twin's `a1x` backend: an adapter from the twin's backend contract onto the A1X driver.

The driver is `galaxeo.a1x.A1X`: connect, the p_des stream, limits, link loss, enable and the gripper,
in wire units and without the twin. This module is a plugin of the twin's backend contract
(`BackendPlugin`): a plain class built by `create(ctx)`, registered under the `machinekind_twin.backends`
entry point. It never imports the twin.

What the adapter adds is the twin's view of the arm:

* units: arm joints in degrees, the gripper 0..100 (0 = closed). ``rad = radians(deg)`` (the driver adds
  `a1x.joint_offsets_deg`), gripper ``p_des = A1X.gripper_pdes(v / 100)``;
* joint names from the context, and faults pinned on a joint the way the twin reads them: ``"<joint>: <text>"``;
* the twin's limits (`set_joint_limits`), kept in degrees and handed to the driver in radians;
* the twin's own numbers back, exactly: `send_joints` returns the command it clamped in degrees and 0..100,
  `goal_positions` the latest of them, and an unmeasured gripper reads back the last command the driver
  took. Degrees -> radians and 0..100 -> p_des do not round-trip exactly in floating point;
* `gripper_state()`, the driver's `GripperState` in 0..100, and `foreign_frames()`, the command frames of
  another sender on the link. While another sender is seen, `faults()` reports it with no joint name, so the
  twin stops.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..a1x import A1X, CONNECT_TIMEOUT_S, A1XConfig, Reading, load_settings, open_bus
from ..protocol import N_JOINTS

__all__ = ["A1XArm", "A1XConfig", "BackendContext", "BackendInfo", "GripperState", "create", "load_settings",
           "open_bus"]

logger = logging.getLogger(__name__)


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
    """Read-only snapshot of the gripper as the driver drives and measures it (`A1XArm.gripper_state()`).

    Openings in app units: 0 = closed, 100 = open.
    """

    commanded: float | None        # last command the backend let through, or `assume_gripper`; None before the first
    target: float | None           # where the stream goes (differs from `commanded` once frozen)
    streamed: float | None         # opening really sent last; the best guess of the jaw without a measurement
    position: float | None         # measured opening; None while the driver does not measure it
    frozen: bool                   # contact while closing: the target stands where the stream was
    effort: float | None           # raw A1X scale; None without fresh feedback
    t: float                       # driver clock when the reading arrived (the queue drain)
    age_s: float | None            # driver clock now minus `t`; None before any reading
    rate: float                    # stream speed [units/s]
    force_threshold: float         # effort above rest at which a closing jaw stops
    measured: bool                 # whether `position` is a measurement


def create(ctx: BackendContext) -> A1XArm:
    """Entry point of the `a1x` backend plugin (`machinekind_twin.backends`)."""
    return A1XArm(ctx)


class A1XArm:
    """Galaxea A1X for the twin: joint names and app units on top of `galaxeo.a1x.A1X`.

    `bus`, `clock`, `threaded`, `sleep` and `connect_timeout_s` go to the driver. `threaded=False` (tests):
    no IO thread - `io_step(now)` runs the exchange ticks.
    """

    def __init__(self, ctx: BackendContext, *, bus: Any = None, clock: Callable[[], float] = time.monotonic,
                 threaded: bool = True, sleep: Callable[[float], None] = time.sleep,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S):
        cfg = load_settings(A1XConfig, ctx.settings, "a1x")
        self.gripper = ctx.gripper
        self.arm = tuple(n for n in ctx.joints if n != ctx.gripper)
        if len(self.arm) != N_JOINTS or ctx.gripper is None:
            raise ValueError("the a1x backend needs 6 arm joints and a gripper - use a station with robot: a1x")

        #: The driver this backend adapts.
        self.a1x = A1X(cfg, ctx.port or None, bus=bus, clock=clock, threaded=threaded, sleep=sleep,
                       connect_timeout_s=connect_timeout_s)
        #: Limits [app units] `send_joints` clamps to; the twin moves them through `set_joint_limits`.
        self.limits: dict[str, tuple[float, float]] = dict(ctx.limits)
        self._lock = threading.RLock()
        self._goal: dict[str, float] = {}            # latest commands [app units]
        self._grip_cmd: float | None = None          # the last gripper command [0..100] the driver took, or assumed
        self._gripper_state_failed = False
        self._limits_to_driver()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(name="Galaxea A1X (CAN)", description=f"link {self.a1x.iface or 'CAN'}", simulated=False)

    def connect(self) -> None:
        with self._lock:
            if not self.a1x.is_connected:
                self._goal = {}
                self._grip_cmd = None
                self._gripper_state_failed = False
        self.a1x.connect()

    def disconnect(self) -> None:
        """Ends the stream and closes the link. Never releases the motors (FF 2): the A1X has no brakes."""
        self.a1x.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.a1x.is_connected

    @property
    def link_silent(self) -> bool:
        """The last `read_joints` got no fresh 0x052 and returned the last values (the twin's contract)."""
        return self.a1x.link_silent

    #: The contract's older name for `link_silent`.
    _link_silent = link_silent

    @property
    def _stale(self) -> list[str]:
        """Joints without a fresh reply in the last `read_joints`: all of them on a silent link (the twin reads it)."""
        return list(self.arm) if self.a1x.link_silent else []

    def io_step(self, now: float) -> None:
        """One exchange tick of the driver (tests, `threaded=False`)."""
        self.a1x.io_step(now)

    def read_joints(self) -> dict[str, float]:
        """Arm degrees (minus the zero offset) + gripper 0..100. On a silent link: the last values."""
        with self._lock:
            r = self.a1x.read()
            out = {name: math.degrees(q) for name, q in zip(self.arm, r.joints)}
            out[self.gripper] = self._grip_app(r)
            return out

    def _grip_app(self, r: Reading) -> float:
        """Gripper 0..100: the measurement, or (open loop, like the driver) the last command / `gripper_start`."""
        if r.gripper_measured:
            return 100.0 * self.a1x.gripper_opening(r.gripper)
        return self.a1x.cfg.gripper_start if self._grip_cmd is None else self._grip_cmd

    def _pdes_app(self, p: float | None) -> float | None:
        """Gripper p_des -> 0..100, the inverse of `A1X.gripper_pdes`; None stays None."""
        return None if p is None else 100.0 * self.a1x.gripper_opening(p)

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Arm and gripper target. Returns the clamped COMMAND in app units.

        Gripper: returns the command, not the p_des frozen on contact - the twin's supervisor must
        keep seeing a command tighter than the measurement, or it does not recognise a grip
        (`_grip_squeezing`, "in hand" in cube tracking). `{}` on a silent link and during the
        enable sequence.
        """
        with self._lock:
            sent: dict[str, float] = {}
            joints: dict[int, float] = {}
            bounds = self.a1x.bounds()
            for i, name in enumerate(self.arm):
                if name not in targets:
                    continue
                lo, hi = self.limits[name]
                bound_lo, bound_hi = (math.degrees(b) for b in bounds[i])
                v = min(max(float(targets[name]), lo, bound_lo), hi, bound_hi)
                joints[i] = math.radians(v)
                sent[name] = v

            gripper = None
            if self.gripper in targets:
                lo, hi = self.limits[self.gripper]
                v = min(max(float(targets[self.gripper]), lo, 0.0), hi, 100.0)
                gripper = self.a1x.gripper_pdes(v / 100.0)
                sent[self.gripper] = v

            took = self.a1x.send(joints, gripper)
            if took is None:
                return {}
            if took.gripper is not None:
                self._grip_cmd = sent[self.gripper]
            self._goal.update(sent)
            return sent

    def efforts(self) -> dict[str, float] | None:
        """Raw A1X effort scale from frame 0x052; arm joints only, no gripper."""
        eff = self.a1x.efforts()
        return None if eff is None else dict(zip(self.arm, eff))

    def gripper_effort(self) -> float | None:
        """|effort| of the gripper group (frame 0x052, eff[6], raw A1X scale), or None without fresh feedback."""
        return self.a1x.gripper_effort()

    @property
    def gripper_force(self) -> float:
        """Gripper effort above rest that means contact (config `gripper_force`)."""
        return self.a1x.gripper_force

    def velocities(self) -> dict[str, float] | None:
        """deg/s from frame 0x052; arm joints only, no gripper."""
        vel = self.a1x.velocities()
        return None if vel is None else {name: math.degrees(v) for name, v in zip(self.arm, vel)}

    def gripper_state(self) -> GripperState | None:
        """The driver's `A1X.gripper_state()` with openings in 0..100; `commanded` is the twin's own number.

        Changes nothing and sends nothing. None without a connection and on any error (logged
        once): a control loop asks every tick and must not crash on it.
        """
        try:
            with self._lock:
                gs = self.a1x.gripper_state()
                if gs is None:
                    return None

                a = self.a1x.cfg
                return GripperState(
                    commanded=self._grip_cmd,
                    target=self._pdes_app(gs.target),
                    streamed=self._pdes_app(gs.streamed),
                    position=self._pdes_app(gs.position),
                    frozen=gs.frozen,
                    effort=gs.effort,
                    t=gs.t,
                    age_s=gs.age_s,
                    rate=gs.rate * 100.0 / abs(a.gripper_open_pdes - a.gripper_closed_pdes),
                    force_threshold=gs.force_threshold,
                    measured=gs.measured,
                )
        except Exception:
            if not self._gripper_state_failed:
                self._gripper_state_failed = True
                logger.exception("A1X: could not read the gripper state")
            return None

    def foreign_frames(self) -> int:
        """Command frames (0x050, 0x051, FF) of another sender received since connecting (`A1X.foreign_frames`)."""
        return self.a1x.foreign_frames()

    def goal_positions(self) -> dict[str, float]:
        """Latest targets [app units]; `{}` before the first command (p_des cannot be read back)."""
        with self._lock:
            return dict(self._goal)

    def faults(self) -> list[str]:
        """The driver's faults; one on a joint starts with the joint's name, which the twin pins it on."""
        return [f.text if f.joint is None else f"{self.arm[f.joint]}: {f.text}" for f in self.a1x.faults()]

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """URDF limits [deg] widened to the start pose (J3 stands ~1.5 deg past its limit); no gripper."""
        return {name: (math.degrees(lo), math.degrees(hi)) for name, (lo, hi) in zip(self.arm, self.a1x.bounds())}

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        with self._lock:
            for name, (lo, hi) in limits.items():
                if name in self.limits:
                    self.limits[name] = (float(lo), float(hi))
            self._limits_to_driver()

    def _limits_to_driver(self) -> None:
        """The arm joints' limits to the driver, in radians: its own clamp, under the one in `send_joints`."""
        self.a1x.set_limits({i: (math.radians(self.limits[name][0]), math.radians(self.limits[name][1]))
                             for i, name in enumerate(self.arm) if name in self.limits})

    def calibration_warnings(self) -> list[str]:
        a = self.a1x.cfg
        if a.gripper_fb_closed is None or a.gripper_fb_open is None:
            return ["A1X gripper: feedback not measured - the panel shows the command, not a measurement; "
                    "measure a1x.gripper_fb_closed/open"]
        return []

    def gripper_known(self) -> bool:
        """Whether the gripper position is known: measured, a command that went out, or `assume_gripper`."""
        return self.a1x.gripper_known()

    def gripper_measured(self) -> bool:
        """Whether the gripper reading is measured (`A1X.gripper_measured`); else it is the last command."""
        return self.a1x.gripper_measured()

    def assume_gripper(self, v: float) -> bool:
        """The operator confirms the gripper opening without a measurement ("Set home = here": jaws closed = 0).

        Refused (False) on a measured gripper, or once the 0x051 stream has started (`A1X.assume_gripper`).
        """
        with self._lock:
            v = min(max(float(v), 0.0), 100.0)
            if not self.a1x.assume_gripper(self.a1x.gripper_pdes(v / 100.0)):
                return False
            self._grip_cmd = v
            return True

    def gripper_drivable(self) -> bool:
        """Whether the twin should send gripper commands (`A1X.gripper_drivable`)."""
        return self.a1x.gripper_drivable()

    def gripper_xcan_trial(self) -> bool:
        """XCAN with the gripper on: commands go out, and the panel's status line says so."""
        return self.a1x.gripper_xcan_trial()

    def gripper_travel_s(self) -> float:
        """Time of a full gripper stroke [s]: how long to wait after a command for the jaws to get there."""
        return self.a1x.gripper_travel_s()

    def enable(self) -> str:
        """The FF 1 -> 5 -> 6 sequence (`A1X.enable`), for the panel's button. Commands are refused meanwhile."""
        self.a1x.enable()
        return "sent FF 1 -> 5 -> 6; the arm holds the measured pose"
