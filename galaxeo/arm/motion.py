"""The joint-space motion primitive: one guarded move from the measured pose to a joint goal.

    m = Motion(transport, lo, hi)
    result = m.move(goal_rad, speed_deg_s=20.0)      # reached | backoff | hold(reason) | rejected(reason)
    m.cancel("stop")                                 # from any thread: the arm holds its measured pose

What a move does, and why:

  * Starts from the MEASURED pose, never from an old command: a setpoint away from
    the arm makes it travel there as fast as it can (docs/SAFETY.md).
  * Refuses to start on missing, stale (> 150 ms) or frozen feedback, and on a jump:
    no joint may travel more than 45 deg in one move.
  * The goal is clamped to the model limits widened to include the start pose
    (J3 reads ~1.5 deg past its limit at rest).
  * p_des follows a smoothstep ramp at `speed_deg_s` average (peak 1.5x), streamed at
    the transport rate, and is additionally slew-capped at 1.5x speed + 5 deg/s.
  * The impact guard (guard.py) watches every tick: lag (command moving, joint still)
    and effort. A joint already above half its effort threshold at the start gets the
    lag rule only ("loaded joint"): it would trip on every move otherwise.
  * Arrival: every joint within 2 deg and slower than 10 deg/s, with a tighter lag
    guard (2.5 deg / 0.3 s) so a joint stalled at the goal trips quickly. A loaded servo
    settles 1-2 deg short, so the command is nudged by the error, at most 2.5 deg from
    the goal; not when descending (keep pressing on the object) and not with a loaded joint.
  * Stale feedback mid-move stops the stream at once: the arm re-latches where it is.
  * On a trip: hold the measured pose (the arm has no compliance and must never be left
    pressing into something), then back off at 8 deg/s to the newest good pose recorded
    >= 0.4 s before the trip and >= 3 deg away. The backoff has its own guard, armed after
    a 0.3 s grace (a joint just pulled off an obstacle still reads high effort). A trip
    during the backoff only holds: backoffs never chain.

Nothing here sends a release (FF 2): the arm has no brakes.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .. import protocol as P
from .guard import Impact, ImpactGuard
from .model import JOINTS

FB_STALE_S = 0.15
MAX_JUMP_DEG = 45.0
MAX_SPEED_DEG_S = 45.0
MIN_DURATION_S = 0.5

LAG_DEG = 6.0
LAG_DEG_DESCEND = 3.0
LAG_S = 0.15
ARRIVE_LAG_DEG = 2.5
ARRIVE_LAG_S = 0.3
EFFORT_N = 3
EFFORT_REST_FRAC = 0.5

ARRIVE_DEG = 2.0
STILL_DEG_S = 10.0
ARRIVE_TIMEOUT_S = 3.0
TRIM_STEPS = 4
TRIM_DEG = 0.2
TRIM_MAX_DEG = 2.5
TRIM_WAIT_S = 0.4

BACKOFF_SPEED_DEG_S = 8.0
BACKOFF_CAP_DEG_S = 15.0
BACKOFF_GRACE_S = 0.3
BACKOFF_S = 0.4
MIN_BACKOFF_DEG = 3.0

ENABLE_STEP_S = 0.3


@dataclass
class MoveResult:
    state: str                          # "reached" | "backoff" | "hold" | "rejected"
    reason: str = ""
    detail: str = ""
    impact: Optional[Impact] = None
    q: Optional[np.ndarray] = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.state == "reached"


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def _named(v: np.ndarray, scale: float = 1.0) -> Dict[str, float]:
    return {j: float(x) * scale for j, x in zip(JOINTS, v)}


_DEG = 180.0 / math.pi


class _End(Exception):
    """Ends a ramp: why, and whether the measured pose should be written as the hold."""

    def __init__(self, kind: str, reason: str, detail: str = "", impact: Optional[Impact] = None):
        super().__init__(reason)
        self.kind, self.reason, self.detail, self.impact = kind, reason, detail, impact


class Motion:
    """Guarded joint moves on one transport. One move at a time; `cancel` is thread-safe."""

    def __init__(self, transport, lo: Sequence[float], hi: Sequence[float], *, rate_hz: Optional[float] = None,
                 on_tick: Optional[Callable] = None):
        self.t = transport
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.rate = float(rate_hz or transport.rate_hz)
        self.period = 1.0 / self.rate
        self.on_tick = on_tick
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._cause = ""
        self.phase = "idle"

    # ---- cancel ----------------------------------------------------------------
    def begin(self) -> None:
        with self._lock:
            self._cancel.clear()
            self._cause = ""

    def cancel(self, cause: str = "stop") -> None:
        with self._lock:
            if not self._cancel.is_set():
                self._cause = cause
                self._cancel.set()

    # ---- feedback --------------------------------------------------------------
    def fresh(self):
        """(reading, "") or (None, reason)."""
        r = self.t.read()
        if r is None:
            return None, "no feedback"
        age = self.t.now() - r.t
        if age > FB_STALE_S:
            return None, f"stale feedback ({age * 1e3:.0f} ms)"
        if r.frozen:
            return None, "frozen telemetry"
        return r, ""

    def _read(self):
        r, why = self.fresh()
        if r is None:
            raise _End("stale", why.split(" (")[0], why)
        return r

    def _hold(self) -> Optional[np.ndarray]:
        """Write the measured pose once; the A1X keeps the last p_des, so the arm stops where it is."""
        r, _ = self.fresh()
        if r is None:
            return None
        self.t.write(r.q)
        return r.q

    # ---- one move --------------------------------------------------------------
    def move(self, goal: Sequence[float], speed_deg_s: float = 20.0, *, profile: str = "normal",
             lag_deg: Optional[float] = None, arrive_lag_s: Optional[float] = None,
             on_state: Optional[Callable[[str, str], None]] = None) -> MoveResult:
        """Move to `goal` [rad]. Call `begin()` first when a `cancel` may already be pending."""
        descend = profile == "descend"
        r, why = self.fresh()
        if r is None:
            return MoveResult("rejected", why.split(" (")[0], why)
        start = r.q.copy()
        lo, hi = np.minimum(self.lo, start), np.maximum(self.hi, start)
        goal = np.clip(np.asarray(goal, float).reshape(6), lo, hi)
        jump = float(np.max(np.abs(goal - start))) * _DEG
        if jump > MAX_JUMP_DEG:
            return MoveResult("rejected", "jump", f"{jump:.0f} deg > {MAX_JUMP_DEG:.0f}", q=start)
        speed = min(max(float(speed_deg_s), 0.5), MAX_SPEED_DEG_S)

        limits = self.t.effort_limits(descend)
        loaded = self._loaded(r.effort, limits)
        effort_abs = {} if limits is None else {
            j: float(limits[k]) for k, j in enumerate(JOINTS) if j not in loaded}
        guard = ImpactGuard(JOINTS, lag_deg=lag_deg or (LAG_DEG_DESCEND if descend else LAG_DEG), lag_s=LAG_S,
                            effort_abs=effort_abs, effort_n=EFFORT_N, backoff_s=BACKOFF_S,
                            min_backoff_deg=MIN_BACKOFF_DEG)
        if on_state:
            on_state("moving", "")
        self.phase = "moving"
        try:
            try:
                self._ramp(start, goal, speed, guard, cap=1.5 * speed + 5.0, arrive_lag_s=arrive_lag_s,
                           trim=not descend and not loaded, limits=limits)
            except _End as end:
                if end.kind == "impact":
                    return self._backoff(end.impact, limits)
                return self._ended(end)
            return MoveResult("reached", q=self.t.read().q.copy())
        finally:
            self.phase = "idle"

    def _ended(self, end: _End, prefix: str = "") -> MoveResult:
        if end.kind == "stale":
            return MoveResult("hold", prefix + end.reason, end.detail)
        q = self._hold()
        return MoveResult("hold", prefix + end.reason, end.detail, q=q)

    def _loaded(self, effort, limits) -> list:
        if limits is None:
            return []
        return [j for k, j in enumerate(JOINTS) if abs(float(effort[k])) > EFFORT_REST_FRAC * float(limits[k])]

    def _backoff(self, hit: Impact, limits) -> MoveResult:
        why = f"impact {hit.kind} {hit.joint}"
        held = self._hold()
        if held is None:
            return MoveResult("hold", why, "feedback lost at the trip", impact=hit)
        if not hit.backoff_pose:
            return MoveResult("hold", why, impact=hit, q=held)
        pose = np.array([math.radians(hit.backoff_pose.get(j, held[k] * _DEG)) for k, j in enumerate(JOINTS)])
        self.phase = "backoff"
        effort_abs = {} if limits is None else _named(np.asarray(self.t.effort_limits(False)))
        guard = ImpactGuard(JOINTS, lag_deg=LAG_DEG, lag_s=LAG_S, effort_abs=effort_abs, effort_n=EFFORT_N)
        try:
            self._wait(self.t.now() + self.period)
            self._ramp(held, pose, BACKOFF_SPEED_DEG_S, guard, cap=BACKOFF_CAP_DEG_S,
                       trim=False, grace_until=self.t.now() + BACKOFF_GRACE_S)
        except _End as end:
            res = self._ended(end, prefix="" if end.kind == "cancel" else why + "; ")
            if end.kind == "impact":
                res.reason, res.detail = why, f"second trip during backoff: {end.reason}"
            res.impact = hit
            return res
        return MoveResult("backoff", why, impact=hit, q=self.t.read().q.copy())

    # ---- the ramp --------------------------------------------------------------
    def _wait(self, until: float) -> None:
        delay = until - self.t.now()
        if delay > 0:
            self.t.sleep(delay)

    def _check(self) -> None:
        if self._cancel.is_set():
            with self._lock:
                cause = self._cause or "stop"
            raise _End("cancel", cause)

    def _step(self, cmd: np.ndarray, guard: ImpactGuard, grace_until: float):
        """One tick: read, write `cmd`, feed the guard. Returns the reading."""
        self._check()
        r = self._read()
        self.t.write(cmd)
        now = self.t.now()
        if self.on_tick is not None:
            self.on_tick(now, cmd, r, self.phase)
        if now >= grace_until:
            hit = guard.feed(now, _named(cmd, _DEG), _named(r.q, _DEG), True,
                             efforts=_named(r.effort), velocities=_named(r.dq, _DEG))
            if hit is not None:
                raise _End("impact", f"impact {hit.kind} {hit.joint}", impact=hit)
        return r

    def _ramp(self, start, goal, speed, guard, *, cap, trim, arrive_lag_s=None, grace_until=-math.inf,
              limits=None) -> None:
        travel = float(np.max(np.abs(goal - start))) * _DEG
        duration = max(MIN_DURATION_S, travel / speed)
        cap_rad = math.radians(cap)
        cmd = start.copy()
        t0 = last = self.t.now()
        nxt = t0
        while True:
            now = self.t.now()
            s = smoothstep((now - t0) / duration)
            want = start + s * (goal - start)
            dt = min(max(now - last, 0.0), 2 * self.period)
            last = now
            step = cap_rad * max(dt, 1e-9)
            cmd = cmd + np.clip(want - cmd, -step, step)
            self._step(cmd, guard, grace_until)
            if s >= 1.0 and np.max(np.abs(goal - cmd)) < 1e-9:
                break
            nxt = self._next(nxt)

        guard.lag_deg = ARRIVE_LAG_DEG
        guard.lag_s = ARRIVE_LAG_S if arrive_lag_s is None else float(arrive_lag_s)
        deadline = self.t.now() + ARRIVE_TIMEOUT_S
        while True:
            nxt = self._next(nxt)
            r = self._step(cmd, guard, grace_until)
            err = np.abs(r.q - goal) * _DEG
            if np.all(err <= ARRIVE_DEG) and np.all(np.abs(r.dq) * _DEG < STILL_DEG_S):
                break
            if self.t.now() > deadline:
                far = ", ".join(f"J{k + 1} {(r.q[k] - goal[k]) * _DEG:+.1f} deg" for k in np.flatnonzero(err > ARRIVE_DEG))
                raise _End("hold", "not arrived", far)

        if not trim or self._loaded(r.effort, limits):
            return
        lim = math.radians(TRIM_MAX_DEG)
        for _ in range(TRIM_STEPS):
            error = goal - r.q
            if np.max(np.abs(error)) * _DEG < TRIM_DEG:
                break
            cmd = np.clip(cmd + error, goal - lim, goal + lim)
            until = self.t.now() + TRIM_WAIT_S
            while self.t.now() < until:
                nxt = self._next(nxt)
                r = self._step(cmd, guard, grace_until)

    def _next(self, nxt: float) -> float:
        nxt += self.period
        now = self.t.now()
        if nxt < now - 0.05:            # fell behind: resync, do not burst
            nxt = now
        self._wait(nxt)
        return nxt


def enable(transport, step_s: float = ENABLE_STEP_S) -> None:
    """Function frames 1 -> 5 -> 6 with p_des = the live measured pose streamed before, between and after.

    FF 5 briefly disengages the motors; enabling with nothing streamed once swung an arm
    76 deg at saturated torque (docs/SAFETY.md). A power-cycled arm obeys 0x050 without
    this: use it only when the arm reports but ignores p_des.
    """
    period = 1.0 / transport.rate_hz

    def stream(secs: float) -> None:
        until = transport.now() + secs
        while transport.now() < until:
            r = transport.read()
            if r is None or transport.now() - r.t > FB_STALE_S or r.frozen:
                raise RuntimeError("enable aborted: no fresh feedback")
            transport.write(r.q)
            transport.sleep(period)

    for code in P.ENABLE_SEQUENCE:
        stream(step_s)
        transport.send_ff(code)
    stream(step_s)
