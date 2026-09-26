"""Impact guard: spot a blocked or overloaded joint from loop samples, and pick a pose to back off to.

    guard = ImpactGuard(JOINTS, effort_abs=20.0)
    hit = guard.feed(t, sent, measured, fresh, efforts=..., velocities=...)   # all per joint name, deg
    if hit:                                            # hit.kind is "lag" or "effort"; hit.backoff_pose

Pure number logic, no clock and no I/O. Two triggers:

* **Lag** (the main one on hardware): the command keeps moving but the joint does not,
  |command - measured| > `lag_deg` for `lag_s` while the joint is nearly still
  (|velocity| < `vel_still` deg/s). A normal ramp has the command just ahead of a moving
  joint, so it does not trigger.
* **Effort**: |effort| above a threshold (`effort_abs`, per joint or one for all) for
  `effort_n` consecutive samples.

Back-off pose: the newest good (unsuspicious) sample at least `backoff_s` older than the
impact whose farthest joint is at least `min_backoff_deg` from the measurement; else the
oldest good sample; else the measurement. Samples with nothing sent and stale ones are
ignored: they say nothing new about the arm.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass
class Impact:
    t: float
    joint: str
    kind: str  # "lag" or "effort"
    value: float  # lag in deg, or effort in backend units
    # Guarded joints only; None when none of them were measured.
    backoff_pose: dict[str, float] | None


class ImpactGuard:
    def __init__(
        self,
        joints: Sequence[str],
        *,
        lag_deg: float = 6.0,
        lag_s: float = 0.15,
        vel_still: float = 15.0,
        effort_abs: Mapping[str, float] | float | None = None,
        effort_n: int = 3,
        backoff_s: float = 0.4,
        keep_s: float = 3.0,
        min_backoff_deg: float = 3.0,
    ):
        self.joints = list(joints)
        self.lag_deg = float(lag_deg)
        self.lag_s = float(lag_s)
        self.vel_still = float(vel_still)
        if effort_abs is None:
            self.effort_abs: dict[str, float] = {}
        elif isinstance(effort_abs, Mapping):
            self.effort_abs = {j: float(v) for j, v in effort_abs.items() if j in self.joints}
        else:
            self.effort_abs = {j: float(effort_abs) for j in self.joints}
        self.effort_n = max(1, int(effort_n))
        self.backoff_s = float(backoff_s)
        self.keep_s = float(keep_s)
        self.min_backoff_deg = float(min_backoff_deg)
        self.reset()

    def reset(self) -> None:
        self._good: deque[tuple[float, dict[str, float]]] = deque()
        self._lag_since: dict[str, float] = {}
        self._effort_count: dict[str, int] = {}
        self._prev: tuple[float, dict[str, float]] | None = None

    def _velocity(self, t: float, measured: dict[str, float],
                  velocities: Mapping[str, float] | None) -> dict[str, float]:
        """Velocities in deg/s: from the backend, missing ones derived from the previous sample; absent = unknown."""
        vel: dict[str, float] = {}
        if velocities:
            vel.update({j: float(velocities[j]) for j in self.joints if j in velocities})
        prev = self._prev
        if prev is not None and t > prev[0]:
            dt = t - prev[0]
            for j in self.joints:
                if j not in vel and j in measured and j in prev[1]:
                    vel[j] = (measured[j] - prev[1][j]) / dt
        return vel

    def feed(
        self,
        t: float,
        sent: Mapping[str, float],
        measured: Mapping[str, float],
        fresh: bool,
        efforts: Mapping[str, float] | None = None,
        velocities: Mapping[str, float] | None = None,
    ) -> Impact | None:
        """Feed one loop sample; returns an `Impact` on the tick it triggers, else None.

        After a trigger the counters restart (the next one needs a fresh `lag_s` / `effort_n`); the good-sample
        buffer is kept until `reset`.
        """
        if not sent or not fresh:
            return None
        m = {j: float(measured[j]) for j in self.joints if j in measured}
        if not m:
            return None
        vel = self._velocity(t, m, velocities)
        self._prev = (t, m)

        hit: tuple[str, str, float] | None = None
        suspicious = False
        for j in self.joints:
            if j not in sent or j not in m:
                self._lag_since.pop(j, None)
                continue
            err = abs(float(sent[j]) - m[j])
            v = vel.get(j)
            if err > self.lag_deg and v is not None and abs(v) < self.vel_still:
                suspicious = True
                t0 = self._lag_since.setdefault(j, t)
                if t - t0 >= self.lag_s and hit is None:
                    hit = (j, "lag", err)
            else:
                self._lag_since.pop(j, None)
        if efforts is not None:
            for j, limit in self.effort_abs.items():
                if j not in efforts:
                    continue
                e = float(efforts[j])
                if abs(e) > limit:
                    suspicious = True
                    n = self._effort_count[j] = self._effort_count.get(j, 0) + 1
                    if n >= self.effort_n and hit is None:
                        hit = (j, "effort", e)
                else:
                    self._effort_count.pop(j, None)

        if hit is not None:
            joint, kind, value = hit
            impact = Impact(t=t, joint=joint, kind=kind, value=value, backoff_pose=self._backoff(t, m))
            self._lag_since.clear()
            self._effort_count.clear()
            return impact
        if not suspicious:
            self._good.append((t, m))
        while self._good and self._good[0][0] < t - self.keep_s:
            self._good.popleft()
        return None

    def _backoff(self, t: float, measured: dict[str, float]) -> dict[str, float] | None:
        for ts, pose in reversed(self._good):
            if ts > t - self.backoff_s:
                continue
            far = max((abs(pose[j] - measured[j]) for j in pose if j in measured), default=0.0)
            if far >= self.min_backoff_deg:
                return dict(pose)
        if self._good:
            return dict(self._good[0][1])
        return dict(measured) if measured else None

    def good_samples(self) -> list[tuple[float, dict[str, float]]]:
        return [(ts, dict(p)) for ts, p in self._good]
