"""`Arm`: the installed A1X for apps. Plan a point, move there under the guard, stop.

    arm = Arm(BusTransport(open_bus("can0"), tx=True), armed=True)
    p = arm.plan([0.35, 0.0, 0.10])                 # Plan: accepted, or rejected with a reason
    if p:
        r = arm.move(p.joints, speed=20.0)           # MoveResult: reached | backoff | hold | rejected
    arm.stop()                                       # any thread: hold the measured pose, disarm

Plan refusals come in this order: "outside volume", "ik", "jump", "collision".
Move results: "reached", "backoff" (tripped, held, backed off), "hold" (stopped where
it is: stop, stale feedback, not arrived, a second trip), "rejected" (never started:
not armed, no/stale/frozen feedback, jump). While moving, `on_state("moving", "")`.
These are the words the digital twin's glasses driver reports.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .collision import CollisionChecker
from .kinematics import IKResult, Kinematics
from .model import GRIPPER, Obstacle, Tool
from .motion import MAX_JUMP_DEG, Motion, MoveResult
from .reach import IK_TOL, ReachBox


@dataclass
class Plan:
    state: str                                   # "accepted" | "rejected"
    reason: str = ""                             # "outside volume" | "ik" | "jump" | "collision"
    detail: str = ""
    joints: Optional[np.ndarray] = None          # arm_joint1..6 [rad]
    ik: Optional[IKResult] = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.state == "accepted"


class Arm:
    """The installed A1X. Without a transport it only plans (FK, IK, reach, collisions)."""

    def __init__(self, transport=None, *, tool: Tool = GRIPPER, armed: bool = False,
                 rate_hz: Optional[float] = None, on_tick: Optional[Callable] = None):
        self.tool = tool
        self.kin = Kinematics(tool)
        self.collision = CollisionChecker(tool)
        self.lo, self.hi = self.kin.lo.copy(), self.kin.hi.copy()
        self.transport = transport
        self.motion = Motion(transport, self.lo, self.hi, rate_hz=rate_hz, on_tick=on_tick) if transport else None
        self._armed = bool(armed)
        self._reach: Optional[ReachBox] = None
        self._kins = {tool: self.kin}
        self._checkers = {tool: self.collision}
        self._busy = threading.Lock()

    # ---- state -------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    def set_armed(self, on: bool) -> None:
        self._armed = bool(on)

    @property
    def reach(self) -> ReachBox:
        """Safe reach box for this arm's tool (cached as JSON next to the model)."""
        if self._reach is None:
            self._reach = ReachBox.load(self.tool, kin=self.kin, checker=self.collision)
        return self._reach

    def measured(self) -> Optional[np.ndarray]:
        """Fresh measured joints [rad], or None."""
        if self.motion is None:
            return None
        r, _ = self.motion.fresh()
        return None if r is None else r.q.copy()

    def fk(self, q: Sequence[float], tool: Optional[Tool] = None) -> np.ndarray:
        return self._kin(tool).fk(q)

    def _kin(self, tool: Optional[Tool]) -> Kinematics:
        tool = tool or self.tool
        if tool not in self._kins:
            self._kins[tool] = Kinematics(tool)
        return self._kins[tool]

    def _checker(self, tool: Optional[Tool]) -> CollisionChecker:
        tool = tool or self.tool
        if tool not in self._checkers:
            self._checkers[tool] = CollisionChecker(tool)
        return self._checkers[tool]

    # ---- plan --------------------------------------------------------------------
    def plan(self, target, tool: Optional[Tool] = None, obstacles: Iterable[Obstacle] = (),
             start: Optional[Sequence[float]] = None, *, ik_tol: float = IK_TOL,
             check_reach: bool = True) -> Plan:
        """Joints that put the tool on `target`: a point [m], or a 4x4 pose (rotation best effort).

        `start` is where the move would begin (default: the measured pose, else home); the jump and
        the path collision check run from it. The reach box is the one of this arm's tool.
        """
        T = np.asarray(target, float)
        if T.shape == (4, 4):
            point, rotation = T[:3, 3], T[:3, :3]
        else:
            point, rotation = T.reshape(3), None
        if check_reach and not self.reach.contains(point):
            return Plan("rejected", "outside volume", f"{np.round(point, 3).tolist()} not in "
                        f"{self.reach.lo.round(3).tolist()}..{self.reach.hi.round(3).tolist()}")

        kin = self._kin(tool)
        if start is None:
            start = self.measured()
        start = np.asarray(start if start is not None else kin.home, float)
        sol = kin.ik(point, rotation, seed=start, tol=ik_tol)
        if not sol.ok:
            return Plan("rejected", "ik", f"residual {sol.pos_err * 1000:.1f} mm", ik=sol)

        jump = float(np.max(np.abs(sol.q - start))) * 180.0 / math.pi
        if jump > MAX_JUMP_DEG:
            return Plan("rejected", "jump", f"{jump:.0f} deg > {MAX_JUMP_DEG:.0f}", joints=sol.q, ik=sol)

        why = self._checker(tool).path_clear(start, sol.q, obstacles)
        if why is not None:
            return Plan("rejected", "collision", why, joints=sol.q, ik=sol)
        return Plan("accepted", joints=sol.q, ik=sol)

    # ---- move --------------------------------------------------------------------
    def move(self, joints: Sequence[float], speed: float = 20.0, *, profile: str = "normal",
             on_state: Optional[Callable[[str, str], None]] = None, **guard) -> MoveResult:
        """Guarded joint move from the measured pose (see motion.py). Blocks until it ends."""
        if self.motion is None:
            raise RuntimeError("Arm has no transport: it can plan, not move")
        if not self._busy.acquire(blocking=False):
            return MoveResult("rejected", "busy")
        try:
            self.motion.begin()
            if not self._armed:
                return MoveResult("rejected", "not armed")
            return self.motion.move(joints, speed, profile=profile, on_state=on_state, **guard)
        finally:
            self._busy.release()

    def stop(self, reason: str = "stop") -> None:
        """Disarm; a move in progress ends holding the measured pose. Idle: nothing is sent."""
        self._armed = False
        if self.motion is not None:
            self.motion.cancel(reason)

    def interrupt(self, reason: str) -> None:
        """End a move in progress holding the measured pose, but stay armed (retarget, preempt)."""
        if self.motion is not None:
            self.motion.cancel(reason)
