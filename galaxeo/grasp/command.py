"""Command bookkeeping: the training environment's `step`, without physics and without IK.

    cmd = GraspCommand(hover.p, hover.yaw, q_hover_rad, floor_z=floor)
    prop = cmd.propose(action)                 # boxed, floored, ceilinged TCP target and yaw
    q, ok = ...                                # one IK step, the caller's business
    cmd.commit(prop, q if ok else None)        # a refused step keeps p, yaw and q
    tail = cmd.obs_tail()                      # entries 10..18 of the 19-vector

The command integrates from itself, never from the measured pose, exactly as training does.
The yaw sum runs in float32 like training's (numpy 2 promotion rules): replaying the recorded
bookkeeping in float64 moves 995 of 1000 yaws by about 1e-7 rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contract import ACT_DIM, BOX_XY, CEILING, CLOSE_BELOW, HOVER, STEP_XYZ, STEP_YAW, grasp_R, wrap


@dataclass
class CommandState:
    p0: np.ndarray                 # (3,) float64, hover = box centre
    p_cmd: np.ndarray              # (3,) float64
    yaw_cmd: float                 # python float at the hover, np.float32 after the first accepted step
    q_cmd: np.ndarray              # (6,) float64 rad, the last accepted joints
    believed_pos: np.ndarray       # (3,) float64 == p0 - [0, 0, HOVER]
    believed_yaw: float            # == the hover's yaw
    g_cmd: float = 1.0
    closing: bool = False
    attempts: int = 0
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(ACT_DIM, np.float32))
    ticks: int = 0


@dataclass(frozen=True)
class Proposal:
    action: np.ndarray             # (5,) float32, clipped to [-1, 1]
    p: np.ndarray                  # (3,) float64, boxed, floored, ceilinged TCP target
    yaw: float                     # np.float32 scalar
    R: np.ndarray                  # (3, 3) grasp_R(yaw)
    g: float                       # python float in [0, 1]
    box_hit: bool
    floor_hit: bool
    ceiling_hit: bool


def _vector(value: np.ndarray, n: int, what: str) -> np.ndarray:
    v = np.array(value, dtype=np.float64)
    if v.shape != (n,):
        raise ValueError(f"{what}: expected shape ({n},), got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{what}: non-finite value {v.tolist()}")
    return v


class GraspCommand:
    """The commanded TCP, yaw, joints and gripper of one episode."""

    def __init__(self, p_hover: np.ndarray, yaw: float, q_hover: np.ndarray, *, floor_z: float,
                 ceiling: float = CEILING, box_xy: float = BOX_XY,
                 step_xyz: float = STEP_XYZ, step_yaw: float = STEP_YAW) -> None:
        p0 = _vector(p_hover, 3, "p_hover")
        q0 = _vector(q_hover, 6, "q_hover")
        if not math.isfinite(yaw):
            raise ValueError(f"yaw: non-finite value {yaw}")
        if not math.isfinite(floor_z) or floor_z > p0[2] + ceiling:
            raise ValueError(f"floor_z {floor_z} above the ceiling {p0[2] + ceiling}")

        # Python floats, so a float32 action keeps the products float32 as in training.
        self.ceiling = float(ceiling)
        self.box_xy = float(box_xy)
        self.step_xyz = float(step_xyz)
        self.step_yaw = float(step_yaw)
        self.floor_z = float(floor_z)

        self.state = CommandState(p0=p0.copy(), p_cmd=p0.copy(), yaw_cmd=float(yaw), q_cmd=q0,
                                  believed_pos=p0 - np.array([0.0, 0.0, HOVER]), believed_yaw=float(yaw))

    def propose(self, action: np.ndarray) -> Proposal:
        """The next target from the command; raises ValueError on a non-finite action or a wrong shape."""
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (ACT_DIM,):
            raise ValueError(f"action: expected shape ({ACT_DIM},), got {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"action: non-finite value {raw.tolist()}")

        s = self.state
        box = self.box_xy
        ceiling_z = s.p0[2] + self.ceiling
        # The caller raises the floor between ticks; a floor above the ceiling would let np.clip
        # return the ceiling and command the tool under the floor without a word.
        if not math.isfinite(self.floor_z) or self.floor_z > ceiling_z:
            raise ValueError(f"floor_z {self.floor_z} above the ceiling {ceiling_z}")

        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        free = s.p_cmd + a[:3] * self.step_xyz                  # float32 product, float64 sum
        p = free.copy()
        p[:2] = np.clip(p[:2], s.p0[:2] - box, s.p0[:2] + box)
        p[2] = np.clip(p[2], self.floor_z, ceiling_z)
        yaw = wrap(s.yaw_cmd + a[3] * self.step_yaw)            # float32 arithmetic, as in training
        g = float(np.clip(0.5 * (a[4] + 1.0), 0.0, 1.0))

        return Proposal(action=a, p=p, yaw=yaw, R=grasp_R(yaw), g=g,
                        box_hit=bool(np.any(p[:2] != free[:2])),
                        floor_hit=bool(free[2] < self.floor_z),
                        ceiling_hit=bool(free[2] > ceiling_z))

    def commit(self, prop: Proposal, q: np.ndarray | None) -> bool:
        """Apply a proposal; `q` None means the IK step was refused. True on an attempt edge."""
        s = self.state
        if q is not None:
            q_new = _vector(q, 6, "q")
            s.p_cmd = prop.p.copy()
            s.yaw_cmd = prop.yaw
            s.q_cmd = q_new

        closing = prop.g < CLOSE_BELOW
        attempt = closing and not s.closing
        s.closing = closing
        s.attempts += int(attempt)
        s.g_cmd = prop.g
        s.prev_action = prop.action.copy()
        s.ticks += 1
        return attempt

    def obs_tail(self) -> np.ndarray:
        """(9,) float32: sin 2e, cos 2e, g_cmd, prev_action (5), attempts / 3 (not clipped)."""
        s = self.state
        e = wrap(s.yaw_cmd - s.believed_yaw)
        return np.array([math.sin(2 * e), math.cos(2 * e), s.g_cmd, *s.prev_action, s.attempts / 3.0],
                        dtype=np.float32)
