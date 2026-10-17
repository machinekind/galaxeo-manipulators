"""A staged joint-space plan: the A1X's joint groups move one after another.

From a folded pose one straight move of all joints can turn the wrist into the shoulder column.
`stage_plan` moves the elbow, then the wrist, then shoulder and base, as `home_stages` in
galaxeo/robot/a1x/robot.yaml does. If a segment collides it tries the stages in reverse, then
one straight move. The caller's collision check decides, so any goal from any IK fits. Joints are
radians, as everywhere in `galaxeo.arm`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .model import JOINTS

#: The A1X's staged home without the gripper stage: elbow, wrist, then shoulder and base.
#: A test holds it equal to `home_stages` in galaxeo/robot/a1x/robot.yaml.
STAGES: tuple[tuple[str, ...], ...] = (
    ("arm_joint3",),
    ("arm_joint4", "arm_joint5", "arm_joint6"),
    ("arm_joint2", "arm_joint1"),
)

#: A stage whose largest joint move is under this [rad] (0.001 deg) is skipped.
STAGE_MIN_RAD = math.radians(1e-3)


@dataclass(frozen=True)
class StagePlan:
    """Intermediate poses [rad] to the goal, the goal last; `collision` set when no plan is clear."""

    poses: tuple[np.ndarray, ...]
    collision: str | None


def _indices(stage: Sequence[str]) -> list[int]:
    return [JOINTS.index(name) for name in stage]


def _walk(start: np.ndarray, goal: np.ndarray, stages: Sequence[Sequence[str]],
          clear: Callable[[np.ndarray, np.ndarray], str | None]) -> StagePlan:
    pose = start.copy()
    poses: list[np.ndarray] = []
    for stage in stages:
        nxt = pose.copy()
        idx = _indices(stage)
        nxt[idx] = goal[idx]
        if float(np.max(np.abs(nxt - pose))) < STAGE_MIN_RAD:
            continue

        collision = clear(pose, nxt)
        if collision is not None:
            return StagePlan(poses=tuple(poses), collision=collision)

        poses.append(nxt)
        pose = nxt
    return StagePlan(poses=tuple(poses), collision=None)


def stage_plan(start: Sequence[float], goal: Sequence[float],
               clear: Callable[[np.ndarray, np.ndarray], str | None]) -> StagePlan:
    """Intermediate poses from `start` to `goal` [rad]: `STAGES` in order, else in reverse, else one
    straight move.

    `clear(a, b)` returns None when the straight joint move from a to b is clear, else a description
    (`CollisionChecker.path_clear` fits). The first plan whose every segment passes wins. With none,
    the plan has no poses and names the first collision.
    """
    q0 = np.asarray(start, float).reshape(6)
    q1 = np.asarray(goal, float).reshape(6)

    first_collision: str | None = None
    for order in (STAGES, STAGES[::-1], ()):
        plan = _walk(q0, q1, [*order, JOINTS], clear)
        if plan.collision is None:
            return plan
        first_collision = first_collision or plan.collision
    return StagePlan(poses=(), collision=first_collision)
