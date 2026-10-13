"""The hover an episode starts from, and the staged joint plan that takes the arm there.

The hover is the believed grasp plus `HOVER` in z, the tool pointing down and the closing axis at
the believed yaw. It is solved once per run with `galaxeo.arm`'s full IK from the model's home
pose, for the yaw and for the yaw + pi (the pinch is symmetric, the joints are not: J6 differs by
180 deg), and the solution whose J6 is further from its limits wins. Solving from home keeps J5 at
zero, as in every training hover. The IK's random restarts start from a fixed seed, so one point
always gives one hover.

`stage_plan` drives the A1X's joint groups one after another, in the order of `home_stages` in
galaxeo/robot/a1x/robot.yaml, else in reverse, else in one straight move; the caller's collision
question decides. Joints are radians here, as everywhere in `galaxeo.arm`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..arm import Kinematics
from ..arm.model import JOINTS
from .contract import (
    HOVER,
    HOVER_J5_MAX_DEG,
    HOVER_J6_MARGIN_DEG,
    IK_TOL_POS,
    IK_TOL_ROT,
    TRAIN_HOVER_Z,
    TRAIN_J1_DEG,
    TRAIN_RADIUS,
    grasp_R,
    wrap,
)

logger = logging.getLogger(__name__)

#: The A1X's staged home without the gripper stage: elbow, wrist, then shoulder and base.
#: A test holds it equal to `home_stages` in galaxeo/robot/a1x/robot.yaml.
STAGES: tuple[tuple[str, ...], ...] = (
    ("arm_joint3",),
    ("arm_joint4", "arm_joint5", "arm_joint6"),
    ("arm_joint2", "arm_joint1"),
)

#: The seed of the full IK's random restarts.
IK_SEED = 0
#: A stage whose largest joint move is under this [rad] (0.001 deg) is skipped.
STAGE_MIN_RAD = math.radians(1e-3)

J5 = 4
J6 = 5


class HoverError(RuntimeError):
    """No usable hover; `detail` names the failed checks in Polish, for the operator."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class JointBox:
    """Arm joint limits [rad]: the caller's limits within the model's ranges."""

    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class Hover:
    """A solved hover, base frame [m]; `q` the six arm joints [rad]."""

    p: np.ndarray
    yaw: float                         # closing-axis yaw: the believed yaw or the believed yaw + pi, wrapped
    believed: np.ndarray               # == p - [0, 0, HOVER]
    q: np.ndarray
    j6_margin_deg: float
    outside_training: tuple[str, ...]  # empty inside the training envelope, else the failed checks


@dataclass(frozen=True)
class _Candidate:
    yaw: float
    q: np.ndarray
    j6_margin_deg: float
    failed: tuple[str, ...]


@dataclass(frozen=True)
class StagePlan:
    """Intermediate poses [rad] to the goal, the goal last; `collision` set when no plan is clear."""

    poses: tuple[np.ndarray, ...]
    collision: str | None


def joint_box(kin: Kinematics, limits: Mapping[str, tuple[float, float]]) -> JointBox:
    """The caller's limits [deg] of the arm joints narrowed to the model's ranges [rad]."""
    lo = np.array(kin.lo, float)
    hi = np.array(kin.hi, float)
    for k, name in enumerate(JOINTS):
        if name not in limits:
            continue
        low, high = limits[name]
        lo[k] = max(lo[k], math.radians(float(low)))
        hi[k] = min(hi[k], math.radians(float(high)))
    return JointBox(lo=lo, hi=hi)


def _rotation_angle(R: np.ndarray) -> float:
    """Angle of a rotation matrix; atan2 keeps it accurate near zero, where acos is not."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(math.atan2(0.5 * float(np.linalg.norm(w)), 0.5 * (float(np.trace(R)) - 1.0)))


def _candidate(kin: Kinematics, p: np.ndarray, yaw: float, box: JointBox) -> _Candidate:
    R = grasp_R(yaw)
    sol = kin.ik(p, R, seed=kin.home, rng=np.random.default_rng(IK_SEED))

    # `sol.ok` ignores the rotation: judge the joints by forward kinematics.
    T = kin.fk(sol.q)
    pos_err = float(np.linalg.norm(p - T[:3, 3]))
    rot_err = _rotation_angle(R @ T[:3, :3].T)
    deg = np.degrees(sol.q)

    failed: list[str] = []
    if not pos_err < IK_TOL_POS:
        failed.append(f"pozycja {pos_err * 1000.0:.2f} mm")
    if not rot_err < IK_TOL_ROT:
        failed.append(f"obrot {rot_err * 1000.0:.2f} mrad")
    if not abs(deg[J5]) <= HOVER_J5_MAX_DEG:
        failed.append(f"J5 {deg[J5]:.2f} st.")

    outside = [f"{JOINTS[k]} {deg[k]:.1f} st." for k in range(6) if not box.lo[k] <= sol.q[k] <= box.hi[k]]
    if outside:
        failed.append("poza limitami " + ", ".join(outside))

    margin = min(deg[J6] - math.degrees(box.lo[J6]), math.degrees(box.hi[J6]) - deg[J6])
    if not margin >= HOVER_J6_MARGIN_DEG:
        failed.append(f"zapas J6 {margin:.1f} st.")

    return _Candidate(yaw=float(yaw), q=sol.q.copy(), j6_margin_deg=float(margin), failed=tuple(failed))


def solve_hover(kin: Kinematics, believed: Sequence[float], yaw: float, box: JointBox) -> Hover:
    """Hover above `believed` (base frame, m) with the closing axis at `yaw` or `yaw + pi`.

    Raises `HoverError` when neither candidate passes. `outside_training` is filled, not acted on.
    """
    point = np.asarray(believed, dtype=float).reshape(3)
    if not (np.all(np.isfinite(point)) and math.isfinite(yaw)):
        raise HoverError(f"nieskonczone: punkt {point.tolist()}, yaw {yaw}")

    p = point + np.array([0.0, 0.0, HOVER])
    candidates = [_candidate(kin, p, y, box) for y in (wrap(float(yaw)), wrap(float(yaw) + math.pi))]
    for c in candidates:
        logger.debug("hover yaw %.1f deg: joints %s deg, J6 margin %.1f deg, %s", math.degrees(c.yaw),
                     np.round(np.degrees(c.q), 2).tolist(), c.j6_margin_deg, c.failed or "ok")

    valid = [c for c in candidates if not c.failed]
    if not valid:
        raise HoverError("; ".join(f"yaw {math.degrees(c.yaw):.1f} st.: {', '.join(c.failed)}" for c in candidates))

    best = max(valid, key=lambda c: c.j6_margin_deg)
    return Hover(
        p=p,
        yaw=best.yaw,
        believed=p - np.array([0.0, 0.0, HOVER]),
        q=best.q,
        j6_margin_deg=best.j6_margin_deg,
        outside_training=training_envelope(best.q, p),
    )


def training_envelope(q: np.ndarray, p: np.ndarray) -> tuple[str, ...]:
    """Which checks of the training envelope a hover fails, in Polish; empty when inside."""
    out: list[str] = []

    j1 = math.degrees(float(q[0]))
    lo, hi = TRAIN_J1_DEG
    if not lo <= j1 <= hi:
        out.append(f"J1 {j1:.1f} st. poza {lo:g}..{hi:g}")

    radius = float(math.hypot(p[0], p[1]))
    lo, hi = TRAIN_RADIUS
    if not lo <= radius <= hi:
        out.append(f"promien {radius:.2f} m poza {lo:g}..{hi:g}")

    z = float(p[2])
    lo, hi = TRAIN_HOVER_Z
    if not lo <= z <= hi:
        out.append(f"wysokosc zawisu {z:.3f} m poza {lo:g}..{hi:g}")

    return tuple(out)


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
    straight move. The first plan whose every segment `clear` passes wins; with none, the first
    collision and no poses."""
    q0 = np.asarray(start, float).reshape(6)
    q1 = np.asarray(goal, float).reshape(6)

    first_collision: str | None = None
    for order in (STAGES, STAGES[::-1], ()):
        plan = _walk(q0, q1, [*order, JOINTS], clear)
        if plan.collision is None:
            return plan
        first_collision = first_collision or plan.collision
    return StagePlan(poses=(), collision=first_collision)
