"""The 19-vector the grasp student reads, built as training's `student_obs` builds it.

    [joints normalised to -1..1 (6), gap * 10, (believed - TCP) * 10 (3), command tail (9)]

The strategy gives the NEWEST measured joints, in radians, as PR 25 did; only the believed pixel
uses the joints paired with the frame's exposure. Training drew both from one instant, so with a
camera latency the two choices differ by the arm's motion over that latency; parity with PR 25,
the design that grasped 46 of 50 cubes in the twin's simulation, decided.

The joints are normalised with the MJCF ranges training used (`JOINT_LO` / `JOINT_HI`), never
with the app limits, and the vector is returned raw: the checkpoint's statistics are applied only
inside the policy. The TCP is forward kinematics of the same joints in the base frame; training's
positions are world coordinates, but the vector holds only a difference, so the frames agree.

One observer per thread: it drives its `Kinematics`, which mutates its `MjData`.
"""

from __future__ import annotations

import numpy as np

from ..arm import Kinematics
from .command import GraspCommand
from .contract import JOINT_HI, JOINT_LO, OBS_DIM

#: How far [rad] the model's joint ranges may lie from training's before the vector is refused.
RANGE_TOL = 1e-4

_LO = np.asarray(JOINT_LO, float)
_HI = np.asarray(JOINT_HI, float)


def range_mismatch(kin: Kinematics) -> str | None:
    """Why the model's arm joint ranges are not training's; None when they are."""
    lo = np.asarray(kin.lo, float)
    hi = np.asarray(kin.hi, float)
    if lo.shape == _LO.shape and np.max(np.abs(lo - _LO)) <= RANGE_TOL and np.max(np.abs(hi - _HI)) <= RANGE_TOL:
        return None
    return (f"arm joint ranges {lo.tolist()} .. {hi.tolist()} are not the ranges the policy was trained "
            f"with {list(JOINT_LO)} .. {list(JOINT_HI)}")


def _joints(q: np.ndarray) -> np.ndarray:
    v = np.asarray(q, float)
    if v.shape != (6,):
        raise ValueError(f"joints: expected shape (6,) [rad], got {v.shape}")
    return v


class GraspObserver:
    """The policy's proprioceptive input from the paired joints, the jaw gap and the command."""

    def __init__(self, kin: Kinematics) -> None:
        mismatch = range_mismatch(kin)
        if mismatch is not None:
            raise ValueError(mismatch)
        self.kin = kin

    def tcp(self, q: np.ndarray) -> np.ndarray:
        """TCP position (3,) of the joints `q` [rad], base frame, m."""
        return self.kin.fk(_joints(q))[:3, 3].copy()

    def build(self, q: np.ndarray, gap: float, cmd: GraspCommand) -> np.ndarray:
        """(19,) float32, not normalised. `q` the measured joints [rad], `gap` the fingertip gap [m]."""
        joints = _joints(q)
        normalised = 2.0 * (joints - _LO) / (_HI - _LO) - 1.0
        offset = (np.asarray(cmd.state.believed_pos, float) - self.tcp(joints)) * 10.0

        parts = [normalised, [float(gap) * 10.0], offset, cmd.obs_tail()]
        obs = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"observation has shape {obs.shape}, expected ({OBS_DIM},)")
        return obs
