"""Forward and inverse kinematics of the tool frame, computed by MuJoCo on the installation model.

The same MJCF drives FK, IK, collision checks and the sim, so they cannot disagree
about geometry. base_link is the world origin of that model, so every pose here is
in the base frame.

IK is damped least squares on the tool-site Jacobian, with two choices that matter:

  * Position has priority over orientation. Orientation is corrected only in the
    null space of the position Jacobian, so a rotation the arm cannot reach never
    pulls the tool away from the point (weighting metres against radians in one
    sum did, by 20 cm for a target rotated 90 deg).
  * Several starts: the seed, the model's home pose and a few random ones, best
    result wins. A single DLS start gets stuck in local minima (30 cm residual was
    measured over most of the A1X workspace).

`ik_step` is the other solver: the one the grasp policy was trained with, one warm
6-D start per control step and nothing else, so the policy sees the joints it learnt on.

Joint limits come from the model and are never left.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np

from .model import GRIPPER, Tool, build, finger_qpos, indices

# Training's step cap: a bigger step in a badly conditioned pose jumps to another branch.
STEP_CAP_RAD = 0.3

_BOX_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


@dataclass
class IKResult:
    q: np.ndarray        # arm_joint1..6 [rad], within the model limits
    pos_err: float       # tool position residual [m]
    rot_err: float       # tool rotation residual [rad]; 0 when no rotation was asked
    ok: bool             # pos_err < tol; orientation is best effort


@dataclass
class IKStep:
    q: np.ndarray        # arm_joint1..6 [rad], clipped to lo/hi
    pos_err: float       # [m] at the returned q
    rot_err: float       # [rad] at the returned q
    jump: float          # max |q - seed| [rad], seed as given (before clipping)
    iters: int           # updates made; 0 when the seed already meets the tolerances
    converged: bool      # True only if the loop exited through the tolerance test


class Kinematics:
    """FK/IK for one tool on its own MjData (does not touch any simulation state)."""

    def __init__(self, tool: Tool = GRIPPER, model: Optional[mujoco.MjModel] = None):
        self.tool = tool
        self.model = model if model is not None else build(tool)
        self.data = mujoco.MjData(self.model)
        self.ix = indices(self.model)
        self.lo, self.hi = self.ix.lo, self.ix.hi
        self.home = self.ix.home

    def _apply(self, q: np.ndarray, opening: Optional[float] = None) -> None:
        d = self.data
        d.qpos[self.ix.qadr] = q
        d.qpos[self.ix.finger_qadr] = finger_qpos(self.ix, opening)
        mujoco.mj_kinematics(self.model, d)

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Tool pose in the base frame, 4x4."""
        self._apply(np.asarray(q, float))
        d = self.data
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self.ix.site].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self.ix.site]
        return T

    def link_pose(self, q: Sequence[float], body: str) -> np.ndarray:
        """Pose of a body of the arm in the base frame, 4x4; the fingers at the home opening."""
        self._apply(np.asarray(q, float))
        d = self.data
        i = self.model.body(body).id
        T = np.eye(4)
        T[:3, :3] = d.xmat[i].reshape(3, 3)
        T[:3, 3] = d.xpos[i]
        return T

    def fingertips(self, q: Sequence[float], opening: Optional[float] = None) -> np.ndarray:
        """Lowest corner (base z) of each curled fingertip box in the base frame [m], rows finger 1, finger 2.

        A tilted tool puts a corner below the tool point: an open finger rolled 5 deg by joint 5
        sits 5 mm lower, so a table rule on the tool point alone lets the fingertips into the table.
        `opening` [m] as in `model.finger_qpos`; None = the home opening.
        """
        self._apply(np.asarray(q, float), opening)
        m = self.model
        d = self.data
        out = np.empty((2, 3))
        for row, g in enumerate(self.ix.tips):
            corners = d.geom_xpos[g] + (_BOX_CORNERS * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T
            out[row] = corners[np.argmin(corners[:, 2])]
        return out

    def ik_step(
        self,
        position: Sequence[float],
        rotation: np.ndarray,
        seed: Sequence[float],
        *,
        iters: int = 60,
        damping: float = 0.05,
        tol_pos: float = 1e-3,
        tol_rot: float = 5e-3,
        lo: Optional[Sequence[float]] = None,
        hi: Optional[Sequence[float]] = None,
    ) -> IKStep:
        """One warm start of the grasp training's solver (sim/a1x_control.py, Arm.ik), step for step.

        The grasp policy learnt on exactly these joints and residuals, so this is 6-D damped least
        squares from `seed` with no restarts and no randomness; `ik` lands elsewhere. `lo`/`hi` [rad]
        narrow the model ranges, so a step the caller's limits would cut comes back unconverged.

        The tolerance test runs before each update, as in training. When the loop runs out, the
        errors are recomputed at the final q but `converged` stays False even if they meet the
        tolerances: training refused such a step, and taking it would command what the policy never saw.
        """
        q_seed = np.asarray(seed, float).reshape(6)
        q_lo = self.lo if lo is None else np.maximum(self.lo, np.asarray(lo, float).reshape(6))
        q_hi = self.hi if hi is None else np.minimum(self.hi, np.asarray(hi, float).reshape(6))
        if np.any(q_lo > q_hi):
            raise ValueError(f"ik_step: lo {q_lo.tolist()} above hi {q_hi.tolist()}")
        q = np.clip(q_seed, q_lo, q_hi)

        target = np.asarray(position, float).reshape(3)
        want = np.zeros(4)
        mujoco.mju_mat2Quat(want, np.ascontiguousarray(rotation, float).ravel())

        m = self.model
        d = self.data
        cols = self.ix.dadr
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        converged = False
        updates = 0
        for _ in range(iters):
            err = self._pose_error(q, target, want)
            if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
                converged = True
                break

            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.ix.site)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            step = min(1.0, STEP_CAP_RAD / (np.linalg.norm(dq) + 1e-9))
            q = np.clip(q + step * dq, q_lo, q_hi)
            updates += 1

        if not converged:
            err = self._pose_error(q, target, want)

        return IKStep(
            q=q,
            pos_err=float(np.linalg.norm(err[:3])),
            rot_err=float(np.linalg.norm(err[3:])),
            jump=float(np.max(np.abs(q - q_seed))),
            iters=updates,
            converged=converged,
        )

    def _pose_error(self, q: np.ndarray, target: np.ndarray, want: np.ndarray) -> np.ndarray:
        """Tool position error [m] and rotation error as a rotation vector [rad] at q, as training measured them."""
        self._apply(q)
        d = self.data
        cur = np.zeros(4)
        err = np.zeros(4)
        w = np.zeros(3)
        mujoco.mju_mat2Quat(cur, d.site_xmat[self.ix.site])
        mujoco.mju_negQuat(cur, cur)
        mujoco.mju_mulQuat(err, want, cur)
        mujoco.mju_quat2Vel(w, err, 1.0)
        return np.concatenate([target - d.site_xpos[self.ix.site], w])

    def ik(
        self,
        position: Sequence[float],
        rotation: Optional[np.ndarray] = None,
        seed: Optional[Sequence[float]] = None,
        *,
        tol: float = 0.003,
        rot_gain: float = 0.5,
        iters: int = 300,
        damping: float = 0.02,
        restarts: int = 6,
        rng: Optional[np.random.Generator] = None,
    ) -> IKResult:
        """Joints that put the tool at `position` [m], rotated as close to `rotation` as it allows."""
        rng = rng if rng is not None else np.random.default_rng(0)
        q_seed = np.clip(np.asarray(seed if seed is not None else self.home, float), self.lo, self.hi)
        starts = [q_seed, self.home.copy()]
        starts += [rng.uniform(self.lo, self.hi) for _ in range(restarts)]

        target = np.asarray(position, float).reshape(3)
        best = None
        for q0 in starts:
            q, e_pos, e_rot = self._solve(q0, target, rotation, rot_gain, iters, damping, tol)
            hit = e_pos < tol
            key = (not hit, e_rot if hit else e_pos)
            if best is None or key < best[0]:
                best = (key, q, e_pos, e_rot)
            if hit and (rotation is None or e_rot < 0.02):
                break
        _, q, e_pos, e_rot = best
        if rotation is not None and e_pos >= tol:
            # The rotation steps can pin joints at their limits and cost the point: the point wins.
            q2, e2, _ = self._solve(q, target, None, rot_gain, iters, damping, tol)
            if e2 < e_pos:
                q, e_pos, e_rot = q2, e2, self._rot_err(q2, rotation)
        return IKResult(q, float(e_pos), float(e_rot), bool(e_pos < tol))

    def _rot_err(self, q, rotation) -> float:
        R = self.fk(q)[:3, :3]
        c = (np.trace(np.asarray(rotation, float) @ R.T) - 1.0) / 2.0
        return float(np.arccos(np.clip(c, -1.0, 1.0)))

    def _solve(self, q, target, rotation, rot_gain, iters, damping, tol):
        m, d, site = self.model, self.data, self.ix.site
        q = np.clip(np.array(q, float), self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.ix.dadr
        want = np.zeros(4)
        cur = np.zeros(4)
        err = np.zeros(4)
        w = np.zeros(3)
        if rotation is not None:
            mujoco.mju_mat2Quat(want, np.asarray(rotation, float).ravel())
        e_rot = 0.0
        for _ in range(iters):
            self._apply(q)
            e_p = target - d.site_xpos[site]
            e_pos = float(np.linalg.norm(e_p))
            if rotation is not None:
                mujoco.mju_mat2Quat(cur, d.site_xmat[site])
                mujoco.mju_negQuat(cur, cur)
                mujoco.mju_mulQuat(err, want, cur)
                mujoco.mju_quat2Vel(w, err, 1.0)
                e_rot = float(np.linalg.norm(w))
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            Jp = jacp[:, cols]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping ** 2 * np.eye(3))
            dq = Jp_pinv @ e_p
            if rotation is not None:
                dq_rot = (np.eye(len(q)) - Jp_pinv @ Jp) @ (rot_gain * jacr[:, cols].T @ w)
                dq = dq + dq_rot
                if e_pos < tol * 0.5 and float(np.linalg.norm(dq_rot)) < 1e-4:
                    break
            elif e_pos < tol * 0.5:
                break
            # A big step in a badly conditioned pose jumps to another branch and never returns.
            step = min(1.0, 0.3 / (float(np.linalg.norm(dq)) + 1e-9))
            q = np.clip(q + step * dq, self.lo, self.hi)
        self._apply(q)
        e_pos = float(np.linalg.norm(target - d.site_xpos[site]))
        if rotation is not None:
            mujoco.mju_mat2Quat(cur, d.site_xmat[site])
            mujoco.mju_negQuat(cur, cur)
            mujoco.mju_mulQuat(err, want, cur)
            mujoco.mju_quat2Vel(w, err, 1.0)
            e_rot = float(np.linalg.norm(w))
        return q, e_pos, e_rot


def down_rotation(point: Sequence[float]) -> np.ndarray:
    """Tool rotation with the approach axis (tool x) pointing down and the jaws (tool y) closing
    across the line from the base to `point`."""
    bearing = float(np.arctan2(point[1], point[0]))
    down = np.array([0.0, 0.0, -1.0])
    side = np.array([-np.sin(bearing), np.cos(bearing), 0.0])
    return np.column_stack([down, side, np.cross(down, side)])
