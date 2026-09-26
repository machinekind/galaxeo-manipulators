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

Joint limits come from the model and are never left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np

from .model import GRIPPER, Tool, build, indices


@dataclass
class IKResult:
    q: np.ndarray        # arm_joint1..6 [rad], within the model limits
    pos_err: float       # tool position residual [m]
    rot_err: float       # tool rotation residual [rad]; 0 when no rotation was asked
    ok: bool             # pos_err < tol; orientation is best effort


class Kinematics:
    """FK/IK for one tool on its own MjData (does not touch any simulation state)."""

    def __init__(self, tool: Tool = GRIPPER, model: Optional[mujoco.MjModel] = None):
        self.tool = tool
        self.model = model if model is not None else build(tool)
        self.data = mujoco.MjData(self.model)
        self.ix = indices(self.model)
        self.lo, self.hi = self.ix.lo, self.ix.hi
        self.home = self.ix.home

    def _apply(self, q: np.ndarray) -> None:
        d = self.data
        d.qpos[self.ix.qadr] = q
        d.qpos[self.ix.finger_qadr] = self.ix.finger_home
        mujoco.mj_kinematics(self.model, d)

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Tool pose in the base frame, 4x4."""
        self._apply(np.asarray(q, float))
        d = self.data
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self.ix.site].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self.ix.site]
        return T

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
