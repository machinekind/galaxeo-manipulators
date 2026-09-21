#!/usr/bin/env python3
"""Shared A1X control machinery: IK on the TCP site, and a timed phase script.

Used by both `handover.py` and the pick-and-place planner in `planner/`.

    from a1x_control import Arm, Script, rot, smoothstep, OPEN, CLOSED

Everything here is pure position control: a waypoint is a tool-centre-point
pose, solved by damped-least-squares IK on MuJoCo's site Jacobian, and joint
targets are interpolated with a smoothstep into the position servos.
"""
import sys

import mujoco
import numpy as np

OPEN, CLOSED = 0.05, 0.0           # finger slide targets [m]; plates meet at 0
TIP_AHEAD = 0.033                  # fingertips reach this far past the TCP [m]
TIP_GAP0 = 0.003                   # fingertip face separation when fully closed [m]


def rot(x_axis, y_axis):
    """Rotation matrix whose columns are the gripper's x (approach) and y (finger closing) axes."""
    x = np.asarray(x_axis, float); x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, float); y = y - x * (x @ y); y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


DOWN_Y = rot([0, 0, -1], [0, 1, 0])          # pointing down, fingers close along world y


class Arm:
    """One A1X: its six joints, its actuators, its gripper and its TCP site."""

    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix
        joints = [model.joint(f"{prefix}arm_joint{i}").id for i in range(1, 7)]
        self.joints = np.array(joints)
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo, self.hi = model.jnt_range[joints].T
        self.acts = np.array([model.actuator(f"{prefix}arm{i}").id for i in range(1, 7)])
        self.grip = model.actuator(f"{prefix}gripper_finger1").id
        self.fingers = [model.joint(f"{prefix}gripper_finger_joint{i}").id for i in (1, 2)]
        self.fadr = np.array([model.jnt_qposadr[j] for j in self.fingers])
        self.site = model.site(f"{prefix}tcp").id
        self.scratch = mujoco.MjData(model)

    def ik(self, q0, pos, R, iters=500, damping=0.05):
        """Damped least squares on the 6-D pose error. Returns (q, converged)."""
        d, m = self.scratch, self.model
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        q_target = np.zeros(4); mujoco.mju_mat2Quat(q_target, np.asarray(R, float).ravel())
        q_cur = np.zeros(4); q_err = np.zeros(4); w = np.zeros(3)
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            e_pos = pos - d.site_xpos[self.site]
            mujoco.mju_mat2Quat(q_cur, d.site_xmat[self.site])
            mujoco.mju_negQuat(q_cur, q_cur)
            mujoco.mju_mulQuat(q_err, q_target, q_cur)
            mujoco.mju_quat2Vel(w, q_err, 1.0)
            err = np.concatenate([e_pos, w])
            if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(w) < 5e-3:
                return q, True
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            step = min(1.0, 0.3 / (np.linalg.norm(dq) + 1e-9))     # cap to 0.3 rad per iter
            q = np.clip(q + step * dq, self.lo, self.hi)
        return q, False

    def tcp_pose(self, data):
        """(position, rotation matrix) of the TCP site in world coordinates."""
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def gap(self, data):
        """Fingertip face separation [m]: 0.003 fully closed, 0.103 fully open.

        The curled tips are the innermost surfaces, so this is what a real
        gripper-gap sensor would report and what the grasp check uses."""
        return float(TIP_GAP0 + abs(data.qpos[self.fadr[0]]) + abs(data.qpos[self.fadr[1]]))


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class Script:
    """A list of timed phases. Each phase moves one arm to an IK pose and/or
    sets a gripper target; joint targets ramp over the phase duration.

    `arms` maps a short name to the model prefix, e.g. {"arm": "arm/"}.

    `t0` is where the script's clock starts. It matters when a script takes
    over a simulation that is already running (the planner finishing an episode
    a policy began): phase times are compared against `data.time`, so a script
    built at t = 0 while the world is at t = 12 would have all its phases
    already in the past. Ramps always start from the measured joint angles and
    the current gripper command, so only the clock needs telling.
    """

    def __init__(self, model, data, arms=None, t0=0.0):
        self.m, self.d = model, data
        if arms is None:
            arms = {n: f"{n}/" for n in ("leader", "follower")}
        self.arms = {n: Arm(model, p) for n, p in arms.items()}
        self.phases = []      # (label, t_start, t_end, arm, q_from, q_to, grip_from, grip_to)
        self.t = float(t0)
        self.last_ok = True
        self.q = {n: data.qpos[a.qadr].copy() for n, a in self.arms.items()}
        self.g = {n: data.ctrl[a.grip] for n, a in self.arms.items()}

    def move(self, arm, pos, R, dur, label, grip=None, quiet=False):
        """Solve IK for a TCP pose and ramp to it. `last_ok` records convergence."""
        q, self.last_ok = self.arms[arm].ik(self.q[arm], np.asarray(pos, float), R)
        if not self.last_ok and not quiet:
            print(f"warning: IK did not converge for '{label}'", file=sys.stderr)
        return self.move_q(arm, q, dur, label, grip=grip)

    def move_q(self, arm, q, dur, label, grip=None):
        """Ramp straight to a joint configuration (already solved elsewhere)."""
        g_to = self.g[arm] if grip is None else grip
        self.phases.append((label, self.t, self.t + dur, arm, self.q[arm], np.asarray(q, float),
                            self.g[arm], g_to))
        self.q[arm], self.g[arm], self.t = np.asarray(q, float), g_to, self.t + dur
        return self

    def gripper(self, arm, value, dur, label):
        self.phases.append((label, self.t, self.t + dur, arm, self.q[arm], self.q[arm], self.g[arm], value))
        self.g[arm], self.t = value, self.t + dur
        return self

    def wait(self, dur, label=None):
        if label is not None:      # hold the current targets, but give the phase a name
            arm = next(iter(self.arms))
            self.phases.append((label, self.t, self.t + dur, arm, self.q[arm], self.q[arm],
                                self.g[arm], self.g[arm]))
        self.t += dur
        return self

    def apply(self, t):
        """Write servo targets for time t. Returns the label of the active phase."""
        active = None
        for label, t0, t1, arm, q0, q1, g0, g1 in self.phases:
            a = self.arms[arm]
            if t < t0:
                continue
            s = smoothstep((t - t0) / (t1 - t0))
            self.d.ctrl[a.acts] = q0 + s * (q1 - q0)
            self.d.ctrl[a.grip] = g0 + s * (g1 - g0)
            if t < t1:
                active = label
        return active
