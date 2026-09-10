#!/usr/bin/env python3
"""Scripted demo: the leader A1X picks a ball off the table and hands it to
the follower A1X, which sets it down on its own side.

    sim/.venv/bin/mjpython sim/handover.py                 # live viewer (macOS needs mjpython)
    sim/.venv/bin/python   sim/handover.py --gif out.gif   # headless recording

Pure position control: every waypoint is a tool-centre-point pose solved by
damped-least-squares IK on MuJoCo's site Jacobian, then joint targets are
interpolated with a smoothstep and fed to the position servos. Grasping is
real contact physics -- no welds, no cheating.
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN, CLOSED = 0.05, 0.0           # finger slide targets [m]; plates meet at 0


def rot(x_axis, y_axis):
    """Rotation matrix whose columns are the gripper's x (approach) and y (finger closing) axes."""
    x = np.asarray(x_axis, float); x /= np.linalg.norm(x)
    y = np.asarray(y_axis, float); y -= x * (x @ y); y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


DOWN_Y = rot([0, 0, -1], [0, 1, 0])          # pointing down, fingers close along world y


class Arm:
    def __init__(self, model, prefix):
        self.model = model
        joints = [model.joint(f"{prefix}arm_joint{i}").id for i in range(1, 7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo, self.hi = model.jnt_range[joints].T
        self.acts = np.array([model.actuator(f"{prefix}arm{i}").id for i in range(1, 7)])
        self.grip = model.actuator(f"{prefix}gripper_finger1").id
        self.site = model.site(f"{prefix}tcp").id
        self.scratch = mujoco.MjData(model)

    def ik(self, q0, pos, R, iters=500, damping=0.05):
        """Damped least squares on the 6-D pose error. Returns (q, converged)."""
        d, m = self.scratch, self.model
        q = np.clip(q0.copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        q_target = np.zeros(4); mujoco.mju_mat2Quat(q_target, R.ravel())
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


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class Script:
    """A list of timed phases. Each phase moves one arm to an IK pose and/or
    sets a gripper target; joint targets ramp over the phase duration."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.arms = {n: Arm(model, f"{n}/") for n in ("leader", "follower")}
        self.phases = []      # (label, t_start, t_end, arm, q_from, q_to, grip_from, grip_to)
        self.t = 0.0
        self.q = {n: data.qpos[a.qadr].copy() for n, a in self.arms.items()}
        self.g = {n: data.ctrl[a.grip] for n, a in self.arms.items()}

    def move(self, arm, pos, R, dur, label, grip=None):
        q, ok = self.arms[arm].ik(self.q[arm], np.asarray(pos, float), R)
        if not ok:
            print(f"warning: IK did not converge for '{label}'", file=sys.stderr)
        g_to = self.g[arm] if grip is None else grip
        self.phases.append((label, self.t, self.t + dur, arm, self.q[arm], q, self.g[arm], g_to))
        self.q[arm], self.g[arm], self.t = q, g_to, self.t + dur
        return self

    def gripper(self, arm, value, dur, label):
        self.phases.append((label, self.t, self.t + dur, arm, self.q[arm], self.q[arm], self.g[arm], value))
        self.g[arm], self.t = value, self.t + dur
        return self

    def wait(self, dur):
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


def build_script(model, data):
    ball = data.xpos[model.body("ball").id].copy()
    H = np.array([0.18, 0.0, 0.97])                 # handover point, between the arms
    P = np.array([0.25, -0.3, 0.73])                # where the follower sets it down
    L_hand = rot([0, -1, 0], [1, 0, 0])             # leader: point at follower, close along x
    F_hand = rot([0, 1, 0], [0, 0, 1])              # follower: point at leader, close along z
    up = np.array([0, 0, 0.12])
    # Fingertips reach 0.033 past the TCP, so keep the TCP 8 mm above the ball
    # centre when picking off the table: tips clear the surface by 5 mm.
    pick = ball + [0, 0, 0.008]
    s = Script(model, data)
    (s.move("leader", ball + up, DOWN_Y, 2.0, "leader: above ball", grip=OPEN)
      .move("leader", pick, DOWN_Y, 1.5, "leader: descend")
      .gripper("leader", CLOSED, 1.0, "leader: grasp")
      .move("leader", ball + up, DOWN_Y, 1.5, "leader: lift")
      .move("leader", H, L_hand, 2.5, "leader: to handover")
      .move("follower", H + [0, -0.12, 0], F_hand, 2.5, "follower: approach", grip=OPEN)
      .move("follower", H + [0, -0.005, 0], F_hand, 1.5, "follower: reach in")
      .gripper("follower", CLOSED, 1.0, "follower: grasp")
      .gripper("leader", OPEN, 0.8, "leader: release")
      .move("leader", H + [0, 0.07, 0.10], L_hand, 1.5, "leader: back off")
      .move("follower", P + up + [0, 0, 0.08], DOWN_Y, 3.0, "follower: carry")
      .move("follower", P + [0, 0, 0.008], DOWN_Y, 1.5, "follower: lower")
      .gripper("follower", OPEN, 0.8, "follower: release")
      .move("follower", P + up, DOWN_Y, 1.5, "follower: retreat")
      .wait(1.0))
    return s


def load():
    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def run_headless(gif, fps=20, width=800, height=500):
    model, data = load()
    script = build_script(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.2, 0, 0.88); cam.distance, cam.azimuth, cam.elevation = 1.3, 150, -15
    frames, last_label = [], None
    with mujoco.Renderer(model, height=height, width=width) as r:
        while data.time < script.t:
            label = script.apply(data.time)
            if label != last_label:
                print(f"t={data.time:5.1f}s  {label}")
                last_label = label
            mujoco.mj_step(model, data)
            if gif and int(data.time * fps) > len(frames) - 1:
                r.update_scene(data, camera=cam); frames.append(r.render())
    ball = data.xpos[model.body("ball").id]
    print(f"ball ended at {np.round(ball, 3)}; target was (0.25, -0.3, 0.73)")
    if gif:
        from PIL import Image
        Image.fromarray(frames[0]).save(gif, save_all=True, optimize=True, loop=0,
                                        duration=int(1000 / fps),
                                        append_images=[Image.fromarray(f) for f in frames[1:]])
        print("wrote", gif, f"({len(frames)} frames)")
    return ball


def run_viewer():
    import mujoco.viewer
    model, data = load()
    script = build_script(model, data)
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = (0.2, 0, 0.88); v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.3, 150, -15
        last_label = None
        while v.is_running():
            t0 = time.time()
            label = script.apply(data.time)
            if label != last_label:
                print(f"t={data.time:5.1f}s  {label}"); last_label = label
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))
            if data.time >= script.t + 2.0:              # loop the demo
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                script = build_script(model, data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", help="record headless to this GIF instead of opening the viewer")
    ap.add_argument("--headless", action="store_true", help="run without viewer or recording")
    args = ap.parse_args()
    if args.gif or args.headless:
        run_headless(args.gif)
    else:
        try:
            run_viewer()
        except RuntimeError as e:          # launch_passive on macOS outside mjpython
            sys.exit(f"{e}\nOn macOS run:  sim/.venv/bin/mjpython sim/handover.py")
