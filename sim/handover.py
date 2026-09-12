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

from a1x_control import CLOSED, DOWN_Y, OPEN, Script, rot

HERE = os.path.dirname(os.path.abspath(__file__))


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
