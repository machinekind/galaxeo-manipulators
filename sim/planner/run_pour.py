#!/usr/bin/env python3
"""Run seeded pick-and-pour episodes headless and report the success rate.

    sim/.venv/bin/python sim/planner/run_pour.py --n 20 --seed 0
    sim/.venv/bin/python sim/planner/run_pour.py --n 1 --seed 3 --gif pour.gif
    sim/.venv/bin/python sim/planner/run_pour.py --n 20 --record /tmp/pour -v
    sim/.venv/bin/mjpython sim/planner/run_pour.py --view --seed 4        # live, loops seeds

Each episode is a freshly compiled `pour_scene.build(seed)`. Frames come from
`laptop_cam`, the randomly placed webcam the policy will see through.

`--lerobot ROOT` writes successful episodes straight into a LeRobot v3
dataset (needs the lerobot package: use sim/.venv-lerobot), with the same
schema `record_a1x.py` writes on the real arm:

    action                       (7,) commanded joint targets + gripper, 0 closed .. 0.05 open
    observation.state            (13,) measured joints + finger position + joint velocities.
                                 The velocities are what tells a 50-step chunk where in
                                 a ramp (or a settle wait) the arm is: without them the
                                 same pose has several futures, and a policy fitted to the
                                 average of them misses by degrees (see POUR.md)
    observation.images.laptop    (240, 320, 3) the webcam, downscaled from its 640x480 render
    observation.images.topdown   (256, 256, 3) that same frame warped onto the table
                                 plane: a 0.7 m square about the workspace, world +y up
                                 and +x right, 2.73 mm per pixel. See planner/topdown.py
    observation.images.wrist     (240, 320, 3) the G1 wrist camera, only when the scene has
                                 one (WRIST_CAMERA=left|right, or gen_dataset --wrist):
                                 downscaled from 640x480, with the seed's exposure draw
    observation.cam_K, cam_T     (9,) (16,) the session's camera calibration, for
                                 provenance only. `cam_K` is the intrinsics of the
                                 *640x480 render*, not of the stored 320x240 frame

The policy is no longer handed the calibration as a state vector: the top-down
map is what consumes it, and the frames arrive already in the arm's frame.

    sim/.venv-lerobot/bin/python sim/planner/run_pour.py --n 100 --lerobot data/pour_sim --fps 20
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from planner.perception import SimPourPerception  # noqa: E402
from planner.pour import Pour, ServoNoise  # noqa: E402
from planner.run_episodes import Recorder, write_gif  # noqa: E402
from planner.topdown import N as TD_N, SIDE as TD_SIDE, topdown  # noqa: E402
from pour_scene import WORKSPACE, build  # noqa: E402

STORE_W, STORE_H = 320, 240        # stored size of the raw webcam stream, and of the wrist stream
WRIST_KEY = "observation.images.wrist"
ENV_KEY = "observation.environment_state"
ENV_NAMES = ["bottle_x", "bottle_y", "bottle_z", "bottle_up_x", "bottle_up_y", "bottle_up_z",
             "glass_x", "glass_y", "bottle_body_r", "bottle_body_h", "bottle_height",
             "bottle_neck_r", "glass_r", "glass_h"]
STATE_NAMES = ([f"arm_joint{i}" for i in range(1, 7)] + ["gripper"] + [f"arm_joint{i}_vel" for i in range(1, 7)])


def robot_state(d, arm):
    """observation.state: the six measured joints, the finger opening and the
    six joint velocities, as `STATE_NAMES` lists them."""
    q, g = d.qpos[arm.qadr], abs(float(d.qpos[arm.fadr[0]]))
    return np.concatenate([q, [g], d.qvel[arm.dadr]]).astype(np.float32)


class EnvState:
    """The privileged state a *state-based* policy reads instead of images: the
    bottle's position and its up vector (the third column of its rotation: the
    tilt without the yaw, which a round bottle does not have), the glass's
    position on the table, and the geometry that decides where the planner
    grasps and how far it tilts. 14 floats, all in the world frame (the arm
    base is fixed). Nothing a camera would have to infer is left out, so a
    policy that fails on this input fails on the action side, not the
    perception side. The glass's height and the bottle's yaw were in the first
    version and dropped: one is constant to 0.1 mm, which mean-std
    normalisation turns into noise, the other varies by the full circle and
    means nothing."""

    def __init__(self, model, info):
        self.bottle = model.body(info["bottle"]["name"]).id
        self.glass = model.body(info["glass"]["name"]).id
        b, g = info["bottle"], info["glass"]
        self.geom = np.array([b["body_r"], b["body_h"], b["height"], b["neck_r"], g["r"], g["h"]], np.float32)

    def __call__(self, d):
        R = d.xmat[self.bottle]
        return np.concatenate([d.xpos[self.bottle], R[2::3], d.xpos[self.glass][:2],
                               self.geom]).astype(np.float32)


def clean_action(pp, arm):
    """The 7-vector the dataset labels a frame with: the planner's own command
    for the six joints, before any servo noise `Pour` added to what the
    actuators received, and the gripper command clipped to its stored range.
    Without noise `ctrl_clean` and `d.ctrl` are the same array of numbers."""
    c = pp.ctrl_clean
    return np.concatenate([c[arm.acts], [np.clip(c[arm.grip], 0.0, 0.05)]])


def wrist_frame(renderer, data, info, wh=(STORE_W, STORE_H)):
    """The wrist camera's frame the way the dataset stores it: rendered at the
    renderer's native 640x480, the seed's exposure draw applied (gain, then
    gamma -- MuJoCo has no exposure, and a camera 60 mm from the object does
    not hold the base camera's), then an area downscale. The evaluator calls
    this too, so a policy sees at test time what it was trained on."""
    w = info["wrist"]
    renderer.update_scene(data, camera=w["camera"])
    img = renderer.render().astype(np.float32) / 255.0
    img = np.clip(img * w["gain"], 0.0, 1.0) ** w["gamma"]
    img = (img * 255.0 + 0.5).astype(np.uint8)
    if (img.shape[1], img.shape[0]) != tuple(wh):
        img = cv2.resize(img, tuple(wh), interpolation=cv2.INTER_AREA)
    return img


class PourRecorder(Recorder):
    def save(self, path, info, result):
        np.savez_compressed(
            path, time=np.array(self.t), qpos=np.array(self.qpos), ctrl=np.array(self.ctrl),
            tcp=np.array(self.tcp), stage=np.array(self.stage),
            frames=np.array(self.frames, dtype=np.uint8), frame_time=np.array(self.ft),
            success=result.success, outcome=result.stage, seed=info["seed"],
            cam_K=info["cam"]["K"], cam_T_cam2base=info["cam"]["T_cam2base"],
            bottle=np.array([info["bottle"][k] for k in ("body_r", "body_h", "neck_r", "neck_h", "height", "mass")]),
            glass=np.array([info["glass"]["r"], info["glass"]["h"]]))


TASK = "pick up the bottle and pour it into the glass"


# The stages a recording keeps. The planner goes on to put the bottle down
# somewhere new and return home, which is the episode's reset, not the task:
# the put-down spot is random, so those frames have no predictable label, and
# the second of idling at home with the bottle standing on the table looks
# exactly like the start of an episode, which taught a policy to stay put.
RECORDED_STAGES = ("pick", "lift", "pour", "upright")


class StateRecorder:
    """Streams one episode's actions, joint states and privileged scene state at
    `fps`, rendering nothing. What `--state` datasets are made of."""

    def __init__(self, ds, model, info, fps, arm):
        self.ds, self.fps, self.arm, self.env = ds, fps, arm, EnvState(model, info)
        self.n = 0

    def __call__(self, pp):
        d = pp.d
        if pp.label not in RECORDED_STAGES or self.n > int(d.time * self.fps):
            return
        self.n += 1
        self.ds.add_frame({"action": clean_action(pp, self.arm).astype(np.float32),
                           "observation.state": robot_state(d, self.arm),
                           ENV_KEY: self.env(d), "task": TASK})


class LeRobotRecorder:
    """Streams one episode into an open LeRobotDataset at `fps`.

    The webcam is rendered at its native 640x480, because that is the
    resolution the top-down warp gets its detail from; the map is computed from
    the full-size frame and only then is the raw frame downscaled to the
    (`W`, `H`) that goes into the dataset."""

    def __init__(self, ds, renderer, info, fps, arm, wh=(STORE_W, STORE_H),
                 td_n=TD_N, td_side=TD_SIDE):
        self.ds, self.r, self.info, self.fps, self.arm = ds, renderer, info, fps, arm
        self.wh, self.td_n, self.td_side = wh, td_n, td_side
        self.n = 0
        # K of the 640x480 render, kept for provenance; the warp uses it too.
        self.K = np.asarray(info["cam"]["K"], np.float32).ravel()
        self.T = np.asarray(info["cam"]["T_cam2base"], np.float32).ravel()
        self.K3 = np.asarray(info["cam"]["K"], float).reshape(3, 3)
        self.T_cam2world = np.asarray(info["cam"]["T_cam2world"], float)
        self.centre = np.asarray(WORKSPACE[:2], float)

    def __call__(self, pp):
        d = pp.d
        if pp.label not in RECORDED_STAGES or self.n > int(d.time * self.fps):
            return
        self.n += 1
        self.r.update_scene(d, camera="laptop_cam")
        full = self.r.render().copy()
        td = topdown(full, self.K3, self.T_cam2world, self.centre, self.td_side, self.td_n)
        img = cv2.resize(full, self.wh, interpolation=cv2.INTER_AREA)
        extra = {WRIST_KEY: wrist_frame(self.r, d, self.info, self.wh)} if self.info.get("wrist") else {}
        act = clean_action(pp, self.arm)
        self.ds.add_frame({**extra,
                           "action": act.astype(np.float32),
                           "observation.state": robot_state(d, self.arm),
                           "observation.images.laptop": img,
                           "observation.images.topdown": td,
                           "observation.cam_K": self.K, "observation.cam_T": self.T,
                           "task": TASK})


def open_state_lerobot(root, repo_id, fps):
    """A state-only dataset: action, joints and the privileged scene state."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names7 = [f"arm_joint{i}" for i in range(1, 7)] + ["gripper"]
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": names7},
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        ENV_KEY: {"dtype": "float32", "shape": (len(ENV_NAMES),), "names": ENV_NAMES},
    }
    if os.path.exists(root):
        return LeRobotDataset(repo_id, root=root)
    return LeRobotDataset.create(repo_id, fps=fps, features=features, root=root,
                                 robot_type="galaxea_a1x", use_videos=False)


def open_lerobot(root, repo_id, fps, W, H, td_n=TD_N, wrist=None):
    """Open or create the dataset. `W`, `H` is the stored size of the raw
    webcam stream and of the wrist stream; `td_n` the side of the square
    top-down map. `wrist` adds the wrist stream; None reads WRIST_CAMERA, the
    switch `pour_scene.build` reads, so the schema follows the scenes."""
    from pour_scene import WRIST_HAND
    wrist = WRIST_HAND if wrist is None else wrist
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names7 = [f"arm_joint{i}" for i in range(1, 7)] + ["gripper"]
    hwc = ["height", "width", "channels"]
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": names7},
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "observation.images.laptop": {"dtype": "video", "shape": (H, W, 3), "names": hwc},
        "observation.images.topdown": {"dtype": "video", "shape": (td_n, td_n, 3), "names": hwc},
        "observation.cam_K": {"dtype": "float32", "shape": (9,), "names": None},
        "observation.cam_T": {"dtype": "float32", "shape": (16,), "names": None},
    }
    if wrist:
        features[WRIST_KEY] = {"dtype": "video", "shape": (H, W, 3), "names": hwc}
    if os.path.exists(root):
        return LeRobotDataset(repo_id, root=root)
    return LeRobotDataset.create(repo_id, fps=fps, features=features, root=root,
                                 robot_type="galaxea_a1x", use_videos=True, image_writer_threads=4)


def episode(seed, gif=False, record=None, verbose=False, noise=0.0, ds=None, fps=20, state=False,
            vary=False, servo_noise=0.0):
    """`vary` draws the grasp among the feasible candidates; `servo_noise` is
    the standard deviation, in degrees, of the drift added to the executed
    joint commands (0 = the planner's own command is what the servos get).
    Both draw from streams seeded by the scene seed, so an episode is
    reproducible and a re-run of a seed range gives the same dataset."""
    model, data, info = build(seed)
    per = SimPourPerception(model, data, info, pos_noise=noise, rot_noise=noise,
                            rng=np.random.default_rng(seed))
    sn = ServoNoise(np.random.default_rng([seed, 2]), sigma=np.radians(servo_noise)) if servo_noise > 0 else None
    pp = Pour(model, data, per, info, verbose=verbose, rng=np.random.default_rng([seed, 1]),
              vary=vary, servo_noise=sn)
    rec = None
    if ds is not None and state:
        result = pp.run(on_step=StateRecorder(ds, model, info, fps, pp.arm))
        if result.success:
            ds.save_episode()
        else:
            ds.clear_episode_buffer()
    elif ds is not None:
        r = mujoco.Renderer(model, height=info["cam"]["H"], width=info["cam"]["W"])
        try:
            result = pp.run(on_step=LeRobotRecorder(ds, r, info, fps, pp.arm))
        finally:
            r.close()
        if result.success:
            ds.save_episode()
        else:
            ds.clear_episode_buffer()
    elif gif or record:
        r = mujoco.Renderer(model, height=info["cam"]["H"] // 2, width=info["cam"]["W"] // 2)
        rec = PourRecorder(r, cam="laptop_cam")
        try:
            result = pp.run(on_step=rec)
        finally:
            r.close()
    else:
        result = pp.run()
    return model, data, info, result, rec


def view(seed, noise=0.0):
    import time
    import mujoco.viewer
    while True:
        model, data, info = build(seed)
        per = SimPourPerception(model, data, info, pos_noise=noise, rot_noise=noise)
        pp = Pour(model, data, per, info, verbose=True)
        for _ in range(50):
            try:
                v = mujoco.viewer.launch_passive(model, data)
                break
            except RuntimeError:
                time.sleep(0.1)
        else:
            raise RuntimeError("viewer did not close")
        with v:
            v.cam.lookat[:] = (0.0, 0.1, 0.75); v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, 150, -20
            t0 = time.time()

            def on_step(pp):
                if not v.is_running():
                    raise SystemExit
                v.sync()
                lag = pp.d.time - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)

            print(f"seed {seed}", flush=True)
            res = pp.run(on_step=on_step)
            print(f"  -> {'SUCCESS' if res.success else 'FAIL at ' + res.stage}  {res.detail}", flush=True)
            end = time.time() + 2.0
            while v.is_running() and time.time() < end:
                mujoco.mj_step(model, data); v.sync(); time.sleep(model.opt.timestep)
            if not v.is_running():
                return
        seed += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", help="record the first episode from laptop_cam")
    ap.add_argument("--record", help="directory for one .npz per episode")
    ap.add_argument("--noise", type=float, default=0.0, help="perception noise sigma [m, rad]")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--view", action="store_true", help="watch episodes live (macOS: mjpython)")
    ap.add_argument("--lerobot", help="root directory of a LeRobot dataset to append successful episodes to")
    ap.add_argument("--repo-id", default="galaxeo/a1x_pour_sim")
    ap.add_argument("--fps", type=int, default=20, help="frame and state rate of the LeRobot dataset")
    ap.add_argument("--state", action="store_true",
                    help="with --lerobot: record the privileged scene state instead of images (no rendering)")
    ap.add_argument("--vary", action="store_true",
                    help="draw the grasp among the feasible candidates instead of taking the first")
    ap.add_argument("--servo-noise", type=float, default=0.0, metavar="DEG",
                    help="std of the smooth drift added to the executed joint commands; the label stays clean")
    args = ap.parse_args()
    if args.view:
        return view(args.seed, noise=args.noise)
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    ds = None
    if args.lerobot and args.state:
        ds = open_state_lerobot(args.lerobot, args.repo_id, args.fps)
    elif args.lerobot:
        ds = open_lerobot(args.lerobot, args.repo_id, args.fps, STORE_W, STORE_H, TD_N)
        print(f"lerobot dataset at {args.lerobot}: {ds.num_episodes} episode(s) so far")
    outcomes, n_ok = Counter(), 0
    for i in range(args.n):
        seed = args.seed + i
        want_gif = args.gif and i == 0
        model, data, info, res, rec = episode(seed, gif=bool(want_gif), record=args.record,
                                              verbose=args.verbose, noise=args.noise, ds=ds, fps=args.fps,
                                              state=args.state, vary=args.vary, servo_noise=args.servo_noise)
        n_ok += res.success
        outcomes[res.stage if not res.success else "success"] += 1
        b = info["bottle"]
        print(f"seed {seed:3d}  {'SUCCESS' if res.success else 'FAIL   '} {res.stage:13s} "
              f"[r={b['body_r'] * 1000:.0f} h={b['height'] * 1000:.0f} m={b['mass']:.2f}] "
              f"t={res.duration:5.1f}s  {res.detail}", flush=True)
        if want_gif:
            write_gif(rec.frames, args.gif)
        if args.record and rec is not None:
            rec.save(os.path.join(args.record, f"ep{seed:04d}.npz"), info, res)
    if ds is not None:
        ds.finalize()
        print(f"lerobot dataset: {ds.num_episodes} episode(s), {ds.num_frames} frames")
    print(f"\n{n_ok}/{args.n} success ({100 * n_ok / args.n:.0f}%)")
    for k, v in outcomes.most_common():
        print(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
