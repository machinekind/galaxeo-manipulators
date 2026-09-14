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
    observation.state            (7,) measured joints + finger position
    observation.images.laptop    the webcam
    observation.cam_K, cam_T     (9,) (16,) the session's camera calibration, for provenance
    observation.environment_state (25,) the same two concatenated: the one key ACT
                                 reads as an environment state, so the policy is
                                 conditioned on the calibration it will have anyway

    sim/.venv-lerobot/bin/python sim/planner/run_pour.py --n 100 --lerobot data/pour_sim --fps 20
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from planner.perception import SimPourPerception  # noqa: E402
from planner.pour import Pour  # noqa: E402
from planner.run_episodes import Recorder, write_gif  # noqa: E402
from pour_scene import build  # noqa: E402


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


class LeRobotRecorder:
    """Streams one episode into an open LeRobotDataset at `fps`."""

    def __init__(self, ds, renderer, info, fps, arm):
        self.ds, self.r, self.info, self.fps, self.arm = ds, renderer, info, fps, arm
        self.n = 0
        self.K = np.asarray(info["cam"]["K"], np.float32).ravel()
        self.T = np.asarray(info["cam"]["T_cam2base"], np.float32).ravel()
        self.env = np.concatenate([self.K, self.T])

    def __call__(self, pp):
        d = pp.d
        if self.n > int(d.time * self.fps):
            return
        self.n += 1
        self.r.update_scene(d, camera="laptop_cam")
        q, g = d.qpos[self.arm.qadr], abs(float(d.qpos[self.arm.fadr[0]]))
        act = np.concatenate([d.ctrl[self.arm.acts], [np.clip(d.ctrl[self.arm.grip], 0.0, 0.05)]])
        self.ds.add_frame({"action": act.astype(np.float32),
                           "observation.state": np.concatenate([q, [g]]).astype(np.float32),
                           "observation.images.laptop": self.r.render().copy(),
                           "observation.cam_K": self.K, "observation.cam_T": self.T,
                           "observation.environment_state": self.env,
                           "task": TASK})


def open_lerobot(root, repo_id, fps, W, H):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names7 = [f"arm_joint{i}" for i in range(1, 7)] + ["gripper"]
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": names7},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names7},
        "observation.images.laptop": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        "observation.cam_K": {"dtype": "float32", "shape": (9,), "names": None},
        "observation.cam_T": {"dtype": "float32", "shape": (16,), "names": None},
        "observation.environment_state": {"dtype": "float32", "shape": (25,), "names": None},
    }
    if os.path.exists(root):
        return LeRobotDataset(repo_id, root=root)
    return LeRobotDataset.create(repo_id, fps=fps, features=features, root=root,
                                 robot_type="galaxea_a1x", use_videos=True, image_writer_threads=4)


def episode(seed, gif=False, record=None, verbose=False, noise=0.0, ds=None, fps=20):
    model, data, info = build(seed)
    per = SimPourPerception(model, data, info, pos_noise=noise, rot_noise=noise,
                            rng=np.random.default_rng(seed))
    pp = Pour(model, data, per, info, verbose=verbose)
    rec = None
    if ds is not None:
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
    args = ap.parse_args()
    if args.view:
        return view(args.seed, noise=args.noise)
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    ds = None
    if args.lerobot:
        from pour_scene import CAM
        ds = open_lerobot(args.lerobot, args.repo_id, args.fps, CAM["W"], CAM["H"])
        print(f"lerobot dataset at {args.lerobot}: {ds.num_episodes} episode(s) so far")
    outcomes, n_ok = Counter(), 0
    for i in range(args.n):
        seed = args.seed + i
        want_gif = args.gif and i == 0
        model, data, info, res, rec = episode(seed, gif=bool(want_gif), record=args.record,
                                              verbose=args.verbose, noise=args.noise, ds=ds, fps=args.fps)
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
