#!/usr/bin/env python3
"""Closed-loop evaluation of a trained LeRobot policy in the pour scene.

    sim/.venv-lerobot/bin/python sim/planner/eval_policy.py --ckpt sim/runs/pour_act_0 --n 20
    sim/.venv-lerobot/bin/python sim/planner/eval_policy.py --ckpt sim/runs/pour_act_0 \
        --n 2 --gif pour_policy.gif

`--ckpt` takes either a `pretrained_model` directory or the run directory a
training job wrote, in which case the last checkpoint is used.

Each episode is a freshly compiled `pour_scene.build(seed)` -- the same
generator the demonstrations came from -- driven at `--fps` (which must match
the dataset, because that is the rate the actions were recorded at). Every tick
renders `laptop_cam`, hands the policy exactly the features its config lists as
inputs, and writes the returned 7-vector to the actuators: six joint targets
straight into `data.ctrl[arm.acts]`, and the gripper through the same mapping
the recorder inverted -- anything under 20 mm means "closed", which on this
model is a command 30 mm past the stop so the fingers squeeze with a set force.

Success is judged from ground truth, the way `planner.pour` judges the scripted
demonstrator: the bottle's mouth inside the glass rim and tilted past
`POUR_MIN_TILT`, held for `POUR_MIN_SECS`, with the glass still upright at the
end and the bottle still on the table. The one difference is that the seconds
are counted cumulatively over the ticks that qualify, rather than as the span
from the first qualifying sample to the last, so a policy that dithers in and
out of the rim has to add up real pouring time.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from a1x_control import Arm  # noqa: E402
from planner.pour import POUR_MIN_SECS, POUR_MIN_TILT  # noqa: E402
from planner.run_episodes import write_gif  # noqa: E402
from pour_scene import GRIP_CMD, build  # noqa: E402

GLASS_MARGIN = 0.004      # mouth must be this far inside the rim [m]
MOUTH_ABOVE = 0.08        # ... and no higher than this above it [m]
UPRIGHT = 0.95            # glass R[2,2] at the end
FLOOR_DROP = 0.05         # bottle centre this far below the table top = on the floor [m]
GRIP_CLOSED = 0.02        # commanded finger opening under this means "close hard" [m]


def find_checkpoint(path):
    """Accept a pretrained_model directory or a training run directory."""
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    for cand in (os.path.join(path, "pretrained_model"),
                 os.path.join(path, "checkpoints", "last", "pretrained_model")):
        if os.path.isfile(os.path.join(cand, "config.json")):
            return os.path.realpath(cand)
    ckpts = os.path.join(path, "checkpoints")
    if os.path.isdir(ckpts):
        steps = sorted(d for d in os.listdir(ckpts) if d.isdigit())
        if steps:
            cand = os.path.join(ckpts, steps[-1], "pretrained_model")
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    raise SystemExit(f"no lerobot checkpoint (a directory with config.json) under {path}")


def load_policy(ckpt, device):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(device)
    policy.eval()
    # Normalisation lives in the processor pipelines in lerobot 0.6, not in the
    # policy, so the checkpoint's own pipelines have to be loaded alongside it:
    # the preprocessor normalises and moves to the device, the postprocessor
    # unnormalises the action back into joint units.
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": {}}})
    return policy, pre, post, cfg


def observation(cfg, renderer, data, arm, info, torch):
    """Exactly the features the policy config lists as inputs, in dataset units.

    Values go in raw: the preprocessor pipeline normalises them. A batch
    dimension is added here for every key, because lerobot only adds one
    automatically for `observation.state` and the image keys."""
    obs = {}
    for key in cfg.input_features:
        if key.startswith("observation.images."):
            renderer.update_scene(data, camera="laptop_cam")
            img = renderer.render().copy()                       # HWC uint8
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            obs[key] = t.unsqueeze(0)
        elif key == "observation.state":
            q = data.qpos[arm.qadr]
            g = abs(float(data.qpos[arm.fadr[0]]))
            obs[key] = torch.from_numpy(
                np.concatenate([q, [g]]).astype(np.float32)).unsqueeze(0)
        elif key == "observation.cam_K":
            obs[key] = torch.from_numpy(
                np.asarray(info["cam"]["K"], np.float32).ravel()).unsqueeze(0)
        elif key == "observation.cam_T":
            obs[key] = torch.from_numpy(
                np.asarray(info["cam"]["T_cam2base"], np.float32).ravel()).unsqueeze(0)
        elif key == "observation.environment_state":
            # the session's camera calibration, intrinsics then extrinsics,
            # which is the one key ACT reads as an environment state
            obs[key] = torch.from_numpy(np.concatenate([
                np.asarray(info["cam"]["K"], np.float32).ravel(),
                np.asarray(info["cam"]["T_cam2base"], np.float32).ravel()])).unsqueeze(0)
        else:
            raise SystemExit(f"the policy wants {key}, which this scene cannot provide")
    return obs


def apply_action(model, data, arm, action):
    data.ctrl[arm.acts] = action[:6]
    grip = float(action[6])
    data.ctrl[arm.grip] = GRIP_CMD if grip < GRIP_CLOSED else float(np.clip(grip, 0.0, 0.05))


class Judge:
    """Ground-truth success, read the way `planner.pour` reads it."""

    def __init__(self, model, data, info):
        self.m, self.d, self.info = model, data, info
        self.bottle = model.body(info["bottle"]["name"]).id
        self.glass = model.body(info["glass"]["name"]).id
        self.rim = model.site(info["glass"]["name"] + "/rim").id
        self.height = info["bottle"]["height"]
        self.r = info["glass"]["r"]
        self.secs = 0.0
        self.max_tilt = 0.0
        self.glass_fell = False

    def _bottle(self):
        return self.d.xpos[self.bottle].copy(), self.d.xmat[self.bottle].reshape(3, 3).copy()

    def tick(self, dt):
        pos, R = self._bottle()
        mouth = pos + R[:, 2] * self.height
        rim = self.d.site_xpos[self.rim]
        tilt = float(np.arccos(np.clip(R[2, 2], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        d_xy = float(np.linalg.norm(mouth[:2] - rim[:2]))
        dz = float(mouth[2] - rim[2])
        inside = d_xy < self.r - GLASS_MARGIN and 0.0 < dz < MOUTH_ABOVE
        if inside and tilt > POUR_MIN_TILT:
            self.secs += dt
        if self.d.xmat[self.glass].reshape(3, 3)[2, 2] <= UPRIGHT:
            self.glass_fell = True

    def result(self):
        glass_up = bool(self.d.xmat[self.glass].reshape(3, 3)[2, 2] > UPRIGHT)
        on_floor = bool(self.d.xpos[self.bottle][2] < self.info["table_top"] - FLOOR_DROP)
        ok = self.secs >= POUR_MIN_SECS and glass_up and not on_floor
        why = " ".join(w for w, c in (("short-pour", self.secs >= POUR_MIN_SECS),
                                      ("glass-down", glass_up),
                                      ("bottle-on-floor", not on_floor)) if not c)
        return dict(success=ok, secs=self.secs, max_tilt=self.max_tilt,
                    glass_fell=self.glass_fell or not glass_up, on_floor=on_floor, why=why)


def episode(policy, pre, post, cfg, seed, fps, max_secs, torch, frames=None):
    policy.reset()
    model, data, info = build(seed)
    arm = Arm(model, "arm/")
    H, W = cfg.input_features["observation.images.laptop"].shape[1:]
    if (H, W) != (info["cam"]["H"], info["cam"]["W"]):
        print(f"  note: policy wants {W}x{H}, the scene camera is "
              f"{info['cam']['W']}x{info['cam']['H']}; rendering at the policy's size")
    judge = Judge(model, data, info)
    n_sub = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    dt = n_sub * model.opt.timestep
    r = mujoco.Renderer(model, height=H, width=W)
    try:
        while data.time < max_secs:
            obs = observation(cfg, r, data, arm, info, torch)
            with torch.no_grad():
                action = post(policy.select_action(pre(obs)))
            apply_action(model, data, arm, np.asarray(action, dtype=np.float64).reshape(-1))
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            judge.tick(dt)
            if frames is not None:
                r.update_scene(data, camera="laptop_cam")
                frames.append(r.render().copy())
    finally:
        r.close()
    out = judge.result()
    out["duration"] = float(data.time)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir, or a training run dir")
    ap.add_argument("--n", type=int, default=10, help="episodes")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--fps", type=int, default=20, help="control rate; must match the dataset")
    ap.add_argument("--max-secs", type=float, default=45.0, help="wall clock budget per episode")
    ap.add_argument("--gif", help="record the first episode from laptop_cam")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = ap.parse_args()

    import torch

    ckpt = find_checkpoint(args.ckpt)
    policy, pre, post, cfg = load_policy(ckpt, args.device)
    print(f"{cfg.type} checkpoint {os.path.basename(os.path.dirname(ckpt))}, "
          f"chunk {cfg.chunk_size}, n_action_steps {cfg.n_action_steps}, on {args.device}")
    print("inputs: " + ", ".join(cfg.input_features))

    n_ok = 0
    for i in range(args.n):
        seed = args.seed + i
        frames = [] if (args.gif and i == 0) else None
        res = episode(policy, pre, post, cfg, seed, args.fps, args.max_secs, torch, frames)
        n_ok += res["success"]
        print(f"seed {seed:3d}  {'SUCCESS' if res['success'] else 'FAIL   '} "
              f"poured {res['secs']:4.1f}s  max tilt {np.degrees(res['max_tilt']):5.1f}deg  "
              f"glass {'FELL' if res['glass_fell'] else 'ok  '}  t={res['duration']:5.1f}s"
              + (f"  {res['why']}" if res["why"] else ""), flush=True)
        if frames:
            write_gif(frames, args.gif, fps=args.fps)
    print(f"\n{n_ok}/{args.n} success ({100 * n_ok / args.n:.0f}%)")


if __name__ == "__main__":
    main()
