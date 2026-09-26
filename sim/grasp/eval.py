#!/usr/bin/env python3
"""Roll a grasp policy out on seeds and report what happened.

    sim/.venv-lerobot/bin/python sim/grasp/eval.py sim/runs/grasp_v1/policy.pt --n 20
    sim/.venv-lerobot/bin/python sim/grasp/eval.py sim/runs/grasp_v1/policy.pt --n 4 --video /tmp/grasp \\
        --wrist hardware/g1_camera_mounts/camera_spec_lashup.json
    sim/.venv-lerobot/bin/python sim/grasp/eval.py --random --n 20        # the no-policy floor

Prints one line per episode (outcome, attempts, ticks) and a summary.
`--video DIR` writes one mp4 per episode: the table camera, and the wrist
camera next to it when `--wrist` gives the camera spec.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from grasp.env import GraspEnv                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", nargs="?", help="policy.pt from train.py")
    ap.add_argument("--random", action="store_true", help="random actions instead of a policy")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10_000, help="first episode seed (training used 0..)")
    ap.add_argument("--hover", type=float, default=0.06)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--video", metavar="DIR")
    ap.add_argument("--wrist", metavar="SPEC_JSON", help="add the wrist camera from this spec (for --video)")
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    if not a.random and not a.policy:
        ap.error("give a policy.pt or --random")
    policy = None
    if not a.random:
        from grasp.ppo import Policy
        policy = Policy.load(a.policy)
    env = GraspEnv(seed=a.seed, pool=a.n, hover=a.hover, wrist=a.wrist)
    if a.video:
        import cv2
        os.makedirs(a.video, exist_ok=True)
    outcomes, attempts = Counter(), []
    for ep in range(a.n):
        obs, info = env.reset(seed=a.seed + ep)
        frames = []
        done = False
        while not done:
            if a.video:
                f = env.render()
                if a.wrist:
                    w = env.render_wrist()
                    w = cv2.resize(w, (int(w.shape[1] * f.shape[0] / w.shape[0]), f.shape[0]))
                    f = np.concatenate([f, w], axis=1)
                frames.append(f)
            if policy is None:
                act = env.action_space.sample()
            else:
                act, _, _ = policy.act(obs[None], deterministic=not a.stochastic)
                act = act[0]
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        outcomes[info["outcome"]] += 1; attempts.append(info["attempts"])
        print(f"seed {a.seed + ep}: {info['outcome']:8s} attempts {info['attempts']}  ticks {env.steps:3d}  "
              f"{info['kind']}", flush=True)
        if a.video and frames:
            path = os.path.join(a.video, f"ep{ep:02d}_{info['outcome']}.mp4")
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
    n = a.n
    print(f"success {outcomes['success']}/{n}  " + "  ".join(f"{k} {v}" for k, v in outcomes.items() if k != "success")
          + f"  mean attempts {np.mean(attempts):.2f}")
    env.close()


if __name__ == "__main__":
    main()
