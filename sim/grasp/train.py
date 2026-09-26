#!/usr/bin/env python3
"""Train the state teacher for the last-centimetres grasp with PPO.

    sim/.venv-lerobot/bin/python sim/grasp/train.py --name grasp_v1 --envs 8 --steps 2_000_000
    sim/.venv-lerobot/bin/python sim/grasp/train.py --name smoke --envs 4 --steps 20000 --rollout 256

Vectorised over processes (one MuJoCo scene pool per process). Writes
`sim/runs/<name>/policy.pt` every update and `log.csv` with the success rate
of the episodes finished in each rollout. Resume with `--init <policy.pt>`.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gymnasium as gym                                            # noqa: E402
import torch                                                       # noqa: E402

from grasp.env import ACT_DIM, OBS_DIM, GraspEnv                   # noqa: E402
from grasp.ppo import Policy, gae, ppo_update                       # noqa: E402


def make_env(seed, kw):
    def thunk():
        return GraspEnv(seed=seed, **kw)
    return thunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="grasp_v1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2_000_000, help="total environment steps")
    ap.add_argument("--rollout", type=int, default=512, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--hover", type=float, default=0.06)
    ap.add_argument("--pool", type=int, default=8, help="scenes per process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="policy.pt to start from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "runs"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    kw = dict(pool=a.pool, hover=a.hover)
    envs = gym.vector.AsyncVectorEnv([make_env(a.seed * 100 + i, kw) for i in range(a.envs)],
                                     shared_memory=False, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    policy = Policy.load(a.init, a.device) if a.init else Policy(OBS_DIM, ACT_DIM, device=a.device)
    opt = None
    N, T = a.envs, a.rollout
    obs, _ = envs.reset(seed=a.seed)
    log = open(os.path.join(out, "log.csv"), "a", newline="")
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(["update", "steps", "success", "episodes", "attempts", "ep_return", "ep_len",
                         "pi_loss", "v_loss", "entropy", "kl", "fps"])
    total, update = 0, 0
    ep_ret, ep_len = np.zeros(N), np.zeros(N, int)
    t_start = time.time()
    while total < a.steps:
        B_obs = np.zeros((T, N, OBS_DIM), np.float32); B_act = np.zeros((T, N, ACT_DIM), np.float32)
        B_logp = np.zeros((T, N), np.float32); B_val = np.zeros((T, N), np.float32)
        B_rew = np.zeros((T, N), np.float32); B_done = np.zeros((T, N), np.float32)
        finished = []
        t0 = time.time()
        for t in range(T):
            policy.norm.update(obs)
            act, logp, val = policy.act(obs)
            nobs, rew, term, trunc, info = envs.step(np.clip(act, -1, 1))
            B_obs[t], B_act[t], B_logp[t], B_val[t], B_rew[t] = obs, act, logp, val, rew
            done = np.logical_or(term, trunc)
            # truncation is not a terminal state: bootstrap it with the value of the final obs
            if trunc.any() and "final_obs" in info:
                for i in np.where(trunc & ~term)[0]:
                    fo = info["final_obs"][i]
                    if fo is not None:
                        _, _, v_last = policy.act(fo[None])
                        B_rew[t, i] += a.gamma * float(v_last[0])
            B_done[t] = done.astype(np.float32)
            ep_ret += rew; ep_len += 1
            if "outcome" in info:
                for i in np.where(done)[0]:
                    finished.append((bool(info["success"][i]), int(info["attempts"][i]), ep_ret[i], ep_len[i]))
                    ep_ret[i] = 0.0; ep_len[i] = 0
            obs = nobs
        _, _, last_val = policy.act(obs)
        adv, ret = gae(B_rew, B_val, B_done, last_val, a.gamma, a.lam)
        stats, opt = ppo_update(policy, B_obs.reshape(-1, OBS_DIM), B_act.reshape(-1, ACT_DIM),
                                B_logp.ravel(), adv.ravel(), ret.ravel(), epochs=a.epochs,
                                minibatch=a.minibatch, lr=a.lr, ent_coef=a.ent, opt=opt)
        total += T * N; update += 1
        fps = T * N / (time.time() - t0)
        if finished:
            succ = np.mean([f[0] for f in finished]); att = np.mean([f[1] for f in finished])
            r_mean = np.mean([f[2] for f in finished]); l_mean = np.mean([f[3] for f in finished])
        else:
            succ = att = r_mean = l_mean = float("nan")
        writer.writerow([update, total, f"{succ:.3f}", len(finished), f"{att:.2f}", f"{r_mean:.2f}",
                         f"{l_mean:.1f}", f"{stats['pi_loss']:.4f}", f"{stats['v_loss']:.4f}",
                         f"{stats['entropy']:.3f}", f"{stats['kl']:.4f}", f"{fps:.0f}"])
        log.flush()
        policy.save(os.path.join(out, "policy.pt"))
        print(f"upd {update:4d} steps {total:9d} success {succ:5.2f} ({len(finished):3d} ep) "
              f"attempts {att:4.2f} return {r_mean:7.2f} len {l_mean:5.1f} ent {stats['entropy']:5.2f} "
              f"kl {stats['kl']:.4f} {fps:5.0f} fps {(time.time() - t_start) / 60:5.1f} min", flush=True)
    envs.close()


if __name__ == "__main__":
    main()
