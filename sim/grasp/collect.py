#!/usr/bin/env python3
"""Collect wrist-image episodes labelled by the teacher, for the student (DAgger).

    # round 0: the teacher drives, its actions are the labels
    sim/.venv-lerobot/bin/python sim/grasp/collect.py --teacher sim/runs/grasp_v2/policy.pt \\
        --out sim/datasets/grasp_student/r0 --episodes 400 --seed 100000
    # round k: the student drives (mixed with the teacher by --beta), the teacher labels
    sim/.venv-lerobot/bin/python sim/grasp/collect.py --teacher sim/runs/grasp_v2/policy.pt \\
        --student sim/runs/grasp_student/r0/student.pt --beta 0.3 \\
        --out sim/datasets/grasp_student/r1 --episodes 400 --seed 200000

Several shards run in parallel as separate processes (`--seed` apart); each
writes one .npz per episode: wrist images (uint8, H x W x 3), the student
observation vector, the teacher's deterministic action at every tick, and
the episode's outcome. Rendering needs a GL context, so one process, one
environment.

The wrist camera is the measured lash-up (`--wrist`), jittered per scene by
1 cm and 2 deg; the scene's object colours are random already; `--vis`
randomises light, table colour and the object's texture-free shade per
episode so the student cannot lean on any of them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from grasp.env import GraspEnv                                      # noqa: E402

WRIST_SPEC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "hardware", "g1_camera_mounts",
                          "camera_spec_lashup.json")


def randomise_visuals(env, rng):
    """Per-episode light and colour jitter on the compiled model (cheap, in place)."""
    m = env.m
    for i in range(m.nlight):
        m.light_diffuse[i] = rng.uniform(0.35, 0.9, 3)
        m.light_ambient[i] = rng.uniform(0.1, 0.4, 3)
    try:
        t = m.geom("table").id
        m.geom_rgba[t, :3] = rng.uniform(0.25, 0.9, 3)
    except KeyError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="privileged policy.pt that labels every tick")
    ap.add_argument("--student", help="student.pt that drives (DAgger rounds)")
    ap.add_argument("--beta", type=float, default=0.0, help="share of ticks the teacher drives when a student is given")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--size", type=int, nargs=2, default=(128, 96), metavar=("W", "H"))
    ap.add_argument("--wrist", default=WRIST_SPEC)
    ap.add_argument("--composites", type=float, default=0.5)
    ap.add_argument("--disturb", type=float, default=0.3)
    ap.add_argument("--force", type=float, default=0.3)
    ap.add_argument("--vis", action="store_true", help="randomise lights and table colour per episode")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--jitter", type=float, nargs=3, default=(3.0, 0.7, 0.3), metavar=("MM", "DEG", "FOVY"),
                   help="per-scene wrist camera jitter. 10/2/1 (the calibration's uncertainty) caps a "
                        "single-frame student: it cannot tell a camera offset from an object offset.")
    a = ap.parse_args()

    from grasp.ppo import Policy
    teacher = Policy.load(a.teacher)
    student = None
    if a.student:
        from grasp.student import Student
        student = Student.load(a.student)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    env = GraspEnv(seed=a.seed, pool=a.pool, wrist=a.wrist, wrist_size=tuple(a.size),
                   composites=a.composites, disturb_p=a.disturb, force_p=a.force, wrist_jitter=tuple(a.jitter))
    t0 = time.time()
    n_ok = 0
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)
        if a.vis:
            randomise_visuals(env, rng)
        imgs, sobs, acts, done = [], [], [], False
        while not done:
            img = env.render_wrist()
            so = env.student_obs()
            label, _, _ = teacher.act(obs[None], deterministic=True)
            label = np.clip(label[0], -1, 1)
            if student is not None and rng.uniform() >= a.beta:
                drive = student.act(img[None], so[None])[0]
            else:
                drive = label
            imgs.append(img); sobs.append(so); acts.append(label.astype(np.float32))
            obs, r, term, trunc, info = env.step(drive)
            done = term or trunc
        n_ok += info["success"]
        np.savez_compressed(os.path.join(a.out, f"ep{a.seed + ep}.npz"),
                            img=np.stack(imgs), obs=np.stack(sobs), act=np.stack(acts),
                            success=bool(info["success"]), outcome=info["outcome"], kind=info["kind"],
                            attempts=info["attempts"], disturbed=info["disturbed"], forced=info["forced"])
        if (ep + 1) % 20 == 0:
            print(f"{ep + 1}/{a.episodes} episodes, success {n_ok / (ep + 1):.2f}, "
                  f"{(time.time() - t0) / (ep + 1):.2f} s/episode", flush=True)
    print(f"done: {a.episodes} episodes, success {n_ok / a.episodes:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
