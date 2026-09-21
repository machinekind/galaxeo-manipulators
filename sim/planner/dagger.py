#!/usr/bin/env python3
"""Takeover DAgger: roll the policy out, stop it going wrong, rewind, and let
the scripted planner finish the episode from there.

    sim/.venv-lerobot/bin/python sim/planner/dagger.py --ckpt sim/runs/pour_act_td2 \
        --n 12 --seed 5000                                   # report only, nothing saved
    sim/.venv-lerobot/bin/python sim/planner/dagger.py --ckpt sim/runs/pour_act_td2 \
        --n 12 --seed 5000 --lerobot sim/datasets/dagger/part_0

The planner is an open-loop timed script, so relabelling the policy's own
frames with what the teacher would have commanded is meaningless for a chunked
policy: the first action of a fresh smoothstep ramp is "stay where you are".
What a simulator makes cheap instead is rewinding. Every `SNAP` seconds of a
rollout the whole physics state is snapshotted; a ground-truth monitor ends the
rollout at the first sign of trouble and names it; the state from a second or
two *before* that is restored and `Pour.takeover` finishes the task from it,
recording its own frames in the dataset's schema. What the policy gets to learn
is therefore the recovery, from states its own mistakes lead to.

A teacher segment is kept only if its pour verified: whole if the whole run
succeeded, cut at the end of `verify_pour` if only the putting-down failed.
That is not known until the teacher has finished, so its frames are buffered
and flushed at the end.

One line per seed. `SUCCESS` there means the seed yielded at least one
correction episode -- so a rollout the policy *poured* prints `FAIL`, which is
right for a data collector: there was nothing to correct. `ROOT/dagger_log.jsonl`
gets one JSON line per saved episode, because per-episode metadata cannot go
into the dataset's features without breaking the merge with the base set.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from a1x_control import Arm  # noqa: E402
from planner.eval_policy import (GLASS_MARGIN, MOUTH_ABOVE, Judge, apply_action,  # noqa: E402
                                 find_checkpoint, load_policy, observation)
from planner.perception import SimPourPerception  # noqa: E402
from planner.pour import POUR_MIN_SECS, Pour, held_by  # noqa: E402
from planner.run_pour import STORE_H, STORE_W, LeRobotRecorder, open_lerobot  # noqa: E402
from planner.topdown import N as TD_N  # noqa: E402
from pour_scene import build  # noqa: E402

SNAP = 0.5              # seconds of sim time between physics snapshots
MOVED = 0.010           # bottle or glass shifted this far in xy = knocked [m]
TILTED = 0.10           # ... or leaned this far while nobody is holding it [rad]
LIFTED = 0.02           # bottle base this far off the table counts as carried [m]
HELD_GAP = 0.25         # without both finger contacts this long = no longer held [s]
MISS_TILT = 1.0         # tipped past this with the mouth outside the rim ...
MISS_SECS = 1.5         # ... for this long, cumulative, is pouring on the table [s]
EVAL_SEEDS = range(2000, 2020)          # held out for evaluation; never collect from these
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def snapshot(model, data):
    buf = np.zeros(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, buf, STATE)
    return buf


def restore(model, data, buf):
    """mjSTATE_INTEGRATION carries time, qpos, qvel, act, ctrl and the warm
    start, so the scene comes back with the policy's last command still on the
    servos -- which is what the takeover script ramps away from."""
    mujoco.mj_setState(model, data, buf, STATE)
    mujoco.mj_forward(model, data)


class Monitor:
    """Ground truth trouble: the first thing that makes the rest of a rollout
    pointless, and what to call it. Everything here is a fact about the world,
    not about the policy's intent, so it applies to any policy."""

    def __init__(self, model, data, info):
        self.m, self.d, self.info = model, data, info
        self.bottle = model.body(info["bottle"]["name"]).id
        self.glass = model.body(info["glass"]["name"]).id
        self.rim = model.site(info["glass"]["name"] + "/rim").id
        self.name = info["bottle"]["name"]
        self.height, self.r, self.top = info["bottle"]["height"], info["glass"]["r"], info["table_top"]
        self.b0 = data.xpos[self.bottle][:2].copy()
        self.g0 = data.xpos[self.glass][:2].copy()
        self.carried, self.miss, self.free = False, 0.0, np.inf

    def tick(self, dt):
        pos = self.d.xpos[self.bottle]
        R = self.d.xmat[self.bottle].reshape(3, 3)
        tilt = float(np.arccos(np.clip(R[2, 2], -1, 1)))
        g_pos, g_R = self.d.xpos[self.glass], self.d.xmat[self.glass].reshape(3, 3)
        if np.linalg.norm(g_pos[:2] - self.g0) > MOVED or g_R[2, 2] < 0.95:
            return "glass"
        # a bottle rolling in the fingers loses one of its two contacts for a
        # tick now and then, so it only counts as let go after `HELD_GAP`
        self.free = 0.0 if held_by(self.m, self.d, self.name) else self.free + dt
        if self.free >= HELD_GAP:
            if self.carried:
                return "dropped"
            if np.linalg.norm(pos[:2] - self.b0) > MOVED or tilt > TILTED:
                return "disturbed"
            return None
        self.carried = self.carried or pos[2] - self.top > LIFTED
        mouth = pos + R[:, 2] * self.height
        rim = self.d.site_xpos[self.rim]
        inside = (np.linalg.norm(mouth[:2] - rim[:2]) < self.r - GLASS_MARGIN
                  and 0.0 < mouth[2] - rim[2] < MOUTH_ABOVE)
        if tilt > MISS_TILT and not inside:
            self.miss += dt
            if self.miss > MISS_SECS:
                return "miss"
        return None


def rollout(policy, pre, post, cfg, model, data, info, arm, renderer, fps, max_secs, torch):
    """(snapshots, trouble, t). `trouble` is None when the policy poured."""
    policy.reset()
    judge, mon = Judge(model, data, info), Monitor(model, data, info)
    n_sub = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    dt = n_sub * model.opt.timestep
    snaps = []
    while data.time < max_secs:
        if not snaps or data.time >= snaps[-1][0] + SNAP - 1e-9:
            snaps.append((float(data.time), snapshot(model, data)))
        obs = observation(cfg, renderer, data, arm, info, torch)
        with torch.no_grad():
            action = post(policy.select_action(pre(obs)))
        apply_action(model, data, arm, np.asarray(action, dtype=np.float64).reshape(-1))
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        judge.tick(dt)
        if judge.secs >= POUR_MIN_SECS and judge.result()["success"]:
            return snaps, None, float(data.time)
        trouble = mon.tick(dt)
        if trouble:
            return snaps, trouble, float(data.time)
    return snaps, "timeout", float(data.time)


def rewind_points(trouble, t_fail, snaps, rewinds, rng):
    """[(snapshot index, rewind)] to hand the teacher, nearest snapshot first.

    A timeout never went obviously wrong, so there is no moment to measure back
    from: the first takeover is a uniformly random snapshot of the rollout and
    the schedule steps back from there."""
    ref = t_fail
    if trouble == "timeout" and snaps:
        ref = snaps[int(rng.integers(len(snaps)))][0] + rewinds[0]
    out = []
    for dl in rewinds:
        j = min(range(len(snaps)), key=lambda i: abs(snaps[i][0] - (ref - dl)))
        if j not in [k for k, _ in out]:
            out.append((j, dl))
    return out


class Buffer:
    """A LeRobotDataset stand-in for the teacher's frames. They are only worth
    keeping if its pour verifies, which is not known until it has finished."""

    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


def teach(model, data, info, renderer, fps, seed):
    """Let the planner finish from the restored state, buffering its frames.

    Returns (result, entry case, frames, cut), where `cut` is how many frames
    were in the buffer when `verify_pour` passed, or None if it never did."""
    per = SimPourPerception(model, data, info, rng=np.random.default_rng(seed))
    pp = Pour(model, data, per, info)
    buf = Buffer()
    rec = LeRobotRecorder(buf, renderer, info, fps, pp.arm)
    rec.n = int(data.time * fps) + 1            # carry the dataset's pacing over the seam
    cut = []
    pp.on_stage = lambda _pp, name, ok: (
        cut.append(len(buf.frames)) if name == "verify_pour" and ok and not cut else None)
    res = pp.takeover(on_step=rec)
    return res, getattr(pp, "case", "?"), buf.frames, (cut[0] if cut else None)


def collect(seed, policy, pre, post, cfg, args, torch, ds, log):
    """One scene: roll the policy out, then walk the rewind schedule until
    `--takeovers` usable teacher segments are in hand. Returns a report dict."""
    model, data, info = build(seed)
    mujoco.mj_forward(model, data)
    arm = Arm(model, "arm/")
    rep = dict(seed=seed, trouble=None, t_fail=0.0, takes=[])
    r = mujoco.Renderer(model, height=info["cam"]["H"], width=info["cam"]["W"])
    try:
        snaps, trouble, t_fail = rollout(policy, pre, post, cfg, model, data, info, arm, r,
                                         args.fps, args.max_secs, torch)
        rep["trouble"], rep["t_fail"] = trouble, t_fail
        if trouble is None:
            return rep
        rng = np.random.default_rng(seed)
        for j, dl in rewind_points(trouble, t_fail, snaps, args.rewind, rng):
            restore(model, data, snaps[j][1])
            res, case, frames, cut = teach(model, data, info, r, args.fps, seed)
            keep = frames if res.success else (frames[:cut] if cut is not None else [])
            take = dict(seed=seed, trouble=trouble, t_fail=round(t_fail, 2),
                        t_takeover=round(snaps[j][0], 2), rewind=dl, case=case,
                        stage=res.stage, detail=res.detail, frames=len(keep),
                        truncated=bool(keep and not res.success))
            rep["takes"].append(take)
            if not keep:
                continue
            if ds is not None:
                for f in keep:
                    ds.add_frame(f)
                ds.save_episode()
                log.write(json.dumps(take) + "\n")
                log.flush()
            if sum(t["frames"] > 0 for t in rep["takes"]) >= args.takeovers:
                break
    finally:
        r.close()
    return rep


def line(rep):
    """One seed, one line, starting the way run_pour's does."""
    saved = [t for t in rep["takes"] if t["frames"]]
    head = (f"seed {rep['seed']:4d}  {'SUCCESS' if saved else 'FAIL   '} "
            f"policy {rep['trouble'] or 'poured':9s} t={rep['t_fail']:5.1f}s")
    bits = [f"-{t['rewind']:.1f}s@{t['t_takeover']:.1f} {t['case']:4s} {t['stage']}"
            f" {t['frames']}f{'*' if t['truncated'] else ''}" for t in rep["takes"]]
    if bits:
        return head + "  | " + " | ".join(bits)
    return head + ("  nothing to correct" if rep["trouble"] is None else "  no usable takeover")


def report(reps, ds):
    trouble = Counter(r["trouble"] or "poured" for r in reps)
    takes = [t for r in reps for t in r["takes"]]
    usable = [t for t in takes if t["frames"]]
    by_case, by_rewind, stages = Counter(), Counter(), Counter()
    for t in takes:
        by_case[(t["case"], bool(t["frames"]))] += 1
        by_rewind[(t["rewind"], bool(t["frames"]))] += 1
        if not t["frames"]:
            stages[t["stage"]] += 1
    print(f"\n{len(reps)} seed(s): " + ", ".join(f"{k} {v}" for k, v in trouble.most_common()))
    pct = 100 * len(usable) / max(1, len(takes))
    print(f"takeovers: {len(usable)}/{len(takes)} usable ({pct:.0f}%)"
          + (f", {sum(t['truncated'] for t in usable)} truncated at verify_pour" if usable else ""))
    for label, counter in (("case", by_case), ("rewind", by_rewind)):
        keys = sorted({k for k, _ in counter})
        if keys:
            print(f"  by {label}: " + "   ".join(
                f"{k}: {counter[(k, True)]}/{counter[(k, True)] + counter[(k, False)]}" for k in keys))
    if stages:
        print("  teacher stopped at: " + ", ".join(f"{k} {v}" for k, v in stages.most_common()))
    frames = sum(t["frames"] for t in usable)
    where = f" in {ds.root}" if ds is not None else " (nothing saved: no --lerobot)"
    print(f"saved {len(usable)} episode(s), {frames} frames{where}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir, or a training run dir")
    ap.add_argument("--lerobot", help="root of a LeRobot dataset to append corrections to")
    ap.add_argument("--repo-id", default="galaxeo/a1x_pour_dagger")
    ap.add_argument("--seed", type=int, default=5000, help="first scene seed")
    ap.add_argument("--n", type=int, default=10, help="scenes to roll out")
    ap.add_argument("--fps", type=int, default=20, help="control rate; must match the dataset")
    ap.add_argument("--max-secs", type=float, default=45.0, help="rollout budget per scene")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--takeovers", type=int, default=1, help="usable segments to harvest per scene")
    ap.add_argument("--rewind", default="1.0,2.5,5.0", help="seconds before the trouble, in order")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.rewind = [float(x) for x in args.rewind.split(",") if x.strip()]
    seeds = range(args.seed, args.seed + args.n)
    if args.lerobot and set(seeds) & set(EVAL_SEEDS):
        sys.exit(f"seeds {EVAL_SEEDS.start}..{EVAL_SEEDS.stop - 1} are held out for evaluation")

    import torch
    torch.set_num_threads(2)                 # a worker per core, eight of these side by side

    ckpt = find_checkpoint(args.ckpt)
    policy, pre, post, cfg = load_policy(ckpt, args.device, None)
    print(f"{cfg.type} checkpoint {os.path.basename(os.path.dirname(ckpt))}, chunk {cfg.chunk_size}, "
          f"on {args.device}; rewind {args.rewind}, up to {args.takeovers} takeover(s) per seed")
    ds, log = None, None
    if args.lerobot:
        ds = open_lerobot(args.lerobot, args.repo_id, args.fps, STORE_W, STORE_H, TD_N)
        print(f"lerobot dataset at {args.lerobot}: {ds.num_episodes} episode(s) so far")
        log = open(os.path.join(args.lerobot, "dagger_log.jsonl"), "a")
    try:
        reps = []
        for seed in seeds:
            rep = collect(seed, policy, pre, post, cfg, args, torch, ds, log)
            reps.append(rep)
            print(line(rep), flush=True)
            if args.verbose:
                for t in rep["takes"]:
                    print(f"    {t['case']:4s} -{t['rewind']:.1f}s  {t['stage']:14s} {t['detail']}",
                          flush=True)
    finally:
        if log is not None:
            log.close()
    if ds is not None:
        ds.finalize()
        print(f"lerobot dataset: {ds.num_episodes} episode(s), {ds.num_frames} frames")
    report(reps, ds)


if __name__ == "__main__":
    main()
