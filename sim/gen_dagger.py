#!/usr/bin/env python3
"""Collect DAgger corrections in parallel and aggregate them with the base set.

    sim/.venv/bin/python sim/gen_dagger.py --ckpt sim/runs/pour_act_td2 \
        --root sim/datasets/pour_dagger --seeds 400

The sibling of `gen_dataset.py`: N `planner/dagger.py` processes on disjoint
scene seeds, each writing its own dataset root. Every worker loads the same
checkpoint and rolls it out, so this is much slower per seed than the planner
alone -- a policy forward pass per control tick, plus a whole teacher episode
per takeover.

Two outputs, because the corrections are worth looking at on their own:

    ROOT/dagger    the parts merged: DAgger episodes only
    ROOT/merged    `--base` and ROOT/dagger merged: what training reads

`--seeds` is a count of scenes to roll out, not a target of episodes: how many
of them yield a usable correction is the number the run reports. The scene
seeds must stay clear of the evaluation range, which is refused outright.
"""
import argparse
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_dataset import (LEROBOT_EDIT, LEROBOT_PY, default_workers, du,  # noqa: E402
                         episodes_in, merge, run_workers)
from planner.dagger import EVAL_SEEDS  # noqa: E402

DAGGER = os.path.join(HERE, "planner", "dagger.py")
LOG = "dagger_log.jsonl"


def collect_logs(parts, out):
    """One JSON line per saved episode, concatenated in worker order."""
    n = 0
    with open(out, "w") as f:
        for _, part in parts:
            path = os.path.join(part, LOG)
            if not os.path.exists(path):
                continue
            with open(path) as g:
                for line in g:
                    f.write(line)
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="the policy to correct")
    ap.add_argument("--root", required=True, help="directory for part_k/, dagger/ and merged/")
    ap.add_argument("--seeds", type=int, default=100, help="scenes to roll out")
    ap.add_argument("--seed0", type=int, default=5000, help="first scene seed")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--base", default="sim/datasets/pour_sim_td/merged",
                    help="dataset to aggregate the corrections with; '' to skip")
    ap.add_argument("--repo-id", default="galaxeo/a1x_pour_dagger")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--takeovers", type=int, default=1)
    ap.add_argument("--rewind", default="1.0,2.5,5.0")
    ap.add_argument("--max-secs", type=float, default=45.0)
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    for exe in (LEROBOT_PY, LEROBOT_EDIT):
        if not os.path.exists(exe):
            sys.exit(f"missing {exe}: create sim/.venv-lerobot (see POUR.md)")
    n_workers = max(1, args.workers)
    n_each = max(1, math.ceil(args.seeds / n_workers))
    seeds = range(args.seed0, args.seed0 + n_workers * n_each)
    if set(seeds) & set(EVAL_SEEDS):
        sys.exit(f"seeds {seeds.start}..{seeds.stop - 1} overlap the held-out evaluation range "
                 f"{EVAL_SEEDS.start}..{EVAL_SEEDS.stop - 1}")
    root = os.path.abspath(args.root)
    base = os.path.abspath(args.base) if args.base else None
    if base and not os.path.exists(os.path.join(base, "meta", "info.json")):
        sys.exit(f"no LeRobot dataset at {base}")
    os.makedirs(root, exist_ok=True)
    print(f"{n_workers} worker(s) x {n_each} seed(s) from {args.seed0}, {args.fps} fps, "
          f"into {root}", flush=True)

    t0 = time.time()
    extra = ["--ckpt", os.path.abspath(args.ckpt), "--takeovers", str(args.takeovers),
             "--rewind", args.rewind, "--max-secs", str(args.max_secs)]
    states = run_workers(root, args.repo_id, n_workers, n_each, args.seed0, args.fps,
                         script=DAGGER, extra=extra)
    for k, st in enumerate(states):
        if st["rc"]:
            print(f"[w{k}] exited {st['rc']}; last of {st['log']}:", flush=True)
            with open(st["log"]) as f:
                print("".join(f.readlines()[-30:]), flush=True)

    parts = []
    for k, st in enumerate(states):
        ep, fr = episodes_in(st["part"])
        print(f"[w{k}] {ep} episode(s), {fr} frame(s) in {st['part']}", flush=True)
        if ep:
            parts.append((k, st["part"]))
    if not parts:
        sys.exit("no usable corrections: nothing to merge")
    n_log = collect_logs(parts, os.path.join(root, LOG))

    dag = os.path.join(root, "dagger")
    if len(parts) == 1:
        if os.path.exists(dag):
            shutil.rmtree(dag)
        shutil.copytree(parts[0][1], dag)
        print("only one non-empty part; copied it to", dag, flush=True)
    else:
        merge(parts, args.repo_id, dag)
    n_ep, n_fr = episodes_in(dag)
    if not args.keep_parts:
        for _, p in parts:
            shutil.rmtree(p, ignore_errors=True)

    out = dag
    if base:
        out = os.path.join(root, "merged")
        merge([("base", base), ("dagger", dag)], args.repo_id, out,
              ids=[f"{args.repo_id}_base", f"{args.repo_id}_dagger"])
    n_all, n_all_fr = episodes_in(out)
    mins = (time.time() - t0) / 60
    print(f"\n{n_ep} correction(s), {n_fr} frame(s) from {len(seeds)} seed(s) in {dag} "
          f"({n_log} log line(s), {du(dag) / 1e6:.1f} MB) in {mins:.1f} min", flush=True)
    if base:
        print(f"{n_all} episode(s), {n_all_fr} frame(s) in {out} ({du(out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
