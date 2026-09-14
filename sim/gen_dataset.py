#!/usr/bin/env python3
"""Generate a LeRobot pour dataset in parallel and merge the parts.

    sim/.venv/bin/python sim/gen_dataset.py --root data/pour_sim --episodes 300

Each worker is a separate `sim/planner/run_pour.py --lerobot ROOT/part_k`
process on its own slice of seeds, writing its own dataset root; the parts are
then merged into `ROOT/merged` with `lerobot-edit-dataset`.

Separate processes rather than threads or an in-process import: MuJoCo stepping
and the video encoder are both CPU bound and share one GIL, `run_pour` already
owns the whole open / record / drop-failures / finalize dance for one root, and
a dataset root is not safe for two writers. The cost is one MuJoCo compile per
process, which is nothing against an episode.

`run_pour` keeps only successful episodes, so `--episodes` is a target of
successes and the seeds are over-provisioned by `OVER` against the planner's
~85 % success rate. The run reports what it actually got rather than topping
up, because a finalized LeRobot root does not want to be reopened and appended
to.

This script itself runs under either venv; the workers and the merge always run
under `sim/.venv-lerobot`, which is the one with `lerobot` in it.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEROBOT_PY = os.path.join(HERE, ".venv-lerobot", "bin", "python")
LEROBOT_EDIT = os.path.join(HERE, ".venv-lerobot", "bin", "lerobot-edit-dataset")
RUN_POUR = os.path.join(HERE, "planner", "run_pour.py")
OVER = 1.25                       # seeds drawn per wanted episode (~85 % of seeds succeed)
OUTCOME = re.compile(r"^seed\s+(\d+)\s+(SUCCESS|FAIL)")
KEEP = re.compile(r"^lerobot dataset|success \(")   # the rest of a worker's chatter goes to its log


def default_workers():
    return max(1, min(8, (os.cpu_count() or 4) - 2))


def _pump(tag, proc, state, lock):
    """Count one worker's outcomes and echo them; everything else goes to its
    log file, which is dumped only if the worker exits non-zero. lerobot's
    torchcodec/ffmpeg/SVT banners are several screens per worker otherwise."""
    with open(state["log"], "w") as log:
        for line in proc.stdout:
            log.write(line)
            line = line.rstrip()
            m = OUTCOME.match(line)
            if m:
                state["done"] += 1
                state["ok"] += m.group(2) == "SUCCESS"
            if m or KEEP.search(line):
                with lock:
                    print(f"[{tag}] {state['ok']:3d}/{state['done']:3d} ok   {line}", flush=True)


def run_workers(root, repo_id, n_workers, n_each, seed0, fps):
    """Start one run_pour per worker on a disjoint seed range; wait for all."""
    lock = threading.Lock()
    workers = []
    for k in range(n_workers):
        part = os.path.join(root, f"part_{k}")
        seed = seed0 + k * n_each
        cmd = [LEROBOT_PY, RUN_POUR, "--lerobot", part, "--repo-id", f"{repo_id}_part{k}",
               "--seed", str(seed), "--n", str(n_each), "--fps", str(fps)]
        print(f"[w{k}] seeds {seed}..{seed + n_each - 1} -> {part}", flush=True)
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        st = {"done": 0, "ok": 0, "part": part, "seed": seed,
              "log": os.path.join(root, f"part_{k}.log")}
        t = threading.Thread(target=_pump, args=(f"w{k}", p, st, lock), daemon=True)
        t.start()
        workers.append((p, t, st))
    for p, t, st in workers:
        t.join()
        st["rc"] = p.wait()
    return [st for _, _, st in workers]


def episodes_in(part):
    """(episodes, frames) recorded in a LeRobot root, or (0, 0) if it is empty."""
    meta = os.path.join(part, "meta", "info.json")
    if not os.path.exists(meta):
        return 0, 0
    with open(meta) as f:
        info = json.load(f)
    return int(info.get("total_episodes", 0)), int(info.get("total_frames", 0))


def merge(parts, repo_id, out):
    """Merge the non-empty parts into `out` with lerobot-edit-dataset."""
    if os.path.exists(out):
        shutil.rmtree(out)
    ids = [f"{repo_id}_part{k}" for k, _ in parts]
    roots = [p for _, p in parts]
    cmd = [LEROBOT_EDIT, "--operation.type", "merge",
           "--operation.repo_ids", "[" + ", ".join(f"'{i}'" for i in ids) + "]",
           "--operation.roots", "[" + ", ".join(f"'{r}'" for r in roots) + "]",
           "--new_repo_id", f"{repo_id}_merged", "--new_root", out]
    print("merging:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def du(path):
    total = 0
    for dirpath, _, names in os.walk(path):
        for n in names:
            fp = os.path.join(dirpath, n)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="directory for part_k/ and merged/")
    ap.add_argument("--episodes", type=int, default=100, help="target successful episodes")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed0", type=int, default=0, help="first scene seed")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--repo-id", default="galaxeo/a1x_pour_sim")
    ap.add_argument("--keep-parts", action="store_true", help="do not delete part_k/ after merging")
    args = ap.parse_args()

    for exe in (LEROBOT_PY, LEROBOT_EDIT):
        if not os.path.exists(exe):
            sys.exit(f"missing {exe}: create sim/.venv-lerobot (see POUR.md)")
    n_workers = max(1, args.workers)
    n_each = max(1, math.ceil(args.episodes * OVER / n_workers))
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    print(f"{args.episodes} episode(s) wanted: {n_workers} worker(s) x {n_each} seed(s) "
          f"from {args.seed0}, {args.fps} fps, into {root}", flush=True)

    t0 = time.time()
    states = run_workers(root, args.repo_id, n_workers, n_each, args.seed0, args.fps)
    for k, st in enumerate(states):
        if st["rc"]:
            print(f"[w{k}] exited {st['rc']}; last of {st['log']}:", flush=True)
            with open(st["log"]) as f:
                print("".join(f.readlines()[-30:]), flush=True)

    parts, n_ep, n_fr = [], 0, 0
    for k, st in enumerate(states):
        ep, fr = episodes_in(st["part"])
        print(f"[w{k}] {ep} episode(s), {fr} frame(s) in {st['part']}", flush=True)
        if ep:
            parts.append((k, st["part"]))
            n_ep += ep
            n_fr += fr
    if not parts:
        sys.exit("no successful episodes: nothing to merge")

    out = os.path.join(root, "merged")
    if len(parts) == 1:
        if os.path.exists(out):
            shutil.rmtree(out)
        shutil.copytree(parts[0][1], out)
        print("only one non-empty part; copied it to", out, flush=True)
    else:
        merge(parts, args.repo_id, out)
    n_ep, n_fr = episodes_in(out)

    if not args.keep_parts:
        for _, p in parts:
            shutil.rmtree(p, ignore_errors=True)
    size = du(out)
    mins = (time.time() - t0) / 60
    print(f"\n{n_ep} episode(s), {n_fr} frame(s) in {out} "
          f"({size / 1e6:.1f} MB, {size / max(1, n_ep) / 1e6:.2f} MB/episode, "
          f"{n_fr / max(1, n_ep):.0f} frames/episode) in {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
