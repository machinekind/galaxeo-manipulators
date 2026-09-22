#!/usr/bin/env python3
"""Fold the per-process results of `eval_policy.py --json` into one table.

    python sim/planner/eval_summary.py sim/runs/<run>/eval 2000 1000 --n 20

Reads every `seeds_*.json` under the directory, prints the success count and
the failure-mode mix per evaluation range (the DAgger monitor's names, or
`timeout, carried` / `timeout, never touched` when nothing else went wrong
first), one line per episode, and writes the union to `all.json`.
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("seed0", type=int, nargs="+", help="first seed of each evaluation range")
    ap.add_argument("--n", type=int, default=20, help="episodes per range")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "seeds_*.json"))):
        with open(f) as fh:
            rows += json.load(fh)["episodes"]
    for s0 in args.seed0:
        eps = sorted((e for e in rows if s0 <= e["seed"] < s0 + args.n), key=lambda e: e["seed"])
        ok = sum(e["success"] for e in eps)
        mix = collections.Counter(
            "success" if e["success"]
            else (e.get("trouble") or ("timeout, carried" if e.get("carried") else "timeout, never touched"))
            for e in eps)
        print(f"seeds {s0}-{s0 + args.n - 1}: {ok}/{len(eps)} success  {dict(mix)}")
        for e in eps:
            print(f"  seed {e['seed']} {'SUCCESS' if e['success'] else 'FAIL'} poured {e['secs']:.1f}s "
                  f"trouble={e.get('trouble')} t={e.get('t_trouble')}")
    with open(os.path.join(args.dir, "all.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
