#!/usr/bin/env python3
"""Run seeded pick-and-place episodes headless and report the success rate.

    sim/.venv/bin/python sim/planner/run_episodes.py --n 20 --seed 0
    sim/.venv/bin/python sim/planner/run_episodes.py --n 1 --gif pickplace.gif
    sim/.venv/bin/python sim/planner/run_episodes.py --n 20 --record /tmp/pp -v
    sim/.venv/bin/mjpython sim/planner/run_episodes.py --view --seed 4    # live viewer, loops seeds

Each episode gets a freshly compiled scene (`pickplace_scene.build(seed)`),
prints one line with its outcome and, on failure, the stage that failed.
`--record DIR` writes one .npz per episode holding time / qpos / ctrl / TCP
pose at 50 Hz and `table_cam` frames at 10 Hz, ready to be folded into a
LeRobot dataset by `ros2_ws/to_lerobot.py`-style tooling.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from pickplace_scene import build  # noqa: E402
from planner.perception import SimPerception  # noqa: E402
from planner.pickplace import PickPlace  # noqa: E402

STATE_HZ, FRAME_HZ = 50, 10
FRAME_W, FRAME_H = 480, 320


class Recorder:
    """Samples state at 50 Hz and, if a renderer is given, frames at 10 Hz."""

    def __init__(self, renderer=None, frame_hz=FRAME_HZ):
        self.r, self.frame_hz = renderer, frame_hz
        self.t, self.qpos, self.ctrl, self.tcp, self.stage = [], [], [], [], []
        self.ft, self.frames = [], []

    def __call__(self, pp):
        d = pp.d
        if len(self.t) <= int(d.time * STATE_HZ):
            pos, R = pp.arm.tcp_pose(d)
            self.t.append(d.time)
            self.qpos.append(d.qpos.copy())
            self.ctrl.append(d.ctrl.copy())
            self.tcp.append(np.concatenate([pos, R.ravel()]))
            self.stage.append(pp.label)
        if self.r is not None and len(self.frames) <= int(d.time * self.frame_hz):
            self.r.update_scene(d, camera="table_cam")
            self.frames.append(self.r.render())
            self.ft.append(d.time)

    def save(self, path, info, result):
        np.savez_compressed(
            path, time=np.array(self.t), qpos=np.array(self.qpos), ctrl=np.array(self.ctrl),
            tcp=np.array(self.tcp), stage=np.array(self.stage),
            frames=np.array(self.frames, dtype=np.uint8), frame_time=np.array(self.ft),
            success=result.success, outcome=result.stage, target=result.target,
            seed=info["seed"], dog_tag=info["dog"]["tag"],
            object_names=np.array([o["name"] for o in info["objects"]]))


def write_gif(frames, path, fps=FRAME_HZ):
    from PIL import Image
    Image.fromarray(frames[0]).save(path, save_all=True, optimize=True, loop=0,
                                    duration=int(1000 / fps),
                                    append_images=[Image.fromarray(f) for f in frames[1:]])
    print(f"wrote {path} ({len(frames)} frames)")


def episode(seed, sway=True, gif=False, record=None, verbose=False, noise=0.0):
    model, data, info = build(seed, sway=sway)
    per = SimPerception(model, data, pos_noise=noise, rot_noise=noise,
                        rng=np.random.default_rng(seed))
    pp = PickPlace(model, data, per, info, verbose=verbose)
    rec = None
    if gif or record:
        r = mujoco.Renderer(model, height=FRAME_H, width=FRAME_W)
        rec = Recorder(r)
        try:
            result = pp.run(on_step=rec)
        finally:
            r.close()
    else:
        result = pp.run()
    return model, data, info, result, rec


def view(seed, sway=True, noise=0.0):
    """Loop episodes in the interactive viewer at roughly real time, one new
    seed after another, until the window is closed. Each seed is a freshly
    compiled model, so each gets its own viewer window; on macOS the next one
    can only open once the previous has actually gone, hence the retry."""
    import time
    import mujoco.viewer
    while True:
        model, data, info = build(seed, sway=sway)
        per = SimPerception(model, data, pos_noise=noise, rot_noise=noise)
        pp = PickPlace(model, data, per, info, verbose=True)
        for _ in range(50):                             # wait for the old window to close
            try:
                v = mujoco.viewer.launch_passive(model, data)
                break
            except RuntimeError:
                time.sleep(0.1)
        else:
            raise RuntimeError("viewer did not close")
        with v:
            v.cam.lookat[:] = (0.0, -0.25, 0.6); v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.9, 150, -20
            t0 = time.time()

            def on_step(pp):
                if not v.is_running():
                    print(f"viewer window closed during {pp.label} at t={pp.d.time:.1f}s", flush=True)
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
    ap.add_argument("--gif", help="record the first episode from table_cam")
    ap.add_argument("--no-sway", action="store_true", help="freeze the dog")
    ap.add_argument("--record", help="directory for one .npz per episode")
    ap.add_argument("--noise", type=float, default=0.0, help="perception noise sigma [m, rad]")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--view", action="store_true", help="watch episodes live, looping seeds (macOS: mjpython)")
    args = ap.parse_args()
    if args.view:
        return view(args.seed, sway=not args.no_sway, noise=args.noise)

    if args.record:
        os.makedirs(args.record, exist_ok=True)
    outcomes, n_ok = Counter(), 0
    for i in range(args.n):
        seed = args.seed + i
        want_gif = args.gif and i == 0
        model, data, info, res, rec = episode(
            seed, sway=not args.no_sway, gif=bool(want_gif),
            record=args.record, verbose=args.verbose, noise=args.noise)
        n_ok += res.success
        outcomes[res.stage if not res.success else "success"] += 1
        kinds = "/".join(o["kind"][:3] for o in info["objects"])
        print(f"seed {seed:3d}  {'SUCCESS' if res.success else 'FAIL   '} "
              f"{res.stage:13s} obj={res.target or '-':5s} [{kinds}] "
              f"t={res.duration:5.1f}s  {res.detail}")
        if want_gif:
            write_gif(rec.frames, args.gif)
        if args.record and rec is not None:
            rec.save(os.path.join(args.record, f"ep{seed:04d}.npz"), info, res)

    print(f"\n{n_ok}/{args.n} success ({100 * n_ok / args.n:.0f}%)")
    for k, v in outcomes.most_common():
        print(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
