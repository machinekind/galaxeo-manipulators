#!/usr/bin/env python3
"""Automated per-session camera calibration: the arm waves, the camera watches.

    sim/.venv/bin/python sim/calib/calibrate.py --sim --seed 3          # score against ground truth
    sim/.venv/bin/python sim/calib/calibrate.py --sim --n 20 --seed 0   # 20 seeds, error table
    sim/.venv/bin/python sim/calib/calibrate.py --sim --seed 3 --out session.json

The laptop is put down wherever; nothing about its pose is known. The arm
visits a fan of poses over the table, a photo is taken at each, the AprilTags
on the gripper (and the one on the base plate) are detected, and the camera
pose in the arm base frame is fitted by reprojection (`handeye.solve`). The
result is written as JSON with the residual, and the session refuses to be
trusted above `--max-px`.

The robot sits behind `Robot`: `SimRobot` drives the MuJoCo scene through
its position servos and renders `laptop_cam`; the real arm gets the same
three methods on top of the CAN driver. Forward kinematics for both comes
from `a1x.xml` at the *measured* joint angles, so servo error is included.
"""
import argparse
import json
import os
import sys
import time
from typing import Protocol

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a1x_control import Arm, Script, rot                      # noqa: E402
from calib.handeye import Observation, pose_error, solve       # noqa: E402
from calib.tags import detect                                  # noqa: E402
from planner.motion import Motion                              # noqa: E402

ARM_XML = os.path.join(os.path.dirname(HERE), "a1x.xml")
MAX_PX = 1.5


class Robot(Protocol):
    def q(self) -> np.ndarray: ...                    # measured joint angles, 6
    def move(self, q, dur: float) -> None: ...        # ramp to q and settle
    def image(self) -> np.ndarray: ...                # RGB frame from the session camera


class FK:
    """Gripper body pose in the arm base frame, from a1x.xml alone."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(ARM_XML)
        self.d = mujoco.MjData(self.m)
        self.body = self.m.body("gripper_link").id
        self.qadr = np.array([self.m.joint(f"arm_joint{i}").qposadr[0] for i in range(1, 7)])

    def gripper2base(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.body].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.body]
        return T


class SimRobot:
    def __init__(self, model, data, info, cam="laptop_cam"):
        self.m, self.d, self.info = model, data, info
        self.arm = Arm(model, "arm/")
        self.script = Script(model, data, {"arm": "arm/"})
        self.motion = Motion(model, data, self.arm)
        self.cam = cam
        self.r = mujoco.Renderer(model, height=info["cam"]["H"], width=info["cam"]["W"])

    def q(self):
        return self.d.qpos[self.arm.qadr].copy()

    def move(self, q, dur=1.5):
        self.script.move_q("arm", q, dur, "calib").wait(0.4)
        while self.d.time < self.script.t - 1e-9:
            self.script.apply(self.d.time)
            mujoco.mj_step(self.m, self.d)

    def image(self):
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def wave_pose(motion, q_prev, home, base, table_top, rng):
    """One collision-free joint configuration with the gripper over the table,
    approach pitched down and yaw spread about the outward direction."""
    for _ in range(40):
        p = np.array([rng.uniform(-0.22, 0.22), rng.uniform(0.0, 0.28), table_top + rng.uniform(0.12, 0.34)])
        a = p[:2] - base[:2]
        yaw = np.arctan2(a[1], a[0]) + rng.uniform(-0.8, 0.8)
        pitch = rng.uniform(0.3, 1.2)
        approach = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        roll = rng.uniform(-0.9, 0.9)                     # shows a different cube face
        close = np.cos(roll) * np.array([-np.sin(yaw), np.cos(yaw), 0.0]) + np.sin(roll) * np.array([0, 0, 1.0])
        q, why = motion.solve(q_prev, p, rot(approach, close), 0.05, home=home, q_from=q_prev)
        if not why:
            return q
    return None


def collect(robot, fk, tag_mounts, home, base, table_top, min_obs=14, min_poses=4, max_poses=40,
            seed=0, verbose=False):
    """Visit poses until enough tag observations are in hand, or give up.

    The camera pose is unknown, so nothing is aimed: poses are drawn at
    random over the table, each with a wrist roll so the tag cube shows a
    different face, and the loop stops once `min_obs` corners sets have been
    seen from at least four poses."""
    rng = np.random.default_rng(seed)
    obs, seen_ids, poses_with_tags, q_prev = [], set(), 0, np.asarray(home, float)
    for i in range(max_poses):
        q = wave_pose(robot.motion, q_prev, home, base, table_top, rng)
        if q is None:
            continue
        robot.move(q)
        q_prev = q
        found = detect(robot.image())
        T_g2b = fk.gripper2base(robot.q())
        n_before = len(obs)
        for tid, corners in found.items():
            if tid not in tag_mounts:
                continue
            body, T_tag2frame = tag_mounts[tid]
            obs.append(Observation(T_g2b if body == "gripper" else np.eye(4), T_tag2frame, corners, tid))
            seen_ids.add(tid)
        poses_with_tags += len(obs) > n_before
        if verbose:
            print(f"  pose {i:2d}: tags {sorted(found)}  ({len(obs)} obs)")
        if len(obs) >= min_obs and poses_with_tags >= min_poses:
            break
    return obs, seen_ids, i + 1


def tag_mounts_from_info(info):
    """{id: ('gripper'|'base', T_tag2frame)} from the scene's tag table."""
    out = {}
    for tid, t in info["tags"].items():
        T = np.eye(4); T[:3, :3], T[:3, 3] = t["R"], t["pos"]
        out[tid] = ("gripper" if t["body"] == "arm/gripper_link" else "base", T)
    return out


def session(robot, K, dist, tag_size, tag_mounts, home, base, table_top, fk=None, refine_tags=(),
            seed=0, max_poses=40, verbose=False):
    fk = fk or FK()
    # Refining the mounts couples them with the camera pose; only a spread of
    # gripper orientations separates the two, so ask for twice the views.
    min_obs, min_poses = (28, 8) if refine_tags else (14, 4)
    obs, seen, n_poses = collect(robot, fk, tag_mounts, home, base, table_top, min_obs=min_obs,
                                 min_poses=min_poses, seed=seed, max_poses=max_poses, verbose=verbose)
    if len(obs) < 6:
        raise RuntimeError(f"only {len(obs)} tag observations in {n_poses} poses; cannot calibrate")
    fit = solve(obs, K, dist, tag_size, refine_tags=refine_tags)
    fit.K, fit.dist = np.asarray(K), dist
    return fit, obs, seen, n_poses


def to_json(fit, K, dist, extra=None):
    d = dict(T_cam2base=fit.T_cam2base.tolist(), K=np.asarray(K).tolist(),
             dist=np.asarray(dist if dist is not None else np.zeros(5)).tolist(),
             rms_px=fit.rms_px, max_px=fit.max_px, n_obs=fit.n_obs, trusted=fit.rms_px <= MAX_PX,
             tag_mounts={str(k): v.tolist() for k, v in fit.tag_mounts.items()},
             time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    d.update(extra or {})
    return d


def run_sim(seed, n_poses, verbose, perturb=0.0, refine=False):
    from pour_scene import build
    model, data, info = build(seed)
    robot = SimRobot(model, data, info)
    try:
        mounts = tag_mounts_from_info(info)
        if perturb:                              # a ruler-measured mount is a few mm off
            rng = np.random.default_rng(seed + 1000)
            for tid, (body, T) in list(mounts.items()):
                if body == "gripper":
                    T = T.copy(); T[:3, 3] += rng.normal(0, perturb, 3)
                    mounts[tid] = (body, T)
        cam = info["cam"]
        refine_tags = [t for t, (b, _) in mounts.items() if b == "gripper"] if refine else ()
        fit, obs, seen, n = session(robot, cam["K"], None, info["tag_size"], mounts, info["q_home"],
                                    info["arm_base"], info["table_top"], refine_tags=refine_tags,
                                    seed=seed, max_poses=n_poses, verbose=verbose)
    finally:
        robot.close()
    dt, dr = pose_error(cam["T_cam2base"], fit.T_cam2base)
    return fit, dt, dr, seen, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="calibrate against pour_scene ground truth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="number of seeds (sim)")
    ap.add_argument("--poses", type=int, default=40, help="most poses to visit")
    ap.add_argument("--perturb", type=float, default=0.0, help="sigma [m] added to nominal gripper tag mounts")
    ap.add_argument("--refine", action="store_true", help="also solve the gripper tag mounts")
    ap.add_argument("--out", help="write the session JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not a.sim:
        sys.exit("only --sim is implemented; the real-arm Robot needs the CAN driver")
    errs = []
    for i in range(a.n):
        seed = a.seed + i
        fit, dt, dr, seen, n_poses = run_sim(seed, a.poses, a.verbose, a.perturb, a.refine)
        errs.append((dt, dr))
        flag = "" if fit.rms_px <= MAX_PX else "  UNTRUSTED"
        print(f"seed {seed:3d}  poses={n_poses:2d} obs={fit.n_obs:3d} tags={sorted(seen)} rms={fit.rms_px:.2f}px "
              f"max={fit.max_px:.2f}px  err={dt * 1000:.1f}mm {np.degrees(dr):.2f}deg{flag}")
        if a.out and a.n == 1:
            with open(a.out, "w") as f:
                json.dump(to_json(fit, fit.K, fit.dist, dict(seed=seed)), f, indent=1)
            print("wrote", a.out)
    if a.n > 1:
        e = np.array(errs)
        print(f"\ntranslation error: median {np.median(e[:, 0]) * 1000:.1f}mm  max {e[:, 0].max() * 1000:.1f}mm")
        print(f"rotation error:    median {np.degrees(np.median(e[:, 1])):.2f}deg  max {np.degrees(e[:, 1].max()):.2f}deg")


if __name__ == "__main__":
    main()
