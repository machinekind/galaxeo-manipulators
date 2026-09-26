#!/usr/bin/env python3
"""Stereo of the two environment cameras → point cloud → GraspGenX.

    # offline: two session JSONs (from calibrate_real.py) and two frames
    python grasp_from_stereo.py --left calib_session/env0.json --right calib_session/env1.json \\
        --left-image L.png --right-image R.png --save cloud.npy --no-infer

    # live capture, then ZMQ (GraspGenX server already running on the GPU machine)
    python grasp_from_stereo.py --left calib_session/env0.json --right calib_session/env1.json \\
        --left-cam 0 --right-cam 1 --host 192.168.x.x --port 5556

The wrist camera is not used here. Cloud is in the arm base. Returned grasps
are printed in that frame and as A1X tool poses (x = approach, y = close).
Transmit is never on: this script does not move the arm.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _open_cam(index, width, height):
    import cv2
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"camera {index} did not open")
    for _ in range(8):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"camera {index} read failed")
    return frame


def _read_image(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not read {path}")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="left env camera session JSON")
    ap.add_argument("--right", required=True, help="right env camera session JSON")
    ap.add_argument("--left-image")
    ap.add_argument("--right-image")
    ap.add_argument("--left-cam", type=int, default=-1,
                    help="OpenCV index for the left env camera (live)")
    ap.add_argument("--right-cam", type=int, default=-1)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-depth", type=float, default=0.20)
    ap.add_argument("--max-depth", type=float, default=1.80)
    ap.add_argument("--no-crop", action="store_true",
                    help="keep points outside the table workspace box")
    ap.add_argument("--save", help="write the (N,3) cloud as .npy")
    ap.add_argument("--no-infer", action="store_true")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--gripper", default="galaxea_g1")
    ap.add_argument("--sweep-volume-json",
                    help="params-only infer_object instead of named gripper")
    ap.add_argument("--planner", default="diffusion")
    ap.add_argument("--num-grasps", type=int, default=200)
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()

    from galaxeo.vision import (
        GraspGenXClient, SweepVolumeParams, grasp_to_tool, load_session,
        reconstruct, workspace_mask,
    )

    left_cam = load_session(a.left)
    right_cam = load_session(a.right)
    if a.left_image and a.right_image:
        L, R = _read_image(a.left_image), _read_image(a.right_image)
    elif a.left_cam >= 0 and a.right_cam >= 0:
        L = _open_cam(a.left_cam, a.width, a.height)
        R = _open_cam(a.right_cam, a.width, a.height)
    else:
        sys.exit("need --left-image/--right-image or --left-cam/--right-cam")
    if L.shape[:2] != R.shape[:2]:
        sys.exit(f"image sizes differ: {L.shape[:2]} vs {R.shape[:2]}")

    xyz = reconstruct(L, R, left_cam, right_cam, min_depth=a.min_depth,
                      max_depth=a.max_depth, stride=a.stride)
    if not a.no_crop:
        xyz = xyz[workspace_mask(xyz)]
    print(f"cloud: {len(xyz)} points in base")
    if len(xyz) == 0:
        sys.exit("empty cloud (check calibration, overlap, lighting)")
    if a.save:
        np.save(a.save, xyz)
        print("wrote", a.save)
    if a.no_infer:
        return 0

    with GraspGenXClient(host=a.host, port=a.port) as client:
        print("server:", client.health())
        if a.sweep_volume_json:
            with open(a.sweep_volume_json) as f:
                params = SweepVolumeParams.from_dict(json.load(f))
            grasps, scores = client.infer_object(
                xyz, params, planner=a.planner, num_grasps=a.num_grasps,
                topk_num_grasps=a.topk)
        else:
            grasps, scores = client.infer(
                xyz, gripper_name=a.gripper, num_grasps=a.num_grasps,
                topk_num_grasps=a.topk)
    tools = grasp_to_tool(grasps)
    print(f"{len(grasps)} grasps (gripper={a.gripper})")
    for i, (G, T, s) in enumerate(zip(grasps, tools, scores)):
        print(f"  [{i}] score={s:.3f}  grasp_xyz={np.round(G[:3, 3], 3)}  "
              f"tool_xyz={np.round(T[:3, 3], 3)}  "
              f"tool_approach={np.round(T[:3, 0], 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
