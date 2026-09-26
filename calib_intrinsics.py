#!/usr/bin/env python3
"""Camera intrinsics from a checkerboard, one-shot, per webcam.

    python3 calib_intrinsics.py --camera 0 --out calib_session/K.npz
    python3 calib_intrinsics.py --camera 0 --cols 9 --rows 6 --square 0.025
    python3 calib_intrinsics.py --camera 0 --rotate --out calib_session/wrist_K.npz   # wrist cam

Hold a printed checkerboard (default 9x6 inner corners, 25 mm squares) in
front of the camera; it collects `--frames` images at ~1 Hz while you tilt
the board, fits K and the distortion coefficients with cv2.calibrateCamera,
and refuses to write anything below `--min-frames` views or above `--max-err`
of reprojection error. A window shows the detected corners live; Ctrl-C (or
any key) stops collection early once `--frames` are in.

The output .npz carries K (3x3) and dist (5,), which is what
`calibrate_real.py --K` reads and what the session JSON stores next to the
extrinsics. Rerun whenever the camera's focus ring, zoom or resolution
changes: intrinsics and extrinsics are a pair, and the session gates on the
pixel residual of both.

`--rotate` turns every frame 180 deg before detection. The wrist board camera
is mounted upside down and every consumer (wrist_cam.py, calib_wrist.py, the
recorder) rotates its frames, so its K must be fitted on rotated frames too:
the principal point moves to (W - cx, H - cy) and the tangential terms flip.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import platform

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_session")
MIN_ERR_PX = 0.7           # reprojection error above this and the fit is not trusted
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def collect(cap, cols, rows, frames, window=True, rotate=False):
    """Collect frames with a full checkerboard visible. Returns the object
    points and image points of every good view."""
    pattern = (cols - 1, rows - 1)          # cv2 wants inner corners
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    last = 0.0
    grabbed = 0
    while True:
        ok, img = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if window:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, found)
            cv2.putText(vis, f"{grabbed}/{frames} views - tilt the board",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            try:
                cv2.imshow("calib_intrinsics", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')) and grabbed >= 5:
                    break
            except cv2.error:           # headless OpenCV build: carry on blind
                window = False
                print("no GUI in this OpenCV build; collecting without a window")
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if found and grabbed < frames and now - last > 1.0:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners)
            grabbed += 1
            last = now
            print(f"  view {grabbed:2d}/{frames}", flush=True)
        if grabbed >= frames:
            break
    return objpoints, imgpoints, grabbed


def fit(objpoints, imgpoints, size):
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None)
    return float(rms), np.asarray(K, float), np.asarray(dist, float).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV device index")
    ap.add_argument("--cols", type=int, default=9, help="checkerboard columns (inner corners + 1)")
    ap.add_argument("--rows", type=int, default=6, help="checkerboard rows (inner corners + 1)")
    ap.add_argument("--square", type=float, default=0.025, help="square side [m]")
    ap.add_argument("--frames", type=int, default=15, help="views to collect")
    ap.add_argument("--min-frames", type=int, default=8,
                    help="fewer good views than this and the fit is refused")
    ap.add_argument("--max-err", type=float, default=MIN_ERR_PX,
                    help="reprojection error above this and the fit is not written as trusted")
    ap.add_argument("--no-window", action="store_true", help="headless collection")
    ap.add_argument("--rotate", action="store_true",
                    help="rotate frames 180 deg first (the wrist camera is upside down)")
    ap.add_argument("--out", help="write the .npz here (default calib_session/K.npz)")
    a = ap.parse_args()

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        sys.exit(f"camera {a.camera} did not open")
    for _ in range(10):                       # let exposure settle
        cap.read()
    # The image size the fit uses must be the one the corners are detected in:
    # read it once, now, before collection (later setters drift).
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        objpoints, imgpoints, grabbed = collect(cap, a.cols, a.rows, a.frames,
                                                window=not a.no_window, rotate=a.rotate)
    finally:
        cap.release()
        if not a.no_window:
            cv2.destroyAllWindows()
    if grabbed < a.min_frames:
        sys.exit(f"only {grabbed} good views (wanted {a.min_frames}); "
                 "hold the board so every corner is visible and tilt it between views")
    rms, K, dist = fit(objpoints, imgpoints, (w, h))
    flag = "" if rms <= a.max_err else "  UNTRUSTED (> --max-err)"
    print(f"views={grabbed}  rms={rms:.3f}px{flag}")
    print("K =\n", np.round(K, 2))
    print("dist =", np.round(dist, 5))
    out = a.out or os.path.join(DEFAULT_OUT_DIR, "K.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, K=K, dist=dist, rms=rms, views=grabbed,
             cols=a.cols, rows=a.rows, square=a.square, rotate=bool(a.rotate),
             width=w, height=h, trusted=bool(rms <= a.max_err))
    print("wrote", out)
    if rms > a.max_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
