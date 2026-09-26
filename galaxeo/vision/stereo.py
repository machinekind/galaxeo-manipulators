"""Rectified SGBM stereo of two independently calibrated environment cameras.

The cameras are not a factory stereo pair: each has its own K and a
`T_cam2base` from a card-wave session. Their relative pose is that pair of
extrinsics; SGBM runs after `stereoRectify`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .camera import Camera, invert
from .frames import transform_points

# Objects on the table, in the arm base. Slightly larger than the gripper
# reach box so a bottle sitting in front of the arm is kept.
WORKSPACE_LO = np.array([0.12, -0.35, 0.002])
WORKSPACE_HI = np.array([0.55, 0.35, 0.28])


def relative_pose(left: Camera, right: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """R, t for OpenCV ``stereoRectify`` / ``stereoCalibrate``.

    A point in the left camera: ``P_right = R @ P_left + t``.
    """
    T_right_from_left = invert(right.T_cam2base) @ left.T_cam2base
    return T_right_from_left[:3, :3].copy(), T_right_from_left[:3, 3].copy()


def workspace_mask(xyz_base: np.ndarray,
                   lo: np.ndarray = WORKSPACE_LO,
                   hi: np.ndarray = WORKSPACE_HI) -> np.ndarray:
    """Boolean (N,) of points inside an axis-aligned box in the base frame."""
    p = np.asarray(xyz_base, float).reshape(-1, 3)
    return np.all((p >= lo) & (p <= hi), axis=1)


def reconstruct(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left: Camera,
    right: Camera,
    *,
    min_depth: float = 0.20,
    max_depth: float = 1.80,
    stride: int = 2,
    num_disparities: Optional[int] = None,
    block_size: int = 5,
) -> np.ndarray:
    """BGR pair → (N, 3) float32 points in the arm base, metres.

    Invalid, too-near, too-far and non-finite points are dropped. `stride`
    keeps every Nth pixel after that (SGBM is dense).
    """
    import cv2

    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(
            f"left/right images must be the same size; got "
            f"{left_bgr.shape[:2]} vs {right_bgr.shape[:2]}"
        )
    h, w = left_bgr.shape[:2]
    R, t = relative_pose(left, right)
    baseline = float(np.linalg.norm(t))
    if baseline < 1e-4:
        raise ValueError(f"stereo baseline is {baseline:.4f} m; cameras look coincident")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left.K, left.dist, right.K, right.dist, (w, h), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1l, map2l = cv2.initUndistortRectifyMap(left.K, left.dist, R1, P1, (w, h), cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(right.K, right.dist, R2, P2, (w, h), cv2.CV_32FC1)
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY) if left_bgr.ndim == 3 else left_bgr
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY) if right_bgr.ndim == 3 else right_bgr
    rect_l = cv2.remap(gray_l, map1l, map2l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(gray_r, map1r, map2r, cv2.INTER_LINEAR)

    fx = float(abs(P1[0, 0]))
    if num_disparities is None:
        # Largest disparity SGBM should search: a point at min_depth.
        nd = int(np.ceil(fx * baseline / max(min_depth, 1e-3) / 16.0) * 16)
        nd = int(np.clip(nd, 64, 256))
    else:
        nd = int(num_disparities)
        if nd % 16:
            nd += 16 - nd % 16

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    valid = (disp > 0) & np.isfinite(pts).all(axis=2)
    z = pts[:, :, 2]
    valid &= (z > min_depth) & (z < max_depth)
    xyz_rect = pts[valid]
    if stride > 1:
        xyz_rect = xyz_rect[::stride]

    # reprojectImageTo3D is in the *rectified* left camera. R1 takes original
    # left → rectified, so original left = R1.T @ p_rect.
    xyz_left = xyz_rect @ R1
    T = np.eye(4)
    T[:3, :3] = left.T_cam2base[:3, :3]
    T[:3, 3] = left.T_cam2base[:3, 3]
    return transform_points(xyz_left, T).astype(np.float32)
