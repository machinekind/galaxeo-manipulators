"""GraspGenX gripper poses ↔ A1X tool poses.

GraspGenX gripper base: +Z approach, +X closing.
A1X tool (`galaxeo.arm.Tool`): +X approach, +Y closing.

Both are right-handed, so the missing axis follows: GraspGenX +Y = A1X +Z.
"""

from __future__ import annotations

import numpy as np

# Columns are A1X tool axes expressed in the GraspGenX gripper frame:
# tool x (approach) = grasp z, tool y (close) = grasp x, tool z = grasp y.
GRASP_TO_TOOL = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=float)

_T_GRASP_TO_TOOL = np.eye(4)
_T_GRASP_TO_TOOL[:3, :3] = GRASP_TO_TOOL


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    R, t = T[:3, :3], T[:3, 3]
    return xyz @ R.T + t


def grasp_to_tool(T_grasp: np.ndarray) -> np.ndarray:
    """GraspGenX gripper pose in some frame → A1X tool pose in the same frame.

    `T_grasp` maps gripper-frame points to the cloud/base frame. The returned
    4x4 maps tool-frame points to that same frame, so it is what
    `Arm.plan` wants as a pose target (`plan(T)`).
    """
    T = np.asarray(T_grasp, float)
    if T.ndim == 3:
        out = np.tile(np.eye(4), (len(T), 1, 1))
        out[:] = T @ _T_GRASP_TO_TOOL
        return out
    if T.shape != (4, 4):
        raise ValueError(f"T_grasp must be (4, 4) or (K, 4, 4); got {T.shape}")
    return T @ _T_GRASP_TO_TOOL
