"""One calibrated pinhole camera in the arm base frame.

OpenCV convention: z forward, x right, y down. `T_cam2base` maps that frame
into the arm base (x forward over the table, z up), the same 4x4
`calibrate_real.py` writes as `T_cam2base`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4."""
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass(frozen=True)
class Camera:
    """Intrinsics and extrinsics of one RGB camera."""

    K: np.ndarray
    dist: np.ndarray
    T_cam2base: np.ndarray
    size: Optional[Tuple[int, int]] = None   # (width, height) when known

    def __post_init__(self) -> None:
        object.__setattr__(self, "K", np.asarray(self.K, float).reshape(3, 3))
        dist = np.asarray(self.dist, float).reshape(-1)
        if dist.size == 0:
            dist = np.zeros(5)
        object.__setattr__(self, "dist", dist)
        T = np.asarray(self.T_cam2base, float)
        if T.shape != (4, 4):
            raise ValueError(f"T_cam2base must be 4x4; got {T.shape}")
        object.__setattr__(self, "T_cam2base", T)

    @property
    def T_base2cam(self) -> np.ndarray:
        return invert(self.T_cam2base)

    def project(self, xyz_base: np.ndarray) -> np.ndarray:
        """Project base-frame points to pixels. xyz_base is (N, 3)."""
        import cv2
        xyz = np.asarray(xyz_base, float).reshape(-1, 3)
        R = self.T_base2cam[:3, :3]
        t = self.T_base2cam[:3, 3]
        uv, _ = cv2.projectPoints(xyz, cv2.Rodrigues(R)[0], t, self.K, self.dist)
        return uv.reshape(-1, 2)


def load_session(path: str) -> Camera:
    """Read a `calibrate_real.py` / `calib.calibrate.to_json` session file."""
    with open(path) as f:
        d = json.load(f)
    size = None
    if "width" in d and "height" in d:
        size = (int(d["width"]), int(d["height"]))
    return Camera(K=d["K"], dist=d.get("dist", [0, 0, 0, 0, 0]),
                  T_cam2base=d["T_cam2base"], size=size)
