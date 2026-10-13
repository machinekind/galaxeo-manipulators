"""The wrist camera as the grasp policy sees it: its 160x120 frame and the believed pixel.

The policy was trained on MuJoCo renders of an ideal pinhole camera on `gripper_link`
(f = FOCAL_PX, principal point at the image centre, upright). The real camera delivers a
1920x1080 frame through a wide, strongly distorting lens, mounted upside down. `WristCamera`
turns that frame into the training geometry with one remap through a 640x480 intermediate,
with the 180 degree rotation folded into the map, then an area resize to 160x120. The
model card's centre crop is not used: it is 5 to 27 px off at 160x120.

`WristCamera.policy_image` takes either frame the twin delivers: its simulated wrist camera
renders training's pinhole at 160x120 already, the real one gives the raw 1920x1080 frame.

`project` is training's `believed_uv`: a base-frame point through the nominal camera pose
(`T_GRIPPER_CAMERA`) into the 160x120 image. It takes the pose of `gripper_link` in the base
frame; computing that pose from the joints is the arm's business (`galaxeo.arm`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .contract import (
    CX,
    CY,
    DOT_MAX_PX,
    FOCAL_PX,
    IMG_H,
    IMG_W,
    NEAR_Z,
    RAW_H,
    RAW_W,
    T_GRIPPER_CAMERA,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)

if TYPE_CHECKING:
    from .rig import Rig

logger = logging.getLogger(__name__)

#: The body `T_GRIPPER_CAMERA` is measured in.
HAND_BODY = "gripper_link"
#: The remap goes to four times the policy size, then an area resize brings it down. Remapping
#: straight to 160x120 would take one sample per roughly 7x7 raw pixels and alias.
MID_SCALE = 4
MID_W = MID_SCALE * IMG_W
MID_H = MID_SCALE * IMG_H
#: Share of the policy frame that may lie past the sensor's edge before the intrinsics are refused.
MAX_OUTSIDE = 0.005


@dataclass(frozen=True)
class WristIntrinsics:
    K: np.ndarray                      # (3, 3), refers to the UPRIGHT 1920x1080 image
    dist: np.ndarray                   # (5,) k1 k2 p1 p2 k3, same frame
    width: int = RAW_W
    height: int = RAW_H
    upside_down: bool = True           # the device delivers the frame rotated by 180 degrees


@dataclass(frozen=True)
class _RemapTables:
    """`cv2.remap`'s fixed-point pair from `cv2.convertMaps`."""

    xy: np.ndarray                     # (MID_H, MID_W, 2) int16, integer source pixel
    fraction: np.ndarray               # (MID_H, MID_W) uint16, interpolation table index


class CameraError(RuntimeError):
    pass


def _inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def project(T_hand2base: np.ndarray, p_base: np.ndarray, *, T_cam2hand: np.ndarray = T_GRIPPER_CAMERA,
            f: float = FOCAL_PX, cx: float = CX, cy: float = CY) -> np.ndarray:
    """Pixel (u, v) of a base-frame point, float32; NaN pair when the point is not in front of the camera."""
    T_cam2base = np.asarray(T_hand2base, float) @ np.asarray(T_cam2hand, float)
    p_cam = (_inverse(T_cam2base) @ np.append(np.asarray(p_base, float), 1.0))[:3]
    depth = p_cam[2]
    if not depth > NEAR_Z:
        return np.array([np.nan, np.nan], np.float32)
    u = f * p_cam[0] / depth + cx
    v = f * p_cam[1] / depth + cy
    return np.array([u, v], np.float32)


def _default_intrinsics() -> WristIntrinsics:
    return WristIntrinsics(WRIST_K_2026_09_26.copy(), WRIST_DIST_2026_09_26.copy())


def wrist_intrinsics(rig: Rig) -> WristIntrinsics:
    pair = rig.intrinsics()
    return WristIntrinsics(np.asarray(pair.K, float), np.asarray(pair.dist, float).ravel(),
                           upside_down=rig.upside_down())


class WristCamera:
    """The wrist camera's geometry: camera frame to policy frame, and base-frame points to pixels.

    Raises `CameraError` when the intrinsics would show more than `MAX_OUTSIDE` of the policy
    frame from outside the sensor (less is logged as a warning), or show it through a lens model
    that folds.
    """

    def __init__(self, intr: WristIntrinsics | None = None, *,
                 T_cam2hand: np.ndarray = T_GRIPPER_CAMERA) -> None:
        self.intr = intr if intr is not None else _default_intrinsics()
        self.T_cam2hand = np.asarray(T_cam2hand, float)
        self._tables = _raw_maps(self.intr)

    def policy_frame(self, raw_rgb: np.ndarray) -> np.ndarray:
        """Raw (1080, 1920, 3) uint8 RGB as the device delivers it -> (120, 160, 3) uint8 RGB."""
        shape = (self.intr.height, self.intr.width, 3)
        if raw_rgb.shape != shape or raw_rgb.dtype != np.uint8:
            # Camera layers do not check the delivered size: a driver that ignores the request
            # delivers another resolution and nothing else complains.
            raise CameraError(f"wrist frame is {raw_rgb.shape} {raw_rgb.dtype}, expected {shape} uint8")
        mid = cv2.remap(raw_rgb, self._tables.xy, self._tables.fraction, cv2.INTER_LINEAR)
        return cv2.resize(mid, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    def policy_image(self, image: np.ndarray) -> np.ndarray:
        """One wrist camera frame from the twin -> (120, 160, 3) uint8 RGB for the policy.

        A 160x120 frame is the twin's simulated camera, which renders training's pinhole
        (`T_GRIPPER_CAMERA`, `FOVY_DEG`) at the policy size: it is used as it is. The real
        camera's raw frame is re-projected by `policy_frame`. Any other frame raises `CameraError`.
        """
        img = np.asarray(image)
        raw_shape = (self.intr.height, self.intr.width, 3)
        if img.dtype != np.uint8:
            raise CameraError(f"wrist frame is {img.dtype}, expected uint8 RGB")
        if img.shape == (IMG_H, IMG_W, 3):
            return img
        if img.shape == raw_shape:
            return self.policy_frame(img)
        raise CameraError(f"wrist frame is {img.shape}: expected {(IMG_H, IMG_W, 3)} from the simulated "
                          f"camera or {raw_shape} from the real one")

    def believed_uv(self, T_hand2base: np.ndarray, believed: np.ndarray) -> np.ndarray:
        """Pixel of the believed grasp point; `T_hand2base` is the pose of `HAND_BODY` in the base frame."""
        return project(T_hand2base, believed, T_cam2hand=self.T_cam2hand)

    def cube_top_uv(self, T_hand2base: np.ndarray, centre_xy: np.ndarray, top_z: float) -> np.ndarray:
        """Pixel of the centre of an object's top face."""
        p = np.array([float(centre_xy[0]), float(centre_xy[1]), float(top_z)])
        return project(T_hand2base, p, T_cam2hand=self.T_cam2hand)


def _raw_maps(intr: WristIntrinsics) -> _RemapTables:
    f = MID_SCALE * FOCAL_PX
    K_mid = np.array([[f, 0.0, MID_SCALE * (CX + 0.5) - 0.5],
                      [0.0, f, MID_SCALE * (CY + 0.5) - 0.5],
                      [0.0, 0.0, 1.0]])
    K = np.asarray(intr.K, float)
    dist = np.asarray(intr.dist, float).ravel()
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, K_mid, (MID_W, MID_H), cv2.CV_32FC1)
    _check_map(mx, my, intr)

    if intr.upside_down:
        # K and dist describe the upright image; the rotation is folded into the map
        # instead of undistorting the raw frame with unrotated intrinsics.
        mx = (intr.width - 1) - mx
        my = (intr.height - 1) - my

    xy, fraction = cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)
    return _RemapTables(xy=xy, fraction=fraction)


def _check_map(mx: np.ndarray, my: np.ndarray, intr: WristIntrinsics) -> None:
    # A K and dist from the rig file are not checked anywhere else, and cv2.remap complains about
    # neither fault: a view past the sensor turns into a black band the policy never saw, and a
    # lens model that folds (a poor fit with a large k2 or k3) samples the wrong raw pixels.
    if not (np.isfinite(mx).all() and np.isfinite(my).all()):
        raise CameraError("wrist camera K and dist give no pixel for part of the policy frame")

    edges = {"left": mx < 0, "right": mx > intr.width - 1, "top": my < 0, "bottom": my > intr.height - 1}
    outside = edges["left"] | edges["right"] | edges["top"] | edges["bottom"]
    if outside.any():
        n = int(outside.sum())
        where = ", ".join(k for k, m in edges.items() if m.any())
        msg = (f"wrist camera K and dist: {n} of {MID_W}x{MID_H} px ({100.0 * outside.mean():.2f} %) "
               f"of the policy frame lie outside the {intr.width}x{intr.height} sensor "
               f"({where} edge of the upright image)")
        # A recalibration can shift the view by a few pixels past one edge; a thin black strip
        # harms the policy less than a run that cannot start.
        if outside.mean() > MAX_OUTSIDE:
            raise CameraError(msg)
        logger.warning("%s; they stay black", msg)

    if not ((np.diff(mx, axis=1) > 0).all() and (np.diff(my, axis=0) > 0).all()):
        raise CameraError("wrist camera K and dist: the lens model folds inside the policy frame")


_DOT = (0, 255, 0)
_EXTRA = (255, 0, 255)
_TEXT = (255, 255, 255)


def draw_dump(image: np.ndarray, uv: np.ndarray, *, scale: int = 4, radius_px: float = DOT_MAX_PX,
              extra: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """The policy frame upscaled by `scale` (nearest), with the believed pixel and its dot limit drawn.

    A cross marks `uv`, a circle of `radius_px` (policy pixels) around it shows the dot check's
    limit. `extra` points (for example the projected cube top) are small labelled squares.
    """
    img = np.asarray(image, np.uint8)
    out = np.ascontiguousarray(np.repeat(np.repeat(img, scale, axis=0), scale, axis=1))
    height = img.shape[0]
    width = img.shape[1]
    far = 8 * scale * max(height, width)               # keeps a pixel near the camera plane drawable
    notes: list[str] = []

    def at(p: np.ndarray) -> tuple[int, int] | None:
        u = float(p[0])
        v = float(p[1])
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        x = round((u + 0.5) * scale - 0.5)
        y = round((v + 0.5) * scale - 0.5)
        return min(max(x, -far), far), min(max(y, -far), far)

    c = at(uv)
    if c is None:
        notes.append("no pixel")
    else:
        if not (-0.5 <= uv[0] <= width - 0.5 and -0.5 <= uv[1] <= height - 0.5):
            notes.append(f"off frame at ({float(uv[0]):.1f}, {float(uv[1]):.1f})")
        arm = 3 * scale
        cv2.line(out, (c[0] - arm, c[1]), (c[0] + arm, c[1]), _DOT, 1, cv2.LINE_AA)
        cv2.line(out, (c[0], c[1] - arm), (c[0], c[1] + arm), _DOT, 1, cv2.LINE_AA)
        cv2.circle(out, c, round(radius_px * scale), _DOT, 1, cv2.LINE_AA)

    for name, p in (extra or {}).items():
        e = at(p)
        if e is None:
            notes.append(f"{name}: no pixel")
            continue
        cv2.rectangle(out, (e[0] - scale, e[1] - scale), (e[0] + scale, e[1] + scale), _EXTRA, 1, cv2.LINE_AA)
        cv2.putText(out, name, (e[0] + 2 * scale, e[1] - 2 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _EXTRA, 1,
                    cv2.LINE_AA)

    for k, text in enumerate(notes):
        cv2.putText(out, text, (6, 16 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1, cv2.LINE_AA)
    return out
