"""Eye-to-hand calibration by reprojection: camera fixed, tags on the arm.

    from calib.handeye import solve
    fit = solve(obs, K, dist, tag_size)
    fit.T_cam2base, fit.rms_px, fit.per_tag

Every observation is one tag seen in one image while the arm stood still:

    Observation(T_frame2base, T_tag2frame, corners)

`T_frame2base` is the pose of the body the tag is on (from forward
kinematics at the measured joint angles; the identity for a tag fixed to the
base), `T_tag2frame` the tag's nominal mount on that body, and `corners` the
four detected pixel corners. The solver finds the camera pose in the base
frame that best reprojects every corner, and can also refine the tag mounts,
which on the real arm are measured with a ruler.

This is deliberately not the closed-form AX = XB solvers: those need pairs of
motions and are sensitive to which pairs you feed them, while minimising
reprojection error uses every corner of every tag in every image and reports
its residual in pixels, which is the number that decides whether a session's
calibration is trusted.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .tags import corners_in_tag, project


@dataclass
class Observation:
    T_frame2base: np.ndarray
    T_tag2frame: np.ndarray
    corners: np.ndarray
    tag_id: int = -1


@dataclass
class Fit:
    T_cam2base: np.ndarray
    rms_px: float
    max_px: float
    n_obs: int
    tag_mounts: dict = field(default_factory=dict)    # tag id -> refined T_tag2frame
    residuals: np.ndarray = None


def _se3(xi):
    """Exponential map of a 6-vector (rotation vector, translation) to 4x4."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(xi[:3], float))
    T[:3, 3] = xi[3:]
    return T


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def _residuals(params, obs, K, dist, tag_size, refine_ids):
    """Stacked (2N,) pixel residuals for camera params and optional tag-mount deltas."""
    T_base2cam = _inv(_se3(params[:6]))
    deltas = {tid: _se3(params[6 + 6 * k:12 + 6 * k]) for k, tid in enumerate(refine_ids)}
    X = np.c_[corners_in_tag(tag_size), np.ones(4)]
    out = []
    for o in obs:
        mount = o.T_tag2frame @ deltas.get(o.tag_id, np.eye(4))
        P = (T_base2cam @ o.T_frame2base @ mount @ X.T).T[:, :3]
        out.append((project(P, K, dist) - o.corners).ravel())
    return np.concatenate(out)


def _initial(obs, K, dist, tag_size):
    """Median of the single-observation solutions T_cam2base = frame2base * tag2frame * cam2tag."""
    from .tags import tag_pose
    Ts = [o.T_frame2base @ o.T_tag2frame @ _inv(tag_pose(o.corners, tag_size, K, dist)) for o in obs]
    t = np.median([T[:3, 3] for T in Ts], axis=0)
    rv = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in Ts])
    R, _ = cv2.Rodrigues(np.median(rv, axis=0))
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    return T


def solve(obs, K, dist, tag_size, refine_tags=(), iters=50, init=None):
    """Levenberg-Marquardt on the reprojection error with a numeric Jacobian.

    `refine_tags` lists tag ids whose mount transform is also optimised (a
    tag fixed to the base cannot be refined, or the problem loses its scale).
    """
    obs = list(obs)
    refine_ids = [t for t in refine_tags if any(o.tag_id == t for o in obs)]
    T0 = _initial(obs, K, dist, tag_size) if init is None else init
    p = np.zeros(6 + 6 * len(refine_ids))
    p[:3] = cv2.Rodrigues(T0[:3, :3])[0].ravel(); p[3:6] = T0[:3, 3]
    lam = 1e-3
    r = _residuals(p, obs, K, dist, tag_size, refine_ids)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros_like(p); dp[k] = 1e-6
            J[:, k] = (_residuals(p + dp, obs, K, dist, tag_size, refine_ids) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            r_new = _residuals(p + step, obs, K, dist, tag_size, refine_ids)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                p, r, cost, lam, improved = p + step, r_new, c_new, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    per_corner = np.linalg.norm(r.reshape(-1, 2), axis=1)
    mounts = {tid: _se3(p[6 + 6 * k:12 + 6 * k]) for k, tid in enumerate(refine_ids)}
    return Fit(_se3(p[:6]), float(np.sqrt(np.mean(per_corner**2))), float(per_corner.max()),
               len(obs), mounts, per_corner)


def pose_error(T_a, T_b):
    """(translation error [m], rotation error [rad]) between two transforms."""
    dT = _inv(T_a) @ T_b
    ang = np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0])
    return float(np.linalg.norm(dT[:3, 3])), float(ang)
