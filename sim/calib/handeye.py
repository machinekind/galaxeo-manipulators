"""Eye-to-hand calibration by reprojection: camera fixed, tags on the arm.

    from calib.handeye import Observation, solve
    fit = solve(obs, K, dist, tag_size, {"card": T_card2gripper_nominal})
    fit.T_cam2base, fit.mounts["card"], fit.rms_px

Every observation is one tag seen in one image while the arm stood still:

    Observation(T_frame2base, T_tag2mount, corners, tag_id, mount)

`T_frame2base` is the pose of the body the mount is bolted to (from forward
kinematics at the measured joint angles; the identity for a mount fixed to the
base), `T_tag2mount` the tag's pose *within* the mount, which is exact printed
geometry, and `corners` the four detected pixel corners. A "mount" is one rigid
carrier of tags -- today the calibration card, whose pose in the gripper is
whatever the hand that pinched it happened to do. The solver estimates the
camera pose in the base frame and every mount pose jointly, 6 dof each, by
minimising corner reprojection error.

Reprojection rather than a closed-form AX = XB: those need pairs of motions and
are sensitive to which pairs you feed them, while minimising reprojection error
uses every corner of every tag in every image and reports its residual in
pixels, which is the number that decides whether a session is trusted. The
closed form is still useful as a starting point, and `handeye_init` below runs
it (`cv2.calibrateHandEye` in its eye-to-hand arrangement) to get one.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .tags import corners_in_tag, project, tag_pose


@dataclass
class Observation:
    T_frame2base: np.ndarray
    T_tag2mount: np.ndarray
    corners: np.ndarray
    tag_id: int = -1
    mount: str = "card"


@dataclass
class Fit:
    T_cam2base: np.ndarray
    rms_px: float
    max_px: float
    n_obs: int
    mounts: dict = field(default_factory=dict)        # mount name -> T_mount2frame
    residuals: np.ndarray = None
    spread_deg: float = float("nan")                  # rotation-axis spread, see below
    init: str = ""                                    # which starting point won


def _se3(xi):
    """Exponential map of a 6-vector (rotation vector, translation) to 4x4."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(xi[:3], float))
    T[:3, 3] = xi[3:]
    return T


def _log(T):
    return np.concatenate([cv2.Rodrigues(np.asarray(T)[:3, :3])[0].ravel(), np.asarray(T)[:3, 3]])


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def _median_R(Rs, R_ref):
    """Median rotation in the tangent space at R_ref: R_ref @ exp(median(log(R_ref^T R_i)))."""
    rv = np.array([cv2.Rodrigues(R_ref.T @ R)[0].ravel() for R in Rs])
    dR, _ = cv2.Rodrigues(np.median(rv, axis=0))
    return R_ref @ dR


def _median_T(Ts):
    """Elementwise median of a list of transforms, re-orthonormalised.

    The rotation median is NOT taken over global Rodrigues vectors: for a
    rotation near 180 deg (a camera off to the side of the base) successive
    estimates land on both sides of the +-pi wrap (~+v and ~-v) and their
    elementwise median is a rotation ~180 deg away from every input. Instead,
    the median of rotation vectors relative to one input (small angles there,
    far from the wrap), then once more relative to that result in case the
    first input was an outlier. Away from the wrap the result differs from the
    old global median by hundredths of a degree.
    """
    Ts = [np.asarray(T, float) for T in Ts]
    t = np.median([T[:3, 3] for T in Ts], axis=0)
    Rs = [T[:3, :3] for T in Ts]
    R = _median_R(Rs, _median_R(Rs, Rs[0]))
    U, _, Vt = np.linalg.svd(R)                        # strips numerical drift from the products
    out = np.eye(4); out[:3, :3], out[:3, 3] = U @ Vt, t
    return out


def rotation_spread_R(Rs):
    """How much a set of orientations turned, in the worst direction, in degrees.

    Hand-eye is only conditioned if the carrier rotated about several
    non-parallel axes: a wave that only yaws leaves the camera pose free to
    slide along that axis, and the fit comes back confident and wrong. Stack
    the rotation vectors of every pair of orientations, take the smallest
    singular value of the (3, N) matrix normalised by sqrt(N), and you get the
    rotation available about the least-covered axis. Turning about one axis
    only scores 0, however far it turns."""
    Rs = list(Rs)
    if len(Rs) < 2:
        return 0.0
    v = [cv2.Rodrigues(Rs[i] @ Rs[j].T)[0].ravel() for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    M = np.array(v).T                                  # (3, N) rotation vectors
    # eigenvalues of M M^T rather than the SVD of M: with fewer than three
    # pairs the SVD returns fewer than three singular values and the smallest
    # of them is not the smallest axis.
    lam = np.linalg.eigvalsh(M @ M.T / M.shape[1])
    return float(np.degrees(np.sqrt(max(lam[0], 0.0))))


def rotation_spread(obs):
    """`rotation_spread_R` over the frames of a list of observations."""
    return rotation_spread_R([o.T_frame2base[:3, :3] for o in obs])


def _params(T_cam2base, mounts, names):
    p = np.empty(6 * (1 + len(names)))
    p[:6] = _log(T_cam2base)
    for k, n in enumerate(names):
        p[6 + 6 * k:12 + 6 * k] = _log(mounts[n])
    return p


def _unpack(p, names):
    return _se3(p[:6]), {n: _se3(p[6 + 6 * k:12 + 6 * k]) for k, n in enumerate(names)}


def _residuals(p, obs, K, dist, tag_size, names):
    """Stacked (2N,) pixel residuals for the camera pose and every mount pose."""
    T_cam2base, mounts = _unpack(p, names)
    T_base2cam = _inv(T_cam2base)
    X = np.c_[corners_in_tag(tag_size), np.ones(4)]
    out = []
    for o in obs:
        T = T_base2cam @ o.T_frame2base @ mounts[o.mount] @ o.T_tag2mount
        out.append((project((T @ X.T).T[:, :3], K, dist) - o.corners).ravel())
    return np.concatenate(out)


# ------------------------------------------------------------------- starts
def mount_poses_in_cam(obs, K, dist, tag_size):
    """Per-observation PnP, lifted from the tag to its mount frame."""
    return [tag_pose(o.corners, tag_size, K, dist) @ _inv(o.T_tag2mount) for o in obs]


MIN_PAIR_DEG = 10.0            # relative rotations smaller than this carry no information


def solve_axxb(A, B):
    """Closed-form AX = XB (Park and Martin), the classic hand-eye step.

    `cv2.calibrateHandEye` is gone in OpenCV 5, so this is the same method
    written out. Rotation first: for A X = X B the rotation vectors obey
    R_X log(R_B) = log(R_A), which is an orthogonal Procrustes problem over
    every motion pair. Translation follows linearly from
    (R_A - I) t_X = R_X t_B - t_A, stacked and solved in least squares."""
    a = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in A])
    b = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in B])
    U, _, Vt = np.linalg.svd(a.T @ b)
    R_X = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    M = np.vstack([TA[:3, :3] - np.eye(3) for TA in A])
    rhs = np.concatenate([R_X @ TB[:3, 3] - TA[:3, 3] for TA, TB in zip(A, B)])
    t_X = np.linalg.lstsq(M, rhs, rcond=None)[0]
    X = np.eye(4); X[:3, :3], X[:3, 3] = R_X, t_X
    return X


def handeye_init(obs, K, dist, tag_size, names, max_pairs=400):
    """Closed-form start, in the eye-to-hand arrangement.

    The camera is fixed in the base and the card rides the gripper, so what is
    constant across the wave is the card's pose in the gripper:

        T_card2gripper = T_base2gripper_i . T_cam2base . T_card2cam_i

    Equate two poses i and j and the unknowns separate into the standard form
    A X = X B with X = T_cam2base, A = FK_j FK_i^-1 the arm's relative motion
    and B = T_card2cam_j T_card2cam_i^-1 the card's motion as the camera saw
    it. The card's pose in the gripper then follows one per observation, and
    their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    pairs = [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
    A, B = [], []
    for i, j in pairs:
        a = obs[j].T_frame2base @ _inv(obs[i].T_frame2base)
        if np.linalg.norm(cv2.Rodrigues(a[:3, :3])[0]) < np.radians(MIN_PAIR_DEG):
            continue
        A.append(a)
        B.append(T_m2c[j] @ _inv(T_m2c[i]))
        if len(A) >= max_pairs:
            break
    if len(A) < 3:
        raise ValueError("too few distinct rotations for the closed-form hand-eye")
    T_cam2base = solve_axxb(A, B)
    mounts = {}
    for n in names:
        Ts = [_inv(o.T_frame2base) @ T_cam2base @ T for o, T in zip(obs, T_m2c) if o.mount == n]
        if Ts:
            mounts[n] = _median_T(Ts)
    return T_cam2base, mounts


def nominal_init(obs, K, dist, tag_size, mounts_nominal):
    """Start from the rough nominal mount ("about 90 mm past the fingertips"):
    each observation then gives a camera pose, and their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    Ts = [o.T_frame2base @ mounts_nominal[o.mount] @ _inv(T) for o, T in zip(obs, T_m2c)]
    return _median_T(Ts), dict(mounts_nominal)


def _lm(p, obs, K, dist, tag_size, names, iters=60):
    lam = 1e-3
    r = _residuals(p, obs, K, dist, tag_size, names)
    cost = float(r @ r)
    step = np.zeros_like(p)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros_like(p); dp[k] = 1e-6
            J[:, k] = (_residuals(p + dp, obs, K, dist, tag_size, names) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            r_new = _residuals(p + step, obs, K, dist, tag_size, names)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                p, r, cost, lam, improved = p + step, r_new, c_new, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    return p, r


def _fit(p, r, obs, names, tag=""):
    per_corner = np.linalg.norm(r.reshape(-1, 2), axis=1)
    T_cam2base, mounts = _unpack(p, names)
    return Fit(T_cam2base, float(np.sqrt(np.mean(per_corner**2))), float(per_corner.max()),
               len(obs), mounts, per_corner, rotation_spread(obs), tag)


def solve(obs, K, dist, tag_size, mounts_nominal, iters=60, restarts=5, max_px=1.5, seed=0):
    """Levenberg-Marquardt on the reprojection error, from the best of several starts.

    `mounts_nominal` maps every mount name appearing in `obs` to a rough guess
    at its pose in the frame it rides on; every one of them is solved for. The
    closed-form hand-eye start and the nominal start are both refined and the
    lower residual wins; if that is still above `max_px` the fit is restarted
    from jittered versions of it, because on a thin session LM can settle in a
    mirrored minimum with a residual of hundreds of pixels."""
    obs = list(obs)
    names = sorted({o.mount for o in obs})
    missing = [n for n in names if n not in mounts_nominal]
    if missing:
        raise ValueError(f"no nominal pose for mount(s) {missing}")
    starts = []
    try:
        starts.append(("handeye", *handeye_init(obs, K, dist, tag_size, names)))
    except (ValueError, cv2.error):
        pass                                   # too few distinct rotations for the closed form
    starts.append(("nominal", *nominal_init(obs, K, dist, tag_size, mounts_nominal)))

    best = None
    for tag, T_cam, mounts in starts:
        mounts = {n: mounts.get(n, mounts_nominal[n]) for n in names}
        p, r = _lm(_params(T_cam, mounts, names), obs, K, dist, tag_size, names, iters)
        fit = _fit(p, r, obs, names, tag)
        if best is None or fit.rms_px < best.rms_px:
            best = fit
    if best.rms_px > max_px:
        rng = np.random.default_rng(seed)
        for i in range(restarts):
            T = best.T_cam2base.copy(); T[:3, 3] += rng.normal(0, 0.05, 3)
            mounts = {n: best.mounts[n] for n in names}
            p, r = _lm(_params(T, mounts, names), obs, K, dist, tag_size, names, iters)
            fit = _fit(p, r, obs, names, f"jitter{i}")
            if fit.rms_px < best.rms_px:
                best = fit
            if best.rms_px <= max_px:
                break
    return best


def pose_error(T_a, T_b):
    """(translation error [m], rotation error [rad]) between two transforms."""
    dT = _inv(T_a) @ T_b
    ang = np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0])
    return float(np.linalg.norm(dT[:3, 3])), float(ang)
