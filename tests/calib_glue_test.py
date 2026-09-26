"""End-to-end glue check: sim session's solver + real-arm FK, synthetic wave.

    python3 /tmp/rv/calib_glue.py
"""
import sys, os
os.chdir('/home/ljaniec/Repositories/galaxeo-manipulators')
sys.path.insert(0, '.')
sys.path.insert(0, 'ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver')
sys.path.insert(0, 'sim')

import numpy as np
import cv2
from kinematics import Chain, ik, pose_error, _T, _rpy


def _inv(T):
    out = np.eye(4); out[:3, :3] = T[:3, :3].T; out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _rotvec(R):
    v, _ = cv2.Rodrigues(np.asarray(R))
    return v


def rot(x_axis, y_axis):
    """Same construction as a1x_control.rot, inlined (a1x_control imports mujoco)."""
    x = np.asarray(x_axis, float); x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, float); y = y - x * (x @ y); y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


from a1x_arm import GripperFK

fk = GripperFK()
q0 = np.array([0.0, 1.0, -1.6, 0.6, 0.0, 0.0])

# --- IK on the sim's wave box, drawn like wave_candidates, from q_prev ---
rng = np.random.default_rng(0)
ok, miss = 0, 0
WAVE_X = (-0.22, 0.22); WAVE_Y = (0.0, 0.28); WAVE_Z = (0.14, 0.34)
q_prev = q0.copy()
for trial in range(60):
    p = np.array([rng.uniform(*WAVE_X), rng.uniform(*WAVE_Y), rng.uniform(*WAVE_Z)])
    yaw = np.arctan2(p[1], p[0])
    pitch = rng.uniform(0.15, 1.25)
    approach = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    e1 = np.cross([0, 0, 1.0], approach); n = np.linalg.norm(e1)
    e1 = np.array([1.0, 0, 0]) if n < 1e-6 else e1 / n
    e2 = np.cross(approach, e1)
    theta = rng.uniform(-np.pi, np.pi)
    close = np.cos(theta) * e1 + np.sin(theta) * e2
    R = rot(approach, close)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    q, T_got = ik(fk.chain, T, q_prev, iters=200, q_bias=q_prev)
    e = T_got @ _inv(T)
    ep = np.linalg.norm(e[:3, 3])
    ew = np.linalg.norm(_rotvec(T_got[:3, :3] @ T[:3, :3].T))
    if ep < 5e-3 and ew < np.radians(1.0):
        ok += 1
        q_prev = q                      # consecutive poses stay on one branch
print(f'IK on the sim wave box (chained from q_prev, like the session): '
      f'{ok}/60 converged (< 5 mm, 1 deg), {miss} missed')

# --- the handeye solver end to end with synthetic observations ------------
from calib.handeye import Observation, solve, pose_error, rotation_spread_R

TIP_X = 0.045 + 0.033
T_card_nom = np.eye(4)
T_card_nom[:3, :3] = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
T_card_nom[:3, 3] = np.array([TIP_X + 0.090 / 2, 0.0, 0.0])

T_cam_true = np.eye(4)
T_cam_true[:3, :3] = cv2.Rodrigues(np.array([0, 0, np.radians(35.0)]))[0]
T_cam_true[:3, 3] = np.array([0.55, 0.75, 0.42])
T_card_true = T_card_nom.copy()
T_card_true[:3, 3] += np.array([0.006, -0.004, 0.003])

H, W = 480, 640
fy = (H / 2) / np.tan(np.radians(85.0 / 2)); fx = fy
K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])


def project_pts(pts_cam):
    px, _ = cv2.projectPoints(np.asarray(pts_cam, float), np.zeros(3), np.zeros(3), K, None)
    return px.reshape(-1, 2)


X = np.c_[np.array([[-0.035, 0.035, 0], [0.035, 0.035, 0],
                    [0.035, -0.035, 0], [-0.035, -0.035, 0]]), np.ones(4)]

obs = []
seen_R = []
rng2 = np.random.default_rng(7)
for trial in range(24):
    q = q0 + rng2.normal(0, 0.25, 6)
    q = np.clip(q, fk.chain.lower, fk.chain.upper)
    T_g2b = fk.gripper2base(q)
    for tid, (tc, Rtag) in {0: (np.array([0, 0, 0.001]), np.eye(3)),
                            1: (np.array([0, 0, -0.001]), np.diag([-1.0, 1.0, -1.0]))}.items():
        T_tag2g = np.eye(4)
        T_tag2g[:3, :3] = T_card_true[:3, :3] @ Rtag
        T_tag2g[:3, 3] = T_card_true[:3, 3] + T_card_true[:3, :3] @ tc
        T_tag2cam = _inv(T_cam_true) @ T_g2b @ T_tag2g
        px = project_pts((T_tag2cam @ X.T).T[:, :3])
        px = px + rng2.normal(0, 0.3, px.shape)      # 0.3 px corner noise
        T_tag2mount = np.eye(4)
        T_tag2mount[:3, :3] = Rtag
        T_tag2mount[:3, 3] = tc
        obs.append(Observation(T_g2b, T_tag2mount, px.astype(np.float32), tid, "card"))
    seen_R.append(T_g2b[:3, :3])

print(f'synthetic wave: {len(obs)} observations, spread {rotation_spread_R(seen_R):.1f} deg')
fit = solve(obs, K, None, 0.070, {"card": T_card_nom}, max_px=1.5, seed=0)
dt, dr = pose_error(T_cam_true, fit.T_cam2base)
ct, cr = pose_error(T_card_true, fit.mounts["card"])
print(f'fit: rms {fit.rms_px:.2f}px max {fit.max_px:.2f}px init={fit.init}')
print(f'camera pose error:  {dt * 1000:.1f} mm  {np.degrees(dr):.2f} deg')
print(f'card mount error:   {ct * 1000:.1f} mm  {np.degrees(cr):.2f} deg')
ok = fit.rms_px <= 1.5 and dt < 0.01 and ct < 0.01
print('END-TO-END GLUE CHECK:', 'PASS' if ok else 'FAIL')
