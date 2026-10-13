"""The grasp student's training contract, as constants.

Positions are metres in the arm base frame (origin on the joint 1 axis, z = 0 at the table top,
x forward, y left); training's world frame is the base frame shifted by `ARM_BASE_WORLD`, with no
rotation. Yaw is the yaw of the gripper's CLOSING axis (`R[:, 1]`) about base +z, in radians.
Joints are radians inside the bookkeeping and the observation vector, degrees towards the twin.

`obs_mean` / `obs_std` are not here: they are read from the checkpoint.
"""

from __future__ import annotations

import math

import numpy as np

CONTROL_HZ = 20.0
TICK_S = 0.05
MAX_TICKS = 120
STEP_XYZ = 0.008                       # m per unit action per tick
STEP_YAW = math.radians(5.0)           # 0.08726646259971647 rad
HOVER = 0.06                           # m, hover above the believed grasp
BOX_XY = 0.08                          # m, half side of the square around the hover, per axis
CEILING = 0.08                         # m above the hover
TRAINING_FLOOR_Z = 0.020               # m, base frame (training: TABLE_TOP + 0.02)
FLOOR_MARGIN = 0.004                   # m above the taught touch
SHADOW_FLOOR_ABOVE_BELIEVED = 0.020    # m
SAG_MAX = 0.008                        # m, largest sag accepted at the hover
SAG_LIMIT = 0.012                      # m, largest sag the floor absorbs within an episode
SAG_FILTER_S = 0.3                     # s, time constant of the sag estimate
TABLE_END_BELOW = 0.0005               # m, measured TCP z under touch + this ...
TABLE_END_TICKS = 2                    # ... on this many ticks in a row ends with `table`
SHADOW_CLEAR = 0.010                   # m, commanded fingertips above the object's top in a shadow episode
OBJECT_HEIGHT_DEFAULT = 0.030          # m, the 3 cm cube
RETREAT_STEP = 0.005                   # m per step of the straight-up retreat
RETREAT_BELOW = 0.010                  # m under the hover height that counts as low
RETREAT_NEAR_XY = 0.15                 # m
RETREAT_TILT_MAX_DEG = 10.0
RETREAT_CLEAR = 0.015                  # m over touch + object height: a refused step there ends the retreat
TIP_CLEAR = 0.006                      # m, fingertips above the table at the grasp
# m, TCP above the lowest fingertip corner, tool pointing down (the twin's model)
TIP_AHEAD = 0.0339
TOUCH_EXPECTED_Z = 0.0339              # m
TOUCH_TOLERANCE = 0.015                # m, a taught touch further off means a mount or kinematics problem
LIFT = 0.04                            # m
CLOSE_BELOW = 0.4                      # g below this counts as closing
GAP_CLOSED = 0.003                     # m
GAP_PER_UNIT = 0.001                   # m per unit of opening (0..100)
# Sim evaluation hover noise (normal sigmas).
NOISE_XY = 0.015                       # m
NOISE_Z = 0.010                        # m
NOISE_YAW = math.radians(10.0)
IMG_W = 160
IMG_H = 120
HEAT_SIGMA_FRAC = 0.06                 # of the WIDTH: sigma 9.6 px
FOVY_DEG = 45.67
FOCAL_PX = (IMG_H / 2) / math.tan(math.radians(FOVY_DEG) / 2)    # 142.49064161307925
CX = IMG_W / 2 - 0.5                   # 79.5
CY = IMG_H / 2 - 0.5                   # 59.5
NEAR_Z = 1e-3                          # m, a point at or behind this depth has no pixel
OBS_DIM = 19
ACT_DIM = 5
JOINT_LO = (-2.8798, 0.0, -3.3161, -1.5708, -1.5708, -2.8798)    # rad, MJCF ranges
JOINT_HI = (2.8798, 3.1416, 0.0, 1.5708, 1.5708, 2.8798)
ARM_BASE_WORLD = (0.0, -0.20, 0.70)    # training world -> base, fixtures only
IK_ITERS = 60
IK_DAMPING = 0.05
IK_TOL_POS = 1e-3                      # m
IK_TOL_ROT = 5e-3                      # rad
IK_MAX_JUMP_DEG = 6.0
ARM_SPEED_DEG_S = 90.0
PREPOSITION_SPEED_DEG_S = 20.0
ARRIVE_DEG = 2.0
ARRIVE_M = 0.003
ARRIVE_STILL_DEG_S = 10.0
ARRIVE_TIMEOUT_S = 3.0
LAG_HOLD_DEG = 8.0
# a joint above this share of its effort limit at rest gets the lag rule only
EFFORT_REST_FRAC = 0.5
HOVER_J5_MAX_DEG = 0.5                 # solved hover
HOVER_J5_MEASURED_MAX_DEG = 1.0        # measured at the hover, real backend
HOVER_J6_MARGIN_DEG = 15.0
TRAIN_J1_DEG = (40.0, 139.0)
TRAIN_RADIUS = (0.22, 0.44)
TRAIN_HOVER_Z = (0.05, 0.14)
STREAM_UNITS_S = 100.0 * 2.0 / 2.6     # 76.92, the A1X backend's gripper stream
STREAM_STROKE_S = 1.3
FREEZE_DISABLED = 100.0
DOT_MAX_PX = 8.0
FRAME_AGE_MAX_S = 0.12
JOINT_AGE_MAX_S = 0.10
PAIR_GAP_MAX_S = 0.06
FB_AGE_END_S = 0.3                     # feedback or joint sample older than this ends the episode
# in force when T_GRIPPER_CAMERA was calibrated
CALIBRATION_JOINT_OFFSETS_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TEACH_MAX_AGE_S = 12 * 3600.0
TABLE_TOUCH_FAR = 0.10                 # m, table touch to grasp point, xy
LATENCY_MAX_S = 0.150
RAW_W = 1920
RAW_H = 1080

CHECKPOINT_SHA256 = "05e5aaf6e7d47126d494a90957238d3a4275f7cccdc36ccc00679f71cdcb8536"
CHECKPOINT_BYTES = 45736059
_PUBLISHED = "https://huggingface.co/hvsr-robotics/a1x-wrist-grasp-sim/resolve/main"
CHECKPOINT_URL = f"{_PUBLISHED}/student.pt"
CAMERA_SPEC_URL = f"{_PUBLISHED}/wrist_camera_spec.json"

# Camera in gripper_link, OpenCV axes (x right, y down, z forward), upright image.
T_GRIPPER_CAMERA = np.array([
    [-0.5396919613474123, -0.34962129600231007, 0.7658312713898854, -0.05060687411211338],
    [-0.840763266463087, 0.27031196633365107, -0.46909334958316684, 0.04299012698832195],
    [-0.043008332021868434, -0.8970487111849055, -0.4398339381382439, 0.0935611657849058],
    [0.0, 0.0, 0.0, 1.0]])

# Measured 2026-09-26 on the upright (180 degree rotated) 1920x1080 frame; 15 views, rms 0.85 px, NOT trusted.
WRIST_K_2026_09_26 = np.array([[1257.2410784993194, 0.0, 1007.7976467876623],
                               [0.0, 1282.5093913726053, 521.1319818950168],
                               [0.0, 0.0, 1.0]])
WRIST_DIST_2026_09_26 = np.array([-0.6480689488, 0.4993297587, 0.0043182905, -0.051339918, -0.1545135673])


def wrap(a: float) -> float:
    """Angle into [-pi, pi). An np.float32 stays np.float32: training wraps the yaw in float32."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def grasp_R(yaw: float) -> np.ndarray:
    """Top-down grasp rotation; columns: approach (down), closing axis, approach x closing."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[0.0, c, s], [0.0, s, -c], [-1.0, 0.0, 0.0]])


def yaw_of(R: np.ndarray) -> float:
    """Yaw of the closing axis `R[:, 1]` about +z."""
    return math.atan2(R[1, 1], R[0, 1])
