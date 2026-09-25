"""Session test: collect_real + _solve_session end to end against the SIM robot.

Swaps only the Robot -- the sim's SimRobot drives the MuJoCo scene through its
position servos and renders laptop_cam, while the wave planning, the IK, the
FK glue and the gates are this module's real-arm code. What the sim's own
--sim does, driven by calibrate_real.collect_real instead.

    python3 /tmp/rv/sim_shim_test.py
"""
import sys, os
os.chdir('/home/ljaniec/Repositories/galaxeo-manipulators')
sys.path.insert(0, '.')
sys.path.insert(0, 'ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver')
sys.path.insert(0, 'sim')

import numpy as np
import calibrate_real as cr
from real_robot_a1x import GripperFK

from pour_scene import build
import mujoco
from a1x_control import Arm

model, data, info = build(3, calib_card=True)
arm = Arm(model, "arm/")
data.qpos[arm.fadr] = 0.0
data.ctrl[arm.grip] = 0.0
mujoco.mj_forward(model, data)


class SimRobotShim:
    """The sim's SimRobot through calibrate_real's Robot contract."""

    def __init__(self):
        from calib.calibrate import SimRobot
        self.r = SimRobot(model, data, info)
        self.chain = GripperFK().chain      # the real-arm URDF chain for the IK
        self.aborted = None

    def q(self):
        return self.r.q()

    def wait_fresh(self, required=False, timeout=3.0):
        return True

    def move(self, q, dur=1.5):
        self.r.move(q, dur)
        self.aborted = None

    def image(self):
        return self.r.image()


robot = SimRobotShim()
H, W = info["cam"]["H"], info["cam"]["W"]
fy = (H / 2) / np.tan(np.radians(info["cam"]["fovy"] / 2)); fx = fy
K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])

card = dict(nominal=cr._card_nominal(), tag_size=0.070,
            tags=cr._CARD_TAGS(0.070), K=K, dist=None)

obs, seen, n_poses = cr.collect_real(robot, GripperFK(), card,
                                     home=info["q_home"], seed=3, verbose=True)
print(f'\ncollect_real: {len(obs)} obs in {n_poses} poses, tags {sorted(seen)}')
fit = cr._solve_session(obs, card, 3)
print(f'fit: rms {fit.rms_px:.2f}px spread {fit.spread_deg:.1f}deg trusted={fit.trusted}')
from calib.handeye import pose_error
dt, dr = pose_error(info["cam"]["T_cam2base"], fit.T_cam2base)
ct, ctr = pose_error(info["calib"]["T_card2gripper"], fit.mounts["card"])
print(f'camera error: {dt*1000:.1f} mm {np.degrees(dr):.2f} deg')
print(f'card error:   {ct*1000:.1f} mm {np.degrees(ctr):.2f} deg')
ok = fit.trusted and dt < 0.02
print('SIM SHIM SESSION TEST:', 'PASS' if ok else 'FAIL')
