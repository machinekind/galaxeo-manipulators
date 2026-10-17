"""galaxeo.arm: the installed Galaxea A1X as a library (pip install "galaxeo[arm]": numpy, mujoco).

    from galaxeo.arm import Arm, BusTransport
    from galaxeo.bus import open_bus

    arm = Arm(BusTransport(open_bus("can0"), tx=True), armed=True)
    plan = arm.plan([0.35, 0.0, 0.10])
    if plan:
        result = arm.move(plan.joints, speed=20.0)

model      the A1X MJCF + table plane z = 0 + tool frame (Tool, Box, Sphere)
kinematics FK and IK of the tool frame, a warm single-start IK step, body poses, fingertips
collision  self, table and obstacle checks, pose and path
stages     staged joint-space plan: elbow, wrist, then shoulder and base
reach      the safe reach box, cached in assets/reach.json
guard      impact guard: lag and effort rules, backoff pose
motion     the guarded joint-space move, and the enable sequence
transport  CAN bus (BusTransport), MuJoCo (SimTransport); fake bus in `fake`
api        Arm: plan / move / stop
"""

from .api import Arm, Plan
from .collision import CollisionChecker
from .kinematics import IKResult, IKStep, JointBox, Kinematics, down_rotation, joint_box
from .model import GRIPPER, JOINTS, Box, Sphere, Tool
from .motion import Motion, MoveResult, enable
from .reach import ReachBox
from .stages import StagePlan, stage_plan
from .transport import BusTransport, Reading, SimTransport

__all__ = [
    "Arm", "Plan", "MoveResult", "Motion", "enable",
    "Kinematics", "IKResult", "IKStep", "JointBox", "joint_box", "down_rotation",
    "CollisionChecker", "ReachBox", "StagePlan", "stage_plan",
    "Tool", "GRIPPER", "JOINTS", "Box", "Sphere",
    "BusTransport", "SimTransport", "Reading",
]
