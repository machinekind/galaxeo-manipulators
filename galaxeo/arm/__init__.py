"""galaxeo.arm: the installed Galaxea A1X as a library (pip install "galaxeo[arm]": numpy, mujoco).

    from galaxeo.arm import Arm, BusTransport
    from galaxeo.bus import open_bus

    arm = Arm(BusTransport(open_bus("can0"), tx=True), armed=True)
    plan = arm.plan([0.35, 0.0, 0.10])
    if plan:
        result = arm.move(plan.joints, speed=20.0)

model      the A1X MJCF + table plane z = 0 + tool frame (Tool, Box, Sphere)
kinematics FK and IK of the tool frame
collision  self, table and obstacle checks, pose and path
reach      the safe reach box, cached in assets/reach.json
guard      impact guard: lag and effort rules, backoff pose
motion     the guarded joint-space move, and the enable sequence
transport  CAN bus (BusTransport), MuJoCo (SimTransport); fake bus in `fake`
api        Arm: plan / move / stop
"""

from .collision import CollisionChecker
from .kinematics import IKResult, Kinematics, down_rotation
from .model import GRIPPER, JOINTS, Box, Sphere, Tool
from .reach import ReachBox

__all__ = [
    "Kinematics", "IKResult", "down_rotation", "CollisionChecker", "ReachBox",
    "Tool", "GRIPPER", "JOINTS", "Box", "Sphere",
]
