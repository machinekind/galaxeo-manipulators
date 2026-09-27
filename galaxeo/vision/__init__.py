"""Workspace RGB stereo → metric cloud in the arm base, then GraspGenX over ZMQ.

The two environment cameras are independently calibrated (`calibrate_real.py`
session JSON: K, dist, T_cam2base). This package rectifies them, runs SGBM,
and puts the cloud in the arm base (x forward over the table, z up). A thin
ZMQ client talks to a GraspGenX server; the client does not import torch or
graspgenx.

    from galaxeo.vision import Camera, reconstruct, GraspGenXClient, grasp_to_tool

pip install -e ".[vision]"
"""

from .camera import Camera, invert, load_session
from .frames import GRASP_TO_TOOL, grasp_to_tool, transform_points
from .graspgenx import GraspGenXClient, SweepVolumeParams
from .stereo import reconstruct, relative_pose, workspace_mask

__all__ = [
    "Camera", "invert", "load_session",
    "reconstruct", "relative_pose", "workspace_mask",
    "GraspGenXClient", "SweepVolumeParams",
    "GRASP_TO_TOOL", "grasp_to_tool", "transform_points",
]
