"""The grasp contract constants against the training recording (`meta.json`) and hand values."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from grasp_recording import RERECORD, meta, meta_value

from galaxeo.grasp import contract as c

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recorded() -> dict:
    return meta()


def run_python(code: str) -> subprocess.CompletedProcess:
    """`code` in a fresh interpreter, where nothing is imported yet."""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def test_camera_intrinsics_equal_the_recording(recorded):
    assert abs(c.FOCAL_PX - meta_value(recorded, "f_px")) < 1e-12
    assert abs(c.CX - meta_value(recorded, "cx")) < 1e-12
    assert abs(c.CY - meta_value(recorded, "cy")) < 1e-12
    assert abs(c.FOVY_DEG - meta_value(recorded, "fovy_deg")) < 1e-12
    assert list(meta_value(recorded, "image_hw")) == [c.IMG_H, c.IMG_W]
    assert abs(c.HEAT_SIGMA_FRAC - meta_value(recorded, "sigma_frac")) < 1e-12


def test_camera_pose_in_gripper_link_equals_the_recording(recorded):
    R = np.array(meta_value(recorded, "cam_R_gripper_link_opencv"))
    pos = np.array(meta_value(recorded, "cam_pos_gripper_link"))
    np.testing.assert_allclose(c.T_GRIPPER_CAMERA[:3, :3], R, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.T_GRIPPER_CAMERA[:3, 3], pos, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(c.T_GRIPPER_CAMERA[3], [0.0, 0.0, 0.0, 1.0])


def test_joint_ranges_equal_the_recording(recorded):
    np.testing.assert_allclose(c.JOINT_LO, meta_value(recorded, "joint_lo"), rtol=0, atol=1e-4)
    np.testing.assert_allclose(c.JOINT_HI, meta_value(recorded, "joint_hi"), rtol=0, atol=1e-4)


def test_steps_box_hover_and_arm_base_equal_the_recording(recorded):
    assert abs(c.STEP_XYZ - meta_value(recorded, "step_xyz")) < 1e-12
    assert abs(c.STEP_YAW - math.radians(meta_value(recorded, "step_yaw_deg"))) < 1e-12
    assert abs(c.BOX_XY - meta_value(recorded, "box_xy")) < 1e-12
    assert abs(c.HOVER - meta_value(recorded, "hover")) < 1e-12
    np.testing.assert_allclose(c.ARM_BASE_WORLD, meta_value(recorded, "arm_base"), rtol=0, atol=1e-12)


def test_training_floor_and_ceiling_equal_the_recording(recorded):
    # meta.json states them as formulas: "TABLE_TOP + 0.02 (world z 0.72)" and "p0_z + 0.08".
    floor = re.search(r"world z ([0-9.]+)", str(meta_value(recorded, "floor")))
    ceiling = re.fullmatch(r"p0_z \+ ([0-9.]+)", str(meta_value(recorded, "ceiling")).strip())
    if floor is None or ceiling is None:
        pytest.fail(f"meta.json floor {recorded['floor']!r} / ceiling {recorded['ceiling']!r} not readable: "
                    f"{RERECORD}")
    floor_base = float(floor.group(1)) - meta_value(recorded, "arm_base")[2]
    assert abs(c.TRAINING_FLOOR_Z - floor_base) < 1e-12
    assert abs(c.CEILING - float(ceiling.group(1))) < 1e-12


def test_calibration_joint_offsets_equal_the_recording(recorded):
    np.testing.assert_allclose(c.CALIBRATION_JOINT_OFFSETS_DEG,
                               meta_value(recorded, "calibration_joint_offsets_deg"), rtol=0, atol=1e-12)


def test_loop_ik_and_checkpoint_constants_equal_the_recording(recorded):
    assert c.CONTROL_HZ == meta_value(recorded, "control_hz")
    assert abs(c.TICK_S - meta_value(recorded, "substeps") * meta_value(recorded, "timestep")) < 1e-12
    assert c.MAX_TICKS == meta_value(recorded, "max_steps")
    assert c.IK_ITERS == meta_value(recorded, "ik_iters")
    assert abs(c.IK_DAMPING - meta_value(recorded, "ik_damping")) < 1e-12
    assert abs(c.IK_TOL_POS - meta_value(recorded, "ik_tol_pos")) < 1e-12
    assert abs(c.IK_TOL_ROT - meta_value(recorded, "ik_tol_rot")) < 1e-12
    assert c.CHECKPOINT_SHA256 == meta_value(recorded, "checkpoint_sha256")
    assert c.CHECKPOINT_BYTES == meta_value(recorded, "checkpoint_bytes")


def test_derived_constants_have_their_documented_values():
    assert c.STEP_YAW == 0.08726646259971647
    assert abs(c.FOCAL_PX - 142.49064161307925) < 1e-12
    assert c.CX == 79.5
    assert c.CY == 59.5
    assert abs(c.STREAM_UNITS_S - 76.92307692307692) < 1e-12
    assert c.TICK_S * c.CONTROL_HZ == 1.0
    assert c.OBS_DIM == 19
    assert c.ACT_DIM == 5
    assert abs(c.TEACH_MAX_AGE_S - 43200.0) < 1e-12


def test_the_published_files_are_on_the_model_page_over_https():
    assert c.CHECKPOINT_URL == "https://huggingface.co/hvsr-robotics/a1x-wrist-grasp-sim/resolve/main/student.pt"
    assert c.CAMERA_SPEC_URL == ("https://huggingface.co/hvsr-robotics/a1x-wrist-grasp-sim/resolve/main/"
                                 "wrist_camera_spec.json")


def test_camera_rotation_is_orthonormal_and_right_handed():
    R = c.T_GRIPPER_CAMERA[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_grasp_rotation_points_down_and_closes_along_the_yaw():
    np.testing.assert_allclose(c.grasp_R(0.0), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.grasp_R(math.pi / 2), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("yaw", [-3.0, -1.2, 0.0, 0.4, 2.9])
def test_grasp_rotation_is_a_right_handed_frame_with_the_closing_axis_at_the_yaw(yaw):
    R = c.grasp_R(yaw)
    np.testing.assert_allclose(R[:, 0], [0, 0, -1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(R[:, 1], [math.cos(yaw), math.sin(yaw), 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(R[:, 2], np.cross(R[:, 0], R[:, 1]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


@pytest.mark.parametrize("yaw", [-3.0, -1.2, 0.0, 0.4, 2.9])
def test_yaw_of_reads_the_closing_axis(yaw):
    assert abs(c.yaw_of(c.grasp_R(yaw)) - yaw) < 1e-12


def test_yaw_of_wraps_and_reads_a_bare_closing_axis():
    assert abs(c.yaw_of(c.grasp_R(math.pi / 2 + 2 * math.pi)) - math.pi / 2) < 1e-12
    R = np.zeros((3, 3))
    R[:, 1] = [0.0, -1.0, 0.0]
    assert abs(c.yaw_of(R) + math.pi / 2) < 1e-12


def test_wrap_maps_into_minus_pi_to_pi():
    assert c.wrap(math.pi) == -math.pi
    assert c.wrap(-math.pi) == -math.pi
    assert abs(c.wrap(0.1 + 2 * math.pi) - 0.1) < 1e-12
    assert abs(c.wrap(-0.1 - 4 * math.pi) + 0.1) < 1e-12
    assert abs(c.wrap(1.5 * math.pi) + 0.5 * math.pi) < 1e-12
    assert c.wrap(0.0) == 0.0


def test_wrap_keeps_float32_as_training_does():
    y = c.wrap(np.float32(3.0) + np.float32(0.5))
    assert isinstance(y, np.float32)
    assert y == (np.float32(3.5) + np.float32(math.pi)) % np.float32(2 * math.pi) - np.float32(math.pi)


def test_contract_rig_and_command_import_neither_torch_nor_opencv_nor_mujoco():
    out = run_python(textwrap.dedent("""
        import sys
        import galaxeo.grasp.contract, galaxeo.grasp.rig, galaxeo.grasp.command
        heavy = ("torch", "cv2", "mujoco")
        print(",".join(sorted(m for m in sys.modules if m.split(".")[0] in heavy or m.startswith("galaxeo.arm"))))
    """))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_no_grasp_module_asks_for_the_twin_torchvision_or_the_hub_client():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    # The finder sees every attempt, also one that a try/except would hide or that fails because
    # the package is absent.
    out = run_python(textwrap.dedent("""
        import sys
        asked = []
        class Spy:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in ("lerobot_mp", "torchvision", "huggingface_hub"):
                    asked.append(name)
                return None
        sys.meta_path.insert(0, Spy())
        import galaxeo.grasp.contract, galaxeo.grasp.rig, galaxeo.grasp.command
        import galaxeo.grasp.net, galaxeo.grasp.policy, galaxeo.grasp.camera
        print(sorted(set(asked)))
    """))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
