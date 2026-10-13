"""The twin's wrist camera in galaxeo/robot/a1x/robot.yaml is the camera the grasp student was trained with."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
mujoco = pytest.importorskip("mujoco")

from galaxeo.grasp.camera import HAND_BODY  # noqa: E402
from galaxeo.grasp.contract import FOVY_DEG, IMG_H, IMG_W, T_GRIPPER_CAMERA  # noqa: E402

FOLDER = Path(__file__).resolve().parents[1] / "galaxeo" / "robot" / "a1x"


@pytest.fixture(scope="module")
def wrist() -> dict:
    return yaml.safe_load((FOLDER / "robot.yaml").read_text(encoding="utf-8"))["cameras"]["wrist"]


def test_the_wrist_camera_rides_on_the_hand_body_of_the_contract(wrist):
    assert wrist["body"] == HAND_BODY


def test_the_wrist_camera_pose_is_the_contracts(wrist):
    assert np.array_equal(np.asarray(wrist["pose"], float), T_GRIPPER_CAMERA)


def test_the_wrist_camera_field_of_view_is_the_contracts(wrist):
    assert wrist["fovy_deg"] == FOVY_DEG


def test_the_wrist_camera_renders_at_the_policy_size(wrist):
    assert (wrist["width"], wrist["height"]) == (IMG_W, IMG_H)


def test_the_wrist_camera_has_no_other_keys(wrist):
    assert set(wrist) == {"body", "pose", "fovy_deg", "width", "height"}


def test_the_hand_body_is_in_the_robot_folders_model():
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "a1x.xml"))
    assert model.body(HAND_BODY).id > 0
