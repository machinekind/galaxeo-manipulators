"""The 19-vector the grasp student reads, against training's recording.

The fixtures come from the training environment (tests/fixtures/grasp/README.md). The vector is
compared with the `*_fk` fields, which training's own model gives AT the recorded joints:
training's recorded `obs` lags its joints by one 2 ms physics substep in the TCP entries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from grasp_recording import npz

pytest.importorskip("mujoco")

from galaxeo.arm import Kinematics  # noqa: E402
from galaxeo.grasp.command import GraspCommand  # noqa: E402
from galaxeo.grasp.contract import (  # noqa: E402
    ARM_BASE_WORLD,
    HOVER,
    JOINT_HI,
    JOINT_LO,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TRAINING_FLOOR_Z,
)
from galaxeo.grasp.observer import GraspObserver, range_mismatch  # noqa: E402

LEFT = np.array([0.0, 0.33, TOUCH_EXPECTED_Z + TIP_CLEAR])     # on the arm's left, where training put objects
SHARD = ("q", "gap", "believed_pos", "believed_yaw", "yaw_cmd", "g_cmd", "prev_action", "attempts", "obs", "obs_fk")
TCP_ENTRIES = slice(7, 10)


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


@pytest.fixture(scope="module")
def observer(kin: Kinematics) -> GraspObserver:
    return GraspObserver(kin)


@pytest.fixture(scope="module")
def shard() -> dict[str, np.ndarray]:
    return npz("shard.npz", *SHARD)


@pytest.fixture(scope="module")
def rebuilt(observer: GraspObserver, shard: dict[str, np.ndarray]) -> np.ndarray:
    """The vector of every recorded tick, from the recorded joints, gap and command."""
    base = np.asarray(ARM_BASE_WORLD)
    rows = []
    for i in range(len(shard["q"])):
        believed = shard["believed_pos"][i] - base
        cmd = GraspCommand(believed + np.array([0.0, 0.0, HOVER]), float(shard["believed_yaw"][i]),
                           shard["q"][i], floor_z=TRAINING_FLOOR_Z)
        s = cmd.state
        s.yaw_cmd = float(shard["yaw_cmd"][i])
        s.g_cmd = float(shard["g_cmd"][i])
        s.prev_action = shard["prev_action"][i].astype(np.float32)
        s.attempts = int(shard["attempts"][i])
        rows.append(observer.build(shard["q"][i], float(shard["gap"][i]), cmd))
    return np.stack(rows)


def command_at(p_believed: np.ndarray, yaw: float = 0.0) -> GraspCommand:
    return GraspCommand(p_believed + np.array([0.0, 0.0, HOVER]), yaw, np.zeros(6), floor_z=TRAINING_FLOOR_Z)


# ------------------------------------------------------------------ against the recording

def test_the_vector_from_the_recorded_joints_equals_trainings_at_those_joints(rebuilt, shard):
    assert np.abs(rebuilt - shard["obs_fk"]).max() <= 1e-5


def test_the_vector_is_within_the_substep_lag_of_what_training_fed_the_policy(rebuilt, shard):
    assert np.abs(rebuilt - shard["obs"]).max() <= 3e-3


def test_trainings_recorded_vector_differs_from_the_one_at_its_joints_only_in_the_tcp_entries(shard):
    others = np.r_[0:7, 10:19]
    assert np.array_equal(shard["obs"][:, others], shard["obs_fk"][:, others])


def test_trainings_recorded_tcp_entries_lag_by_at_most_three_thousandths(shard):
    lag = np.abs(shard["obs"][:, TCP_ENTRIES].astype(float) - shard["obs_fk"][:, TCP_ENTRIES].astype(float))
    assert lag.max() <= 3e-3


def test_the_vector_at_the_settled_hover_equals_trainings_first_observation(observer):
    h = npz("hover.npz", "p_hover", "yaw", "q_hover", "q_measured", "gap0", "obs0_fk")
    rows = []
    for i in range(len(h["yaw"])):
        cmd = GraspCommand(h["p_hover"][i], float(h["yaw"][i]), h["q_hover"][i], floor_z=TRAINING_FLOOR_Z)
        rows.append(observer.build(h["q_measured"][i], float(h["gap0"][i]), cmd))
    assert np.abs(np.stack(rows) - h["obs0_fk"]).max() <= 1e-5


# ------------------------------------------------------------------ the parts

def test_the_vector_is_19_float32(observer):
    v = observer.build(np.zeros(6), 0.103, command_at(LEFT))
    assert (v.dtype, v.shape) == (np.float32, (19,))


def test_joints_at_the_model_lower_limits_normalise_to_minus_one(observer):
    v = observer.build(np.asarray(JOINT_LO), 0.103, command_at(LEFT))
    np.testing.assert_allclose(v[:6], -1.0, atol=1e-6)


def test_joints_at_the_model_upper_limits_normalise_to_one(observer):
    v = observer.build(np.asarray(JOINT_HI), 0.103, command_at(LEFT))
    np.testing.assert_allclose(v[:6], 1.0, atol=1e-6)


def test_joints_at_the_app_limits_stay_inside_the_normalised_range(observer):
    # robot.yaml's limits sit 3-5 degrees inside the MJCF ranges: they are not the ends of the range.
    app_hi = np.radians([160.0, 177.0, -3.0, 85.0, 85.0, 160.0])
    v = observer.build(app_hi, 0.103, command_at(LEFT))
    assert np.all(v[:6] < 0.99)


def test_the_gap_enters_raw_times_ten(observer):
    v = observer.build(np.zeros(6), 0.103, command_at(LEFT))
    assert v[6] == np.float32(1.03)


def test_the_offset_is_the_believed_point_minus_the_fk_tcp_times_ten(observer, kin):
    q = np.radians([90.0, 120.0, -77.0, 47.0, 0.0, 0.0])
    cmd = command_at(LEFT)
    v = observer.build(q, 0.103, cmd)
    want = (cmd.state.believed_pos - kin.fk(q)[:3, 3]) * 10.0
    np.testing.assert_allclose(v[TCP_ENTRIES], want, atol=1e-6)


def test_the_tail_is_the_commands_own(observer):
    cmd = command_at(LEFT, yaw=0.7)
    v = observer.build(np.zeros(6), 0.103, cmd)
    assert v[10:].tolist() == cmd.obs_tail().tolist()


def test_the_tcp_is_forward_kinematics_of_the_joints(observer, kin):
    q = np.radians([90.0, 120.0, -77.0, 47.0, 0.0, 0.0])
    np.testing.assert_allclose(observer.tcp(q), kin.fk(q)[:3, 3], atol=1e-12)


def test_a_wrong_number_of_joints_raises(observer):
    with pytest.raises(ValueError):
        observer.build(np.zeros(7), 0.103, command_at(LEFT))


# ------------------------------------------------------------------ the model's ranges

def test_galaxeos_model_has_trainings_joint_ranges(kin):
    assert range_mismatch(kin) is None


def test_a_model_with_other_joint_ranges_is_refused():
    other = Kinematics()
    other.lo = other.lo.copy()
    other.lo[0] -= math.radians(1.0)
    with pytest.raises(ValueError, match="trained"):
        GraspObserver(other)
