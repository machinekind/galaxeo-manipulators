"""Command bookkeeping against 1000 recorded ticks of the training environment, IK result injected."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from grasp_recording import npz

from galaxeo.grasp.command import CommandState, GraspCommand, Proposal
from galaxeo.grasp.contract import (
    ARM_BASE_WORLD,
    CEILING,
    HOVER,
    STEP_XYZ,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    TRAINING_FLOOR_Z,
    grasp_R,
    wrap,
)

BOOK_KEYS = ("action", "p_cmd0", "yaw_cmd0", "q_cmd0", "p0", "g_cmd0", "closing0", "attempts0", "ik_pos",
             "ik_yaw", "ik_R", "ik_q0", "ik_q", "ik_ok", "p_cmd1", "yaw_cmd1", "q_cmd1", "g_cmd1", "closing1",
             "attempts1", "prev_action1", "ctrl_arm", "ctrl_grip", "obs12_18", "episode", "tick")
BASE = np.array(ARM_BASE_WORLD)
#: 0.33 m out on the arm's left, as in training.
LEFT = np.array([0.0, 0.33, TOUCH_EXPECTED_Z + TIP_CLEAR])


@dataclass(frozen=True)
class Snapshot:
    """The command's state at one moment, copied."""

    p: np.ndarray
    yaw: float
    q: np.ndarray
    g: float
    closing: bool
    attempts: int
    prev_action: np.ndarray


@dataclass(frozen=True)
class Tick:
    """One recorded tick replayed: the state before, the proposal, the commit, the state after."""

    i: int
    before: Snapshot
    prop: Proposal
    edge: bool
    after: Snapshot
    tail: np.ndarray


def snapshot(s: CommandState) -> Snapshot:
    return Snapshot(p=s.p_cmd.copy(), yaw=s.yaw_cmd, q=s.q_cmd.copy(), g=s.g_cmd, closing=s.closing,
                    attempts=s.attempts, prev_action=s.prev_action.copy())


def failing(ticks: list[Tick], holds: Callable[[Tick], bool]) -> list[int]:
    """Indices of the ticks on which `holds` is false."""
    return [t.i for t in ticks if not holds(t)]


@pytest.fixture(scope="module")
def book() -> dict[str, np.ndarray]:
    return npz("bookkeeping.npz", *BOOK_KEYS)


@pytest.fixture(scope="module")
def replay(book) -> list[Tick]:
    """Every tick through one GraspCommand per episode, carried from tick to tick like training."""
    ticks = []
    cmd = None
    for i in range(len(book["action"])):
        if cmd is None or book["tick"][i] == 0:
            assert book["tick"][i] == 0, "an episode must start at tick 0"
            cmd = GraspCommand(book["p0"][i] - BASE, float(book["yaw_cmd0"][i]), book["q_cmd0"][i],
                               floor_z=TRAINING_FLOOR_Z)

        before = snapshot(cmd.state)
        prop = cmd.propose(book["action"][i])
        accepted = bool(book["ik_ok"][i])
        edge = cmd.commit(prop, book["ik_q"][i] if accepted else None)
        ticks.append(Tick(i=i, before=before, prop=prop, edge=edge, after=snapshot(cmd.state),
                          tail=cmd.obs_tail()))
    return ticks


# ------------------------------------------------------------------ against the recording

def test_the_state_carried_from_tick_to_tick_matches_training(book, replay):
    assert failing(replay, lambda t: np.abs(t.before.p - (book["p_cmd0"][t.i] - BASE)).max() <= 1e-12) == []
    assert failing(replay, lambda t: float(t.before.yaw) == book["yaw_cmd0"][t.i]) == []
    assert failing(replay, lambda t: np.array_equal(t.before.q, book["q_cmd0"][t.i])) == []
    assert failing(replay, lambda t: t.before.g == book["g_cmd0"][t.i]) == []
    assert failing(replay, lambda t: t.before.closing == book["closing0"][t.i]) == []
    assert failing(replay, lambda t: t.before.attempts == book["attempts0"][t.i]) == []


def test_proposals_match_the_ik_call_of_training(book, replay):
    assert failing(replay, lambda t: np.abs(t.prop.p - (book["ik_pos"][t.i] - BASE)).max() <= 1e-12) == []
    assert failing(replay, lambda t: isinstance(t.prop.yaw, np.float32)) == []
    assert failing(replay, lambda t: t.prop.yaw == np.float32(book["ik_yaw"][t.i])) == []
    # The recorded IK yaw is atan2 of the float32 yaw's rotation, in float64: one ulp off at most.
    assert failing(replay, lambda t: abs(wrap(float(t.prop.yaw) - float(book["ik_yaw"][t.i]))) <= 1e-12) == []
    assert failing(replay, lambda t: np.abs(t.prop.R - book["ik_R"][t.i]).max() <= 1e-12) == []
    assert failing(replay, lambda t: np.abs(t.prop.R - grasp_R(t.prop.yaw)).max() == 0.0) == []
    assert failing(replay, lambda t: t.prop.action.dtype == np.float32) == []
    assert failing(replay, lambda t: np.array_equal(t.prop.action, np.clip(book["action"][t.i], -1.0, 1.0))) == []
    assert failing(replay, lambda t: book["ctrl_grip"][t.i] == 0.05 * t.prop.g) == []


def test_state_after_each_tick_matches_training(book, replay):
    assert failing(replay, lambda t: np.abs(t.after.p - (book["p_cmd1"][t.i] - BASE)).max() <= 1e-12) == []
    # float32 values, bit for bit
    assert failing(replay, lambda t: float(t.after.yaw) == book["yaw_cmd1"][t.i]) == []
    assert failing(replay, lambda t: np.array_equal(t.after.q, book["q_cmd1"][t.i])) == []
    assert failing(replay, lambda t: np.array_equal(t.after.q, book["ctrl_arm"][t.i])) == []
    assert failing(replay, lambda t: t.after.g == book["g_cmd1"][t.i]) == []
    assert failing(replay, lambda t: t.after.closing == book["closing1"][t.i]) == []
    assert failing(replay, lambda t: t.after.attempts == book["attempts1"][t.i]) == []
    assert failing(replay, lambda t: t.edge == (t.after.attempts == book["attempts0"][t.i] + 1)) == []
    assert failing(replay, lambda t: np.array_equal(t.after.prev_action, book["prev_action1"][t.i])) == []
    assert failing(replay, lambda t: np.array_equal(t.tail[2:], book["obs12_18"][t.i])) == []


def test_accepted_yaw_is_float32_arithmetic_not_float64(book, replay):
    accepted = [t for t in replay if book["ik_ok"][t.i]]
    assert all(isinstance(t.prop.yaw, np.float32) for t in accepted)
    in_float64 = [wrap(float(book["yaw_cmd0"][t.i]) + float(np.clip(book["action"][t.i][3], -1, 1)) * math.radians(5.0))
                  for t in accepted]
    differ = sum(f != float(t.prop.yaw) for f, t in zip(in_float64, accepted))
    assert differ > len(accepted) // 2


def test_box_floor_ceiling_hits_refused_steps_and_attempts_all_occur(book, replay):
    assert sum(t.prop.box_hit for t in replay) > 0
    assert sum(t.prop.floor_hit for t in replay) > 0
    assert sum(t.prop.ceiling_hit for t in replay) > 0
    assert (~book["ik_ok"]).sum() > 0
    assert sum(t.edge for t in replay) > 0


def test_every_proposal_stays_in_the_box_between_floor_and_ceiling(book, replay):
    p0 = {t.i: book["p0"][t.i] - BASE for t in replay}
    assert failing(replay, lambda t: np.all(np.abs(t.prop.p[:2] - p0[t.i][:2]) <= 0.08 + 1e-12)) == []
    assert failing(replay, lambda t: TRAINING_FLOOR_Z <= t.prop.p[2] <= p0[t.i][2] + CEILING + 1e-12) == []


def test_the_observation_tail_equals_training_on_the_student_driven_ticks():
    shard = npz("shard.npz", "obs", "believed_pos", "believed_yaw", "yaw_cmd", "g_cmd", "prev_action", "attempts")
    bad = []
    for i in range(len(shard["obs"])):
        p_hover = shard["believed_pos"][i] - BASE + [0.0, 0.0, HOVER]
        cmd = GraspCommand(p_hover, float(shard["believed_yaw"][i]), np.zeros(6), floor_z=TRAINING_FLOOR_Z)
        s = cmd.state
        # Training's yaw_cmd is the hover's python float until a step is accepted, np.float32 after.
        yaw = float(shard["yaw_cmd"][i])
        s.yaw_cmd = yaw if yaw == s.believed_yaw else np.float32(yaw)
        s.g_cmd = float(shard["g_cmd"][i])
        s.prev_action = shard["prev_action"][i].copy()
        s.attempts = int(shard["attempts"][i])
        if not np.array_equal(cmd.obs_tail(), shard["obs"][i][10:19]):
            bad.append(i)
    assert bad == []
    assert (shard["yaw_cmd"] != shard["believed_yaw"]).sum() > len(shard["obs"]) // 2


def test_the_first_tick_tail_is_no_yaw_error_an_open_jaw_and_no_history():
    cmd = GraspCommand(LEFT + np.array([0.0, 0.0, HOVER]), 0.7, np.zeros(6), floor_z=TRAINING_FLOOR_Z)
    want = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cmd.obs_tail().tolist() == want
    hover = npz("hover.npz", "obs0_fk")
    assert np.array_equal(hover["obs0_fk"][:, 10:], np.tile(np.float32(want), (len(hover["obs0_fk"]), 1)))


# ------------------------------------------------------------------ by hand

def hover_command(floor_z: float = TRAINING_FLOOR_Z) -> GraspCommand:
    return GraspCommand(np.array([0.0, 0.33, 0.099]), 1.2, np.array([1.5, 2.0, -1.2, 0.7, 0.0, 0.8]),
                        floor_z=floor_z)


def test_a_new_command_starts_at_the_hover_with_the_tick0_tail():
    cmd = hover_command()
    s = cmd.state
    np.testing.assert_array_equal(s.p_cmd, s.p0)
    np.testing.assert_allclose(s.believed_pos, s.p0 - [0, 0, HOVER], rtol=0, atol=1e-15)
    assert type(s.yaw_cmd) is float
    assert s.believed_yaw == s.yaw_cmd
    tail = cmd.obs_tail()
    assert tail.dtype == np.float32
    np.testing.assert_array_equal(tail, [0, 1, 1, 0, 0, 0, 0, 0, 0])


def test_a_refused_ik_keeps_p_yaw_and_q_but_updates_g_attempts_and_prev_action():
    cmd = hover_command()
    s = cmd.state
    p = s.p_cmd.copy()
    yaw = s.yaw_cmd
    q = s.q_cmd.copy()

    prop = cmd.propose(np.array([1.0, -1.0, -1.0, 1.0, -1.0], np.float32))
    assert cmd.commit(prop, None) is True
    np.testing.assert_array_equal(s.p_cmd, p)
    assert s.yaw_cmd == yaw
    np.testing.assert_array_equal(s.q_cmd, q)
    assert s.g_cmd == 0.0
    assert s.closing
    assert s.attempts == 1
    assert s.ticks == 1
    np.testing.assert_array_equal(s.prev_action, prop.action)


def test_an_attempt_counts_once_per_close_and_again_after_an_opening():
    cmd = hover_command()
    cmd.commit(cmd.propose(np.array([0, 0, 0, 0, -1.0], np.float32)), None)
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, -0.9], np.float32)), None) is False
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, 1.0], np.float32)), None) is False
    assert cmd.commit(cmd.propose(np.array([0, 0, 0, 0, -1.0], np.float32)), None) is True
    assert cmd.state.attempts == 2
    assert abs(cmd.obs_tail()[8] - 2 / 3) < 1e-7


def test_an_accepted_step_moves_the_command_by_the_step_from_the_command():
    cmd = hover_command()
    s = cmd.state
    q_new = s.q_cmd + 0.01
    prop = cmd.propose(np.array([1.0, 0.5, -0.25, 1.0, 1.0], np.float32))
    assert cmd.commit(prop, q_new) is False

    step = np.array([1.0, 0.5, -0.25], np.float32) * STEP_XYZ          # float32 product, as in training
    assert step.dtype == np.float32
    np.testing.assert_array_equal(s.p_cmd, s.p0 + step)
    assert np.abs(s.p_cmd - s.p0 - np.array([1.0, 0.5, -0.25]) * STEP_XYZ).max() < 1e-9
    assert isinstance(s.yaw_cmd, np.float32)
    assert abs(float(s.yaw_cmd) - (1.2 + math.radians(5.0))) < 1e-6
    np.testing.assert_array_equal(s.q_cmd, q_new)

    tail = cmd.obs_tail()
    assert abs(tail[0] - math.sin(2 * math.radians(5.0))) < 1e-6
    assert abs(tail[1] - math.cos(2 * math.radians(5.0))) < 1e-6


@pytest.mark.parametrize("bad", [
    [math.nan, 0, 0, 0, 0], [0, 0, 0, 0, math.inf], [0, 0, 0, -math.inf, 0], [0, 0, 0, 0], [[0, 0, 0, 0, 0]],
])
def test_a_non_finite_or_misshaped_action_raises_and_changes_nothing(bad):
    cmd = hover_command()
    p_before = cmd.state.p_cmd.copy()
    with pytest.raises(ValueError):
        cmd.propose(np.array(bad, dtype=np.float64))
    np.testing.assert_array_equal(cmd.state.p_cmd, p_before)
    assert cmd.state.ticks == 0


def test_a_non_finite_ik_result_raises_before_anything_changes():
    cmd = hover_command()
    prop = cmd.propose(np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        cmd.commit(prop, np.array([0, 0, math.nan, 0, 0, 0.0]))
    assert cmd.state.ticks == 0


def test_raising_the_floor_between_ticks_lifts_the_next_proposal_to_it():
    cmd = hover_command()
    s = cmd.state
    for _ in range(12):                                   # walk down to the floor
        prop = cmd.propose(np.array([0, 0, -1.0, 0, 1.0], np.float32))
        cmd.commit(prop, s.q_cmd)
    assert prop.floor_hit
    assert s.p_cmd[2] == TRAINING_FLOOR_Z

    cmd.floor_z = 0.031
    prop = cmd.propose(np.zeros(5, np.float32))
    assert prop.p[2] == 0.031
    assert prop.floor_hit
    cmd.commit(prop, s.q_cmd)
    assert s.p_cmd[2] == 0.031


def test_a_floor_above_the_ceiling_is_refused():
    with pytest.raises(ValueError):
        hover_command(floor_z=0.099 + CEILING + 1e-6)
    cmd = hover_command()
    cmd.floor_z = 0.2
    with pytest.raises(ValueError):
        cmd.propose(np.zeros(5, np.float32))
