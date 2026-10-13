"""The hover over the believed grasp and the staged plan to it, against training's hovers.

hover.npz holds 200 episode starts recorded from the training environment
(tests/fixtures/grasp/README.md): the believed point, the closing-axis yaw and the joints
training solved for that yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from grasp_recording import npz

pytest.importorskip("mujoco")
yaml = pytest.importorskip("yaml")

from galaxeo.arm import CollisionChecker, Kinematics  # noqa: E402
from galaxeo.arm.model import Box  # noqa: E402
from galaxeo.grasp.contract import (  # noqa: E402
    HOVER,
    HOVER_J5_MAX_DEG,
    HOVER_J6_MARGIN_DEG,
    IK_TOL_POS,
    IK_TOL_ROT,
    JOINT_HI,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    grasp_R,
    wrap,
)
from galaxeo.grasp.hover import (  # noqa: E402
    STAGES,
    Hover,
    HoverError,
    joint_box,
    solve_hover,
    stage_plan,
    training_envelope,
)

ROBOT_YAML = Path(__file__).resolve().parents[1] / "galaxeo" / "robot" / "a1x" / "robot.yaml"
LIMITS = {"arm_joint1": (-160.0, 160.0), "arm_joint2": (3.0, 177.0), "arm_joint3": (-187.0, -3.0),
          "arm_joint4": (-85.0, 85.0), "arm_joint5": (-85.0, 85.0), "arm_joint6": (-160.0, 160.0)}
LEFT = np.array([0.0, 0.33, TOUCH_EXPECTED_Z + TIP_CLEAR])
J1 = 0
J5 = 4
J6 = 5


@dataclass(frozen=True)
class Solved:
    believed: np.ndarray
    yaw: float                  # training's yaw
    q_training: np.ndarray
    hover: Hover


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def rotation_angle(R: np.ndarray) -> float:
    return math.acos(min(1.0, max(-1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def j6_margin(j6_deg: float) -> float:
    lo, hi = LIMITS["arm_joint6"]
    return min(j6_deg - lo, hi - j6_deg)


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


@pytest.fixture(scope="module")
def box(kin):
    return joint_box(kin, LIMITS)


@pytest.fixture(scope="module")
def solved(kin, box) -> list[Solved]:
    h = npz("hover.npz", "believed", "yaw", "q_hover")
    return [Solved(h["believed"][i], float(h["yaw"][i]), h["q_hover"][i],
                   solve_hover(kin, h["believed"][i], float(h["yaw"][i]), box))
            for i in range(len(h["yaw"]))]


# ------------------------------------------------------------------ training's hovers

def test_every_training_hover_is_the_believed_point_plus_the_hover_height(solved):
    err = max(float(np.abs(s.hover.p - (s.believed + np.array([0.0, 0.0, HOVER]))).max()) for s in solved)
    assert err <= 1e-12


def test_every_training_hover_keeps_the_believed_point(solved):
    assert all(np.array_equal(s.hover.believed, s.hover.p - np.array([0.0, 0.0, HOVER])) for s in solved)


def test_every_training_hover_reaches_its_point_within_the_ik_tolerance(solved, kin):
    assert max(float(np.linalg.norm(s.hover.p - kin.fk(s.hover.q)[:3, 3])) for s in solved) < IK_TOL_POS


def test_every_training_hover_points_down_at_its_yaw_within_the_ik_tolerance(solved, kin):
    err = max(rotation_angle(grasp_R(s.hover.yaw) @ kin.fk(s.hover.q)[:3, :3].T) for s in solved)
    assert err < IK_TOL_ROT


def test_every_training_hover_has_j5_at_zero(solved):
    assert max(abs(math.degrees(s.hover.q[J5])) for s in solved) <= HOVER_J5_MAX_DEG


def test_every_training_hover_keeps_j6_clear_of_its_limits(solved):
    assert min(s.hover.j6_margin_deg for s in solved) >= HOVER_J6_MARGIN_DEG


def test_the_reported_j6_margin_is_the_joints(solved):
    err = max(abs(s.hover.j6_margin_deg - j6_margin(math.degrees(s.hover.q[J6]))) for s in solved)
    assert err <= 1e-9


def test_the_hover_yaw_is_the_believed_yaw_or_its_flip(solved):
    assert all(s.hover.yaw in (wrap(s.yaw), wrap(s.yaw + math.pi)) for s in solved)


def test_j6_is_j1_plus_90_minus_the_yaw_for_a_tool_pointing_down(solved):
    err = max(abs(wrap_deg(math.degrees(s.hover.q[J6]) - (math.degrees(s.hover.q[J1]) + 90.0
                                                           - math.degrees(s.hover.yaw)))) for s in solved)
    assert err <= 0.05


def test_the_chosen_candidate_is_the_one_further_from_a_j6_limit(solved):
    # The other candidate stands 180 degrees away on J6.
    worse = [s for s in solved
             if s.hover.j6_margin_deg < j6_margin(wrap_deg(math.degrees(s.hover.q[J6]) + 180.0)) - 0.05]
    assert worse == []


def test_every_training_hover_lies_inside_the_training_envelope(solved):
    assert all(s.hover.outside_training == () for s in solved)


def test_where_training_kept_its_yaw_and_room_on_j6_the_joints_are_trainings_within_a_degree(solved):
    same = [s for s in solved
            if s.hover.yaw == wrap(s.yaw) and j6_margin(math.degrees(s.q_training[J6])) >= HOVER_J6_MARGIN_DEG]
    assert max(float(np.max(np.abs(np.degrees(s.hover.q - s.q_training)))) for s in same) <= 1.0


def test_one_point_always_gives_one_hover(kin, box):
    first = solve_hover(kin, LEFT, 0.4, box)
    again = solve_hover(kin, LEFT, 0.4, box)
    assert np.array_equal(first.q, again.q)


# ------------------------------------------------------------------ the J6 rule and the limits

def test_a_yaw_that_puts_j6_near_its_limit_is_flipped(kin, box):
    # J1 is 90 deg on the arm's left; J6 = J1 + 90 - yaw: -60 deg gives J6 240, the flip 60.
    hv = solve_hover(kin, LEFT, math.radians(-60.0), box)
    assert hv.yaw == pytest.approx(math.radians(120.0))


def test_the_flipped_hover_stands_at_j6_60(kin, box):
    hv = solve_hover(kin, LEFT, math.radians(-60.0), box)
    assert math.degrees(hv.q[J6]) == pytest.approx(60.0, abs=0.5)


def test_a_yaw_with_room_on_j6_is_kept(kin, box):
    hv = solve_hover(kin, LEFT, math.radians(150.0), box)
    assert hv.yaw == pytest.approx(math.radians(150.0))


def test_the_callers_limits_decide_between_the_candidates(kin):
    narrow = joint_box(kin, {**LIMITS, "arm_joint6": (-160.0, 40.0)})
    hv = solve_hover(kin, LEFT, math.radians(-60.0), narrow)
    assert math.degrees(hv.q[J6]) == pytest.approx(-120.0, abs=0.5)


def test_a_limit_wider_than_the_model_is_cut_to_the_model(kin):
    wide = joint_box(kin, {**LIMITS, "arm_joint6": (-400.0, 400.0)})
    assert wide.hi[J6] == pytest.approx(JOINT_HI[J6])


def test_without_room_on_j6_for_either_candidate_there_is_no_hover(kin):
    tight = joint_box(kin, {**LIMITS, "arm_joint6": (-130.0, 70.0)})
    with pytest.raises(HoverError) as e:
        solve_hover(kin, LEFT, math.radians(-60.0), tight)
    assert e.value.detail.count("zapas J6") == 2


def test_a_point_that_is_not_a_number_has_no_hover(kin, box):
    with pytest.raises(HoverError):
        solve_hover(kin, np.array([math.nan, 0.33, 0.04]), 0.0, box)


def test_a_point_in_front_of_the_arm_is_outside_training_on_joint_1(kin, box):
    hv = solve_hover(kin, np.array([0.33, 0.0, TOUCH_EXPECTED_Z + TIP_CLEAR]), 0.0, box)
    assert any(s.startswith("J1 ") for s in hv.outside_training)


def test_the_training_envelope_names_every_failed_check():
    assert training_envelope(np.radians([12.0, 0, 0, 0, 0, 0]), np.array([0.19, 0.0, 0.15])) == (
        "J1 12.0 st. poza 40..139",
        "promien 0.19 m poza 0.22..0.44",
        "wysokosc zawisu 0.150 m poza 0.05..0.14",
    )


# ------------------------------------------------------------------ the staged plan

START = np.radians([0.0, 60.0, -90.0, 30.0, 0.0, 0.0])
GOAL = np.radians([90.0, 120.0, -80.0, 50.0, 0.0, 20.0])


def moved(a: np.ndarray, b: np.ndarray) -> set[str]:
    names = [f"arm_joint{k}" for k in range(1, 7)]
    return {names[k] for k in range(6) if a[k] != b[k]}


def never(a: np.ndarray, b: np.ndarray) -> str | None:
    return None


def test_the_stages_are_the_robot_folders_staged_home_without_the_gripper():
    stages = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))["home_stages"]
    arm = [tuple(j for j in stage if j != "gripper") for stage in stages]
    assert tuple(stage for stage in arm if stage) == STAGES


def test_the_plan_moves_the_elbow_then_the_wrist_then_shoulder_and_base():
    plan = stage_plan(START, GOAL, never)
    steps = [moved(a, b) for a, b in zip([START, *plan.poses], plan.poses)]
    assert steps == [{"arm_joint3"}, {"arm_joint4", "arm_joint6"}, {"arm_joint1", "arm_joint2"}]


def test_the_plan_ends_at_the_goal():
    plan = stage_plan(START, GOAL, never)
    assert np.array_equal(plan.poses[-1], GOAL)


def test_a_refused_first_stage_tries_the_stages_in_reverse():
    def no_elbow_first(a: np.ndarray, b: np.ndarray) -> str | None:
        return "kolizja: test" if a[2] != b[2] and a[0] == START[0] else None

    plan = stage_plan(START, GOAL, no_elbow_first)
    assert moved(START, plan.poses[0]) == {"arm_joint1", "arm_joint2"}


def test_with_every_segment_refused_the_plan_names_the_first_collision():
    plan = stage_plan(START, GOAL, lambda a, b: "kolizja: test")
    assert (plan.poses, plan.collision) == ((), "kolizja: test")


def test_a_start_at_the_goal_needs_no_pose():
    assert stage_plan(GOAL, GOAL, never).poses == ()


def test_the_plan_from_home_to_a_training_hover_is_clear_of_the_table_and_the_object(kin, box):
    checker = CollisionChecker()
    cube = Box(center=(0.0, 0.33, 0.015), half=(0.015, 0.015, 0.015), name="obiekt")
    hv = solve_hover(kin, LEFT, 0.0, box)
    plan = stage_plan(kin.home, hv.q, lambda a, b: checker.path_clear(a, b, (cube,), opening=0.0))
    assert plan.collision is None
