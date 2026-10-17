"""galaxeo.arm.stages: the staged joint-space plan, with a collision check of the caller's."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")
yaml = pytest.importorskip("yaml")

from galaxeo.arm import Box, CollisionChecker, Kinematics, down_rotation, stage_plan  # noqa: E402
from galaxeo.arm.stages import STAGES  # noqa: E402

ROBOT_YAML = Path(__file__).resolve().parents[1] / "galaxeo" / "robot" / "a1x" / "robot.yaml"
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
        return "collision: test" if a[2] != b[2] and a[0] == START[0] else None

    plan = stage_plan(START, GOAL, no_elbow_first)
    assert moved(START, plan.poses[0]) == {"arm_joint1", "arm_joint2"}


def test_with_both_orders_refused_the_plan_is_one_straight_move():
    def staged_never(a: np.ndarray, b: np.ndarray) -> str | None:
        return "collision: test" if len(moved(a, b)) < 5 else None

    plan = stage_plan(START, GOAL, staged_never)
    assert plan.collision is None
    assert len(plan.poses) == 1
    assert np.array_equal(plan.poses[0], GOAL)


def test_with_every_segment_refused_the_plan_names_the_first_collision():
    plan = stage_plan(START, GOAL, lambda a, b: "collision: test")
    assert (plan.poses, plan.collision) == ((), "collision: test")


def test_a_start_at_the_goal_needs_no_pose():
    assert stage_plan(GOAL, GOAL, never).poses == ()


def test_the_plan_from_home_to_a_pose_over_an_object_is_clear_of_the_table_and_the_object():
    kin = Kinematics()
    checker = CollisionChecker()
    point = [0.0, 0.33, 0.10]
    goal = kin.ik(point, down_rotation(point), seed=kin.home)
    cube = Box(center=(0.0, 0.33, 0.015), half=(0.015, 0.015, 0.015), name="object")

    plan = stage_plan(kin.home, goal.q, lambda a, b: checker.path_clear(a, b, (cube,), opening=0.0))

    assert goal.ok
    assert plan.collision is None
    assert np.array_equal(plan.poses[-1], goal.q)
