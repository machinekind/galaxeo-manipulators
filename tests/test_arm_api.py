"""Arm.plan: accepted, or refused in the order outside volume, ik, jump, collision."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo.arm import Arm, Box, BusTransport, Sphere, down_rotation  # noqa: E402
from galaxeo.arm.fake import FakeA1XBus  # noqa: E402


@pytest.fixture(scope="module")
def arm():
    return Arm()


@pytest.fixture(scope="module")
def center(arm):
    return arm.kin.ik(arm.reach.center, seed=arm.kin.home).q


def test_accepted_plan_lands_the_tool_on_the_point(arm, center):
    p = arm.reach.center + np.array([0.05, -0.05, 0.02])
    plan = arm.plan(p, start=center)
    assert plan and plan.state == "accepted" and plan.reason == ""
    assert np.linalg.norm(arm.fk(plan.joints)[:3, 3] - p) < 0.003
    assert plan.ik.pos_err < 0.003


def test_a_pose_target_points_the_tool_down(arm, center):
    p = arm.reach.center
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = down_rotation(p), p
    plan = arm.plan(T, start=center)
    assert plan, plan
    assert arm.fk(plan.joints)[2, 0] < -0.99


def test_refusal_order(arm, center):
    # outside the box comes first, even for a point IK could not reach either
    assert arm.plan([1.5, 0.0, 0.3], start=center).reason == "outside volume"
    assert arm.plan([0.35, 0.0, 0.02], start=center).reason == "outside volume"
    # ik: only when the box check is skipped
    plan = arm.plan([1.5, 0.0, 0.3], start=center, check_reach=False)
    assert plan.reason == "ik" and "mm" in plan.detail and plan.ik.pos_err > 0.5
    # jump: every point in the box is more than 45 deg from the folded rest pose
    plan = arm.plan(arm.reach.center, start=np.zeros(6))
    assert plan.reason == "jump" and plan.joints is not None
    # collision: the path from the start passes through an obstacle
    a = arm.plan(arm.reach.center + [0.0, -0.15, 0.0], start=center).joints
    b_point = arm.reach.center + [0.0, 0.15, 0.0]
    post = Box((float(arm.reach.center[0]), 0.0, 0.10), (0.02, 0.02, 0.10), "post")
    plan = arm.plan(b_point, start=a, obstacles=[post])
    assert plan.reason == "collision" and "post" in plan.detail
    assert arm.plan(b_point, start=a)


def test_obstacle_at_the_goal(arm, center):
    p = arm.reach.center
    plan = arm.plan(p, start=center, obstacles=[Sphere(tuple(p), 0.06, "cup")])
    assert (plan.state, plan.reason) == ("rejected", "collision")


def test_plan_starts_from_the_measured_pose(center):
    bus = FakeA1XBus(start_deg=np.degrees(center))
    bus.tick(n=5)
    arm = Arm(BusTransport(bus, clock=bus.now, sleep=bus.sleep))
    assert np.allclose(arm.measured(), center, atol=1e-3)
    plan = arm.plan(arm.reach.center + [0.0, 0.05, 0.0])
    assert plan
    assert math.degrees(np.max(np.abs(plan.joints - center))) < 20.0
    assert bus.tx == []
