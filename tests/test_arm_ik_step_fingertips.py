"""galaxeo.arm: the warm IK step, the caller's joint box, body poses and fingertips.

Everything runs on the installation model alone; the twin's model galaxeo/robot/a1x/a1x.xml names the
fingertip boxes and serves as the reference for finding them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")

from galaxeo.arm import GRIPPER, IKStep, Kinematics, joint_box  # noqa: E402

ROBOT_MJCF = Path(__file__).resolve().parents[1] / "galaxeo" / "robot" / "a1x" / "a1x.xml"
J6 = 5

# A top-down pose on the arm's left.
HOVER = np.radians([90.0, 120.0, -77.0, 47.0, 0.0, 0.0])

# The generator's layout (sim/urdf2mjcf.py): finger body 0.03689 m out of gripper_link, its curled tip box
# centred 0.038 m further with a half-length of 0.004 m, the tool point 0.045 m out of gripper_link.
TIP_BELOW_TOOL = 0.03689 + 0.038 + 0.004 - 0.045
TOOL_OUT_OF_GRIPPER_LINK = 0.045


@dataclass(frozen=True)
class PoseError:
    pos: float           # [m]
    rot: float           # [rad]


@dataclass(frozen=True)
class Case:
    seed: np.ndarray
    position: np.ndarray
    rotation: np.ndarray


def pose_error(kin: Kinematics, q: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> PoseError:
    """Tool error at q from forward kinematics: the quaternion difference as a rotation vector."""
    T = kin.fk(q)
    want = np.zeros(4)
    cur = np.zeros(4)
    err = np.zeros(4)
    w = np.zeros(3)
    mujoco.mju_mat2Quat(want, np.ascontiguousarray(rotation, float).ravel())
    mujoco.mju_mat2Quat(cur, np.ascontiguousarray(T[:3, :3]).ravel())
    mujoco.mju_negQuat(cur, cur)
    mujoco.mju_mulQuat(err, want, cur)
    mujoco.mju_quat2Vel(w, err, 1.0)
    return PoseError(pos=float(np.linalg.norm(position - T[:3, 3])), rot=float(np.linalg.norm(w)))


def warm_cases(kin: Kinematics, n: int = 50) -> list[Case]:
    """Targets at random joints inside the model's range, each with a seed up to 0.2 rad per joint away."""
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(n):
        q = rng.uniform(kin.lo + 0.2, kin.hi - 0.2)
        T = kin.fk(q)
        cases.append(Case(seed=q + rng.uniform(-0.2, 0.2, 6), position=T[:3, 3].copy(), rotation=T[:3, :3].copy()))
    return cases


def with_j6(q: np.ndarray, deg: float) -> np.ndarray:
    out = q.copy()
    out[J6] = np.radians(deg)
    return out


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


# ---- body poses ------------------------------------------------------------------------------
def test_link_pose_is_a_rigid_transform_with_a_bottom_row(kin):
    T = kin.link_pose(HOVER, "arm_link6")

    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])


def test_the_last_link_carries_the_tool_frame(kin):
    for case in warm_cases(kin, 20):
        assert np.allclose(kin.link_pose(case.seed, "arm_link6") @ GRIPPER.matrix(), kin.fk(case.seed), atol=1e-12)


def test_the_tool_point_is_0_045_m_out_along_gripper_link_x(kin):
    for case in warm_cases(kin, 20):
        T = kin.link_pose(case.seed, "gripper_link")

        assert np.allclose(T[:3, 3] + TOOL_OUT_OF_GRIPPER_LINK * T[:3, 0], kin.fk(case.seed)[:3, 3], atol=1e-12)


# ---- ik_step ---------------------------------------------------------------------------------
@pytest.mark.parametrize("iters", [60, 1])
def test_reported_errors_are_measured_at_the_returned_joints(kin, iters):
    unconverged = 0
    for case in warm_cases(kin):
        step = kin.ik_step(case.position, case.rotation, case.seed, iters=iters)
        want = pose_error(kin, step.q, case.position, case.rotation)
        unconverged += not step.converged

        assert step.pos_err == pytest.approx(want.pos, abs=1e-12)
        assert step.rot_err == pytest.approx(want.rot, abs=1e-12)

    if iters == 1:
        assert unconverged > 0                 # the errors recomputed after the last update are checked too


def test_warm_seeds_converge_inside_the_tolerances(kin):
    steps = [kin.ik_step(c.position, c.rotation, c.seed) for c in warm_cases(kin)]

    assert all(s.converged for s in steps)
    assert max(s.pos_err for s in steps) < 1e-3
    assert max(s.rot_err for s in steps) < 5e-3


def test_the_jump_is_the_largest_joint_change_from_the_seed(kin):
    for case in warm_cases(kin):
        step = kin.ik_step(case.position, case.rotation, case.seed)

        assert step.jump == pytest.approx(float(np.max(np.abs(step.q - case.seed))), abs=1e-15)


def test_a_seed_that_meets_the_tolerances_takes_no_update(kin):
    T = kin.fk(HOVER)
    step = kin.ik_step(T[:3, 3], T[:3, :3], HOVER)

    assert isinstance(step, IKStep)
    assert step.converged
    assert step.iters == 0
    assert step.jump == 0.0
    assert np.array_equal(step.q, HOVER)


def test_no_iterations_is_not_converged_even_at_the_target(kin):
    T = kin.fk(HOVER)
    step = kin.ik_step(T[:3, 3], T[:3, :3], HOVER, iters=0)

    assert not step.converged
    assert step.iters == 0
    assert step.pos_err == pytest.approx(0.0, abs=1e-15)


def test_a_last_update_that_lands_inside_the_tolerances_is_not_converged(kin):
    T = kin.fk(HOVER)
    target = T[:3, 3] + np.array([0.003, 0.0, 0.0])
    full = kin.ik_step(target, T[:3, :3], HOVER)
    last = kin.ik_step(target, T[:3, :3], HOVER, iters=1)

    assert full.converged
    assert full.iters == 1                     # one update is enough; the tolerance test after it says so
    assert not last.converged
    assert last.iters == 1
    assert last.pos_err < 1e-3
    assert last.rot_err < 5e-3
    assert np.array_equal(last.q, full.q)


def test_running_out_of_iterations_is_not_converged_and_counts_every_update(kin):
    T = kin.fk(HOVER)
    step = kin.ik_step([1.5, 0.0, 0.3], T[:3, :3], HOVER, iters=25)

    assert not step.converged
    assert step.iters == 25
    assert step.pos_err > 0.5


def test_one_update_towards_a_far_target_moves_the_joints_by_the_0_3_rad_cap(kin):
    T = kin.fk(HOVER)
    far = kin.ik_step(T[:3, 3] + np.array([-0.15, 0.10, 0.10]), T[:3, :3], HOVER, iters=1)

    assert np.linalg.norm(far.q - HOVER) == pytest.approx(0.3, abs=1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_lo_hi_narrow_the_model_range_and_refuse_a_step_beyond_them(kin, sign):
    seed = with_j6(HOVER, sign * 158.0)
    T = kin.fk(with_j6(HOVER, sign * 163.0))
    lo = kin.lo.copy()
    hi = kin.hi.copy()
    lo[J6] = np.radians(-160.0)
    hi[J6] = np.radians(160.0)

    free = kin.ik_step(T[:3, 3], T[:3, :3], seed)
    narrowed = kin.ik_step(T[:3, 3], T[:3, :3], seed, lo=lo, hi=hi)

    assert free.converged
    assert abs(np.degrees(free.q[J6])) > 162.0
    assert not narrowed.converged
    assert abs(np.degrees(narrowed.q[J6])) <= 160.0 + 1e-9


def test_lo_hi_wider_than_the_model_leave_the_model_range_in_force(kin):
    seed = with_j6(HOVER, 163.0)
    T = kin.fk(with_j6(HOVER, 168.0))
    hi = kin.hi.copy()
    hi[J6] = np.radians(200.0)

    step = kin.ik_step(T[:3, 3], T[:3, :3], seed, hi=hi)

    assert not step.converged
    assert step.q[J6] <= kin.hi[J6]


def test_the_jump_counts_from_the_seed_as_given_before_clipping(kin):
    seed = with_j6(HOVER, 163.0)
    T = kin.fk(seed)
    hi = kin.hi.copy()
    hi[J6] = np.radians(160.0)

    clipped = kin.ik_step(T[:3, 3], T[:3, :3], seed, iters=0, hi=hi)

    assert np.degrees(clipped.q[J6]) == pytest.approx(160.0, abs=1e-9)
    assert np.degrees(clipped.jump) == pytest.approx(3.0, abs=1e-9)


def test_lo_above_hi_is_refused(kin):
    T = kin.fk(HOVER)
    lo = kin.lo.copy()
    lo[0] = kin.hi[0] + 0.1

    with pytest.raises(ValueError, match="above hi"):
        kin.ik_step(T[:3, 3], T[:3, :3], HOVER, lo=lo)


# ---- the caller's joint box ------------------------------------------------------------------
def test_a_joint_box_takes_degrees_and_narrows_the_model_range(kin):
    box = joint_box(kin, {"arm_joint6": (-160.0, 40.0)})

    assert box.lo[J6] == pytest.approx(np.radians(-160.0))
    assert box.hi[J6] == pytest.approx(np.radians(40.0))


def test_a_limit_wider_than_the_model_is_cut_to_the_model(kin):
    box = joint_box(kin, {"arm_joint6": (-400.0, 400.0)})

    assert box.lo[J6] == kin.lo[J6]
    assert box.hi[J6] == kin.hi[J6]


def test_a_joint_the_caller_does_not_name_keeps_the_model_range(kin):
    box = joint_box(kin, {"arm_joint6": (-160.0, 40.0), "gripper": (0.0, 100.0)})

    assert np.array_equal(box.lo[:J6], kin.lo[:J6])
    assert np.array_equal(box.hi[:J6], kin.hi[:J6])


def test_a_step_beyond_the_joint_box_comes_back_unconverged(kin):
    box = joint_box(kin, {"arm_joint6": (-160.0, 160.0)})
    T = kin.fk(with_j6(HOVER, 163.0))

    step = kin.ik_step(T[:3, 3], T[:3, :3], with_j6(HOVER, 158.0), lo=box.lo, hi=box.hi)

    assert not step.converged
    assert np.degrees(step.q[J6]) <= 160.0 + 1e-9


# ---- fingertips ------------------------------------------------------------------------------
def test_the_fingertips_are_the_boxes_the_twin_model_names_finger1_tip_and_finger2_tip(kin):
    named = mujoco.MjModel.from_xml_path(str(ROBOT_MJCF))
    for g, name in zip(kin.ix.tips, ("finger1_tip", "finger2_tip")):
        tip = named.geom(name)

        assert kin.model.body(int(kin.model.geom_bodyid[g])).name == named.body(int(tip.bodyid[0])).name
        assert np.array_equal(kin.model.geom_pos[g], tip.pos)
        assert np.array_equal(kin.model.geom_size[g], tip.size)


def test_pointing_down_the_fingertips_reach_below_the_tool_point_by_the_generators_layout(kin):
    tool_z = kin.fk(HOVER)[2, 3]
    tips = kin.fingertips(HOVER)

    assert tips.shape == (2, 3)
    assert tips[0, 2] == pytest.approx(tool_z - TIP_BELOW_TOOL, abs=1e-9)
    assert tips[1, 2] == pytest.approx(tool_z - TIP_BELOW_TOOL, abs=1e-9)


def test_the_opening_slides_each_fingertip_along_the_closing_axis(kin):
    closing = kin.fk(HOVER)[:3, 1]
    closed = kin.fingertips(HOVER, 0.0)
    wide = kin.fingertips(HOVER, 0.05)

    assert np.allclose(wide[0] - closed[0], 0.05 * closing, atol=1e-12)
    assert np.allclose(wide[1] - closed[1], -0.05 * closing, atol=1e-12)


def test_an_open_finger_rolled_5_deg_by_joint_5_sits_lower_than_a_closed_one(kin):
    rolled = HOVER.copy()
    rolled[4] = np.radians(5.0)
    tool_z = kin.fk(rolled)[2, 3]

    closed_below = tool_z - kin.fingertips(rolled, 0.0)[:, 2].min()
    open_below = tool_z - kin.fingertips(rolled, 0.05)[:, 2].min()

    assert closed_below - TIP_BELOW_TOOL < 0.001
    assert open_below - TIP_BELOW_TOOL > 0.004


@pytest.mark.parametrize("opening", [-0.001, 0.051, float("nan")])
def test_an_opening_outside_the_finger_range_is_refused(kin, opening):
    with pytest.raises(ValueError, match="finger opening"):
        kin.fingertips(HOVER, opening)
