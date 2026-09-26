"""galaxeo.arm collision checks and the safe reach box."""

from __future__ import annotations

import itertools
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo.arm import Box, CollisionChecker, Kinematics, ReachBox, Sphere  # noqa: E402
from galaxeo.arm import reach as R  # noqa: E402


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def checker():
    return CollisionChecker()


def solve(kin, p):
    sol = kin.ik(p, seed=kin.home)
    assert sol.ok
    return sol.q


# ---- collision ------------------------------------------------------------------------------
def test_home_and_the_folded_rest_pose_are_clear(kin, checker):
    assert checker.config_clear(kin.home) is None
    assert checker.config_clear(np.zeros(6)) is None


def test_the_table_is_the_plane_z_0(kin, checker):
    assert checker.config_clear(solve(kin, [0.35, 0.0, 0.08])) is None
    why = checker.config_clear(solve(kin, [0.35, 0.0, -0.01]))
    assert why is not None and "table" in why


def test_self_collision(checker):
    why = checker.config_clear([-0.68, 1.58, -3.26, -0.02, 1.48, -1.24])
    assert why is not None and "arm_link" in why and "table" not in why


def test_caller_obstacles_boxes_and_spheres(kin, checker):
    q = solve(kin, [0.35, 0.0, 0.10])
    tip = kin.fk(q)[:3, 3]
    assert checker.config_clear(q) is None
    why = checker.config_clear(q, [Sphere(tuple(tip), 0.06, "cup")])
    assert why is not None and "cup" in why
    shelf = Box((0.35, 0.0, 0.20), (0.05, 0.30, 0.02), "shelf")
    assert "shelf" in checker.config_clear(q, [shelf])
    assert checker.config_clear(q, [Sphere((0.0, 0.5, 0.1), 0.05)]) is None
    # obstacles do not leak into later checks
    assert checker.config_clear(q) is None


def test_path_is_checked_not_only_the_goal(kin, checker):
    a = solve(kin, [0.30, -0.20, 0.10])
    b = solve(kin, [0.30, 0.20, 0.10])
    assert checker.config_clear(a) is None and checker.config_clear(b) is None
    assert checker.path_clear(a, b) is None
    post = Box((0.30, 0.0, 0.10), (0.02, 0.02, 0.10), "post")
    assert checker.config_clear(a, [post]) is None and checker.config_clear(b, [post]) is None
    why = checker.path_clear(a, b, [post])
    assert why is not None and "post" in why and "of the way" in why


# ---- reach ----------------------------------------------------------------------------------
def test_every_grid_point_of_the_box_is_reachable_and_collision_free(kin, checker):
    box = ReachBox.load()
    assert box.lo[2] >= R.MIN_Z
    points = box.grid()
    corners = {tuple(np.round(c, 6)) for c in itertools.product(*zip(box.lo, box.hi))}
    assert corners <= {tuple(np.round(p, 6)) for p in points}
    bad = [(p.round(3).tolist(), why) for p in points if (why := R.check_point(p, kin, checker))]
    assert bad == []


def test_a1x_numbers_and_why_the_front_starts_at_0_20():
    box = ReachBox.load()
    assert box.lo.tolist() == pytest.approx([0.20, -0.30, 0.05])
    assert box.hi.tolist() == pytest.approx([0.50, 0.30, 0.30])
    computed, log = R.compute()
    assert computed.lo.tolist() == pytest.approx(box.lo.tolist())
    assert computed.hi.tolist() == pytest.approx(box.hi.tolist())
    assert "lo[x]" in log[0] and "0.18" in log[0] and "table" in log[0]


def test_the_cache_is_current():
    with open(R.CACHE) as f:
        entries = json.load(f)
    assert R.cache_key(R.GRIPPER) in entries, "reach.json is stale: delete it and call ReachBox.load()"


def test_cache_is_written_for_a_new_tool(tmp_path):
    from galaxeo.arm import Tool

    path = tmp_path / "reach.json"
    tool = Tool(pos=(0.14, 0.0, 0.0))
    box = ReachBox.load(tool, path=str(path))
    data = json.loads(path.read_text())
    assert R.cache_key(tool) in data
    assert ReachBox.load(tool, path=str(path)).lo.tolist() == box.lo.tolist()


def test_contains_margin_and_clamp():
    box = ReachBox(np.array([0.2, -0.3, 0.05]), np.array([0.5, 0.3, 0.3]))
    assert box.contains([0.35, 0.0, 0.1])
    assert box.contains([0.2, -0.3, 0.05])
    assert not box.contains([0.19, 0.0, 0.1])
    assert not box.contains([0.21, 0.0, 0.1], margin=0.02)
    assert box.contains([0.19, 0.0, 0.1], margin=-0.02)
    assert not box.contains([np.nan, 0.0, 0.1])
    assert box.clamp([0.0, 1.0, 0.1]).tolist() == pytest.approx([0.2, 0.3, 0.1])
    assert box.center.tolist() == pytest.approx([0.35, 0.0, 0.175])
    with pytest.raises(ValueError):
        ReachBox(np.array([0.5, 0, 0]), np.array([0.2, 1, 1]))
    with pytest.raises(ValueError):
        ReachBox.from_dict({"lo": [0, 0, 0]})
