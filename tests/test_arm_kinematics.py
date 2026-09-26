"""galaxeo.arm model and kinematics: the packaged MJCF, the tool frame, FK against the URDF, IK."""

from __future__ import annotations

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import Kinematics, Tool  # noqa: E402
from galaxeo.arm.model import MJCF, build, indices  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def random_poses(kin, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(kin.lo, kin.hi) for _ in range(n)]


def test_packaged_model_is_the_generator_output(tmp_path):
    gen = _load(os.path.join(ROOT, "sim", "urdf2mjcf.py"), "urdf2mjcf")
    out = tmp_path / "a1x.xml"
    gen.main(str(out), "meshes")
    with open(MJCF) as f:
        assert out.read_text() == f.read(), "rerun: python sim/urdf2mjcf.py --out galaxeo/arm/assets/a1x.xml --meshdir meshes"


def test_limits_come_from_the_model_and_match_the_protocol_table(kin):
    assert np.allclose(kin.lo, [lo for lo, _ in P.URDF_LIMITS_RAD], atol=1e-3)
    assert np.allclose(kin.hi, [hi for _, hi in P.URDF_LIMITS_RAD], atol=1e-3)


def test_default_tool_is_the_gripper_tcp_site(kin):
    import mujoco

    m = kin.model
    d = mujoco.MjData(m)
    ix = indices(m)
    tcp = m.site("tcp").id
    for q in random_poses(kin, 20):
        d.qpos[ix.qadr] = q
        mujoco.mj_kinematics(m, d)
        T = kin.fk(q)
        assert np.allclose(T[:3, 3], d.site_xpos[tcp], atol=1e-9)
        assert np.allclose(T[:3, :3], d.site_xmat[tcp].reshape(3, 3), atol=1e-9)


def test_fk_agrees_with_the_urdf_chain(kin):
    """The numpy URDF chain (kinematics.py) and MuJoCo on the generated MJCF are the same arm."""
    chain_mod = _load(os.path.join(ROOT, "kinematics.py"), "urdf_kinematics")
    urdf = os.path.join(ROOT, "ros2_ws", "src", "galaxea_a1xy_description", "urdf", "a1x.urdf")
    chain = chain_mod.Chain(urdf, [f"arm_joint{i}" for i in range(1, 7)])
    tool = np.eye(4)
    tool[:3, 3] = Tool().pos
    for q in random_poses(kin, 30, seed=1):
        assert np.allclose(chain.fk(q) @ tool, kin.fk(q), atol=1e-6)


def test_a_custom_tool_moves_the_tool_frame():
    tool = Tool(pos=(0.20, 0.0, 0.02))
    a, b = Kinematics(), Kinematics(tool)
    q = a.home
    link6 = a.fk(q) @ np.linalg.inv(Tool().matrix())
    assert np.allclose(b.fk(q), link6 @ tool.matrix(), atol=1e-9)
    sol = b.ik([0.35, 0.05, 0.12], seed=b.home)
    assert sol.ok and np.linalg.norm(b.fk(sol.q)[:3, 3] - [0.35, 0.05, 0.12]) < 0.003


def test_ik_round_trip_within_limits_and_reports_the_residual(kin):
    rng = np.random.default_rng(2)
    n_ok = 0
    for _ in range(40):
        q = np.clip(kin.home + rng.uniform(-0.6, 0.6, 6), kin.lo, kin.hi)
        p = kin.fk(q)[:3, 3]
        if p[2] < 0.03:
            continue
        sol = kin.ik(p, seed=kin.home)
        assert np.all(sol.q >= kin.lo - 1e-12) and np.all(sol.q <= kin.hi + 1e-12)
        assert sol.pos_err == pytest.approx(np.linalg.norm(kin.fk(sol.q)[:3, 3] - p), abs=1e-9)
        assert sol.ok and sol.pos_err < 0.003
        n_ok += 1
    assert n_ok >= 20


def test_ik_out_of_reach_says_so(kin):
    sol = kin.ik([1.5, 0.0, 0.3])
    assert not sol.ok and sol.pos_err > 0.5
    assert np.all(sol.q >= kin.lo) and np.all(sol.q <= kin.hi)


def test_orientation_is_secondary_to_position(kin):
    from galaxeo.arm import down_rotation

    p = np.array([0.35, 0.1, 0.12])
    sol = kin.ik(p, down_rotation(p), seed=kin.home)
    assert sol.ok and sol.rot_err < np.radians(5)
    approach = kin.fk(sol.q)[:3, 0]
    assert approach[2] < -0.99
    # An unreachable rotation (tool pointing up, low over the table) still lands on the point.
    up = -down_rotation(p)
    up[:, 1] *= -1
    sol = kin.ik(p, up, seed=kin.home)
    assert sol.ok


def test_build_adds_the_table_plane_and_obstacles():
    import mujoco

    from galaxeo.arm import Box, Sphere

    m = build(obstacles=[Box((0.3, 0, 0.1), (0.02, 0.02, 0.02)), Sphere((0.3, 0.1, 0.1), 0.03, "cup")])
    assert m.geom("table").type == mujoco.mjtGeom.mjGEOM_PLANE
    assert m.geom("obstacle0").type == mujoco.mjtGeom.mjGEOM_BOX
    assert m.geom("obstacle1 cup").size[0] == pytest.approx(0.03)
