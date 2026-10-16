"""The twin's A1X model: the six post-processing rules, byte-for-byte regeneration, and robot.yaml against it.

`import_twin_model.py` generates `galaxeo/robot/a1x/a1x.xml` from the vendor URDF; each rule has a test
here, so the description and the model cannot drift apart.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

import import_twin_model  # noqa: E402

from galaxeo.robot import ROBOT_DIR  # noqa: E402

SPEC = yaml.safe_load((ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
MODEL_PATH = ROBOT_DIR / SPEC["mjcf"]
#: The options the twin's scene copies from the arm model before it attaches it (`twin/scene.py`).
SCENE_OPTIONS = ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def names(model: mujoco.MjModel, kind: str, count: int) -> list[str]:
    return [getattr(model, kind)(i).name for i in range(count)]


def test_the_script_reproduces_the_committed_model_byte_for_byte(tmp_path):
    out = tmp_path / "a1x.xml"

    assert import_twin_model.main(["--out", str(out)]) == 0
    assert out.read_bytes() == MODEL_PATH.read_bytes()


def test_the_model_takes_the_meshes_galaxeo_arm_ships(model):
    meshdir = ET.parse(MODEL_PATH).getroot().find("compiler").get("meshdir")
    meshes = ROBOT_DIR.parent.parent / "arm" / "assets" / "meshes"

    assert (ROBOT_DIR / meshdir).resolve() == meshes.resolve()
    assert model.nmesh == 10


def test_the_model_attaches_into_a_scene_the_way_the_twin_builds_it():
    robot = mujoco.MjSpec.from_file(str(MODEL_PATH))
    scene = mujoco.MjSpec()
    for option in SCENE_OPTIONS:
        setattr(scene.option, option, getattr(robot.option, option))

    scene.attach(robot, prefix="a1x/", frame=scene.worldbody.add_frame())
    compiled = scene.compile()

    assert compiled.nmesh == 10
    assert compiled.body("a1x/gripper_link").id > 0


def test_the_home_keyframe_holds_still_and_is_unchanged(model):
    key = model.key("home")
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, key.id)
    mujoco.mj_forward(model, data)

    assert model.nbody == 11                                       # world + the 10 URDF links
    assert key.qpos == pytest.approx([0.0, 1.0, -1.6, 0.6, 0.0, 0.0, 0.03, -0.03])
    assert key.ctrl == pytest.approx([0.0, 1.0, -1.6, 0.6, 0.0, 0.0, 0.03])
    assert np.isfinite(data.qacc).all()


def test_rule_1_the_first_jaw_joint_is_the_gripper(model):
    assert names(model, "joint", model.njnt) == [*SPEC["joints"], "gripper_finger_joint2"]


def test_rule_2_actuators_are_named_after_their_joints(model):
    actuators = names(model, "actuator", model.nu)
    driven = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]

    assert actuators == SPEC["joints"]
    assert driven == actuators


@pytest.mark.parametrize("tip, jaw", [("finger1_tip", "gripper_finger_link1"), ("finger2_tip", "gripper_finger_link2")])
def test_rule_3_the_curled_jaw_tips_are_named_boxes(model, tip, jaw):
    geom = model.geom(tip)

    assert model.body(model.geom_bodyid[geom.id]).name == jaw
    assert model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_BOX


def test_rule_4_the_physics_options(model):
    assert model.opt.timestep == 0.002
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_rule_5_the_tcp_site_is_not_rotated(model):
    site = model.site(SPEC["tcp_site"])

    assert site.pos == pytest.approx([0.045, 0.0, 0.0])
    assert model.site_quat[site.id] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert model.body(model.site_bodyid[site.id]).name == SPEC["hand_body"]


def test_rule_6_the_jaw_equality_is_stiffened(model):
    pair = {model.joint(model.eq_obj1id[0]).name, model.joint(model.eq_obj2id[0]).name}

    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert pair == {"gripper_finger_joint2", "gripper"}
    assert model.eq_solref[0] == pytest.approx([0.01, 1.0])
    assert model.eq_solimp[0] == pytest.approx([0.99, 0.999, 0.001, 0.5, 2.0])


def test_robot_yaml_mirror_joint_is_the_model_equality(model):
    mirrored = "gripper_finger_joint2"
    source = SPEC["mirror_joints"][mirrored][0]
    gain = SPEC["mirror_joints"][mirrored][1]

    assert list(SPEC["mirror_joints"]) == [mirrored]
    assert model.joint(model.eq_obj1id[0]).name == mirrored
    assert model.joint(model.eq_obj2id[0]).name == source
    # polycoef: q1 = c0 + c1 * q2 + ... -> 0 - 1 * q2
    assert model.eq_data[0, :5] == pytest.approx([0.0, gain, 0.0, 0.0, 0.0])


def test_robot_yaml_names_only_what_the_model_has(model):
    joints = set(names(model, "joint", model.njnt))
    bodies = set(names(model, "body", model.nbody))
    geoms = set(names(model, "geom", model.ngeom))
    sites = set(names(model, "site", model.nsite))

    assert set(SPEC["joints"]) <= joints
    assert SPEC["gripper"] in joints
    assert SPEC["roll_joint"] in joints
    assert set(SPEC["fingertips"]) <= geoms
    assert SPEC["tcp_site"] in sites
    assert SPEC["hand_body"] in bodies
    assert SPEC["base_body"] in bodies
    assert set(SPEC["jaw_bodies"]) <= bodies


def test_robot_yaml_gripper_range_is_the_jaw_travel(model):
    jaw = model.joint(SPEC["gripper"])

    assert SPEC["gripper_range_q"] == pytest.approx(list(model.jnt_range[jaw.id]))
