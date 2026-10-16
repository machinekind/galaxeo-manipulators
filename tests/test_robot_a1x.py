"""The twin's `a1x` backend without hardware: the adapter onto `galaxeo.a1x.A1X`, driven like the twin drives it.

What these check is the twin's side: the context and its `a1x:` settings, joint names, degrees and the
gripper 0..100 at the boundary, faults named the way the twin pins them, the entry points and the package
data, plus end-to-end runs through `A1XArm` on the fake bus. The driver's own tests are in `test_a1x.py`.
"""

from __future__ import annotations

import importlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from a1x_fakes import (CLOSED, GRIP_SLEW, LSB, LSB_DEG, OPEN, OPEN_LOOP, START_DEG, STEP, Clock, FakeA1XBus,
                       gripper_pdes, ticking_sleep)

from galaxeo import protocol as P
from galaxeo.robot.backend import A1XArm, A1XConfig, BackendInfo, create

REPO = Path(__file__).resolve().parents[1]
ARM = tuple(f"arm_joint{i}" for i in range(1, 7))


@dataclass(frozen=True)
class JointRange:
    min: float
    max: float
    max_vel: float


#: A1X joints in app units: the tests' own limits, not robot.yaml's.
A1X_JOINTS = {
    "arm_joint1": JointRange(min=-160.0, max=160.0, max_vel=90.0),
    "arm_joint2": JointRange(min=-2.0, max=175.0, max_vel=90.0),
    "arm_joint3": JointRange(min=-185.0, max=2.0, max_vel=90.0),
    "arm_joint4": JointRange(min=-88.0, max=88.0, max_vel=120.0),
    "arm_joint5": JointRange(min=-88.0, max=88.0, max_vel=120.0),
    "arm_joint6": JointRange(min=-160.0, max=160.0, max_vel=150.0),
    "gripper": JointRange(min=0.0, max=100.0, max_vel=300.0),
}


@dataclass(frozen=True)
class Context:
    """What the twin hands a backend factory (its `BackendContext`), as plain data."""

    robot: str
    joints: tuple[str, ...]
    gripper: str | None
    limits: Mapping[str, tuple[float, float]]
    max_vel: Mapping[str, float]
    port: str | None
    settings: Mapping[str, Any]


def context(port: str | None = None, **settings: Any) -> Context:
    """The `a1x` backend's context: the test joints, `port` and the `a1x:` section `settings`."""
    return Context(robot="a1x", joints=tuple(A1X_JOINTS), gripper="gripper",
                   limits={n: (j.min, j.max) for n, j in A1X_JOINTS.items()},
                   max_vel={n: j.max_vel for n, j in A1X_JOINTS.items()}, port=port, settings=settings)


@dataclass
class Rig:
    arm: A1XArm
    bus: FakeA1XBus
    clock: Clock

    def run(self, secs: float, read: bool = True) -> None:
        for _ in range(int(round(secs / STEP))):
            self.clock.t += STEP
            self.bus.tick(STEP)
            self.arm.io_step(self.clock.t)
            if read:
                self.arm.read_joints()


def make_rig(bus: FakeA1XBus | None = None, **a1x: Any) -> Rig:
    clock = Clock()
    bus = bus or FakeA1XBus()
    arm = A1XArm(context(**a1x), bus=bus, clock=clock, threaded=False, sleep=ticking_sleep(clock, bus),
                 connect_timeout_s=0.05)
    return Rig(arm=arm, bus=bus, clock=clock)


def connected(**kw: Any) -> Rig:
    rig = make_rig(**kw)
    rig.bus.tick(n=60)
    rig.arm.connect()
    return rig


def connected_open_loop(**kw: Any) -> Rig:
    """As `connected`, but the gripper without feedback (`gripper_fb_*` = None)."""
    return connected(**OPEN_LOOP, **kw)


def fault_joint(fault: str) -> str | None:
    """The joint the twin pins a fault on: the text before the first ':' or ' (' (a wire contract)."""
    head = fault.split(":", 1)[0].split(" (", 1)[0].strip()
    return head if head in (*ARM, "gripper") else None


def test_degrees_names_and_zero_offsets_convert_at_the_boundary():
    rig = connected(joint_offsets_deg=[0.0, 0.0, 1.49, 0.0, 0.0, 0.0])
    joints = rig.arm.read_joints()
    assert list(joints) == [*ARM, "gripper"]
    assert joints["arm_joint1"] == pytest.approx(START_DEG[0], abs=LSB_DEG)
    assert joints["arm_joint3"] == pytest.approx(START_DEG[2] - 1.49, abs=LSB_DEG)

    # radians(-120.0) does not come back as -120.0: the backend returns the degrees it clamped.
    assert math.degrees(math.radians(-120.0)) != -120.0
    assert rig.arm.send_joints({"arm_joint1": -120.0, "arm_joint3": -50.0}) == {"arm_joint1": -120.0,
                                                                                 "arm_joint3": -50.0}
    assert rig.arm.goal_positions() == {"arm_joint1": -120.0, "arm_joint3": -50.0}
    rig.run(2.0)
    frame = rig.bus.pdes_deg(rig.bus.frames(P.CMD_ID)[-1])
    assert frame[0] == pytest.approx(-120.0, abs=2 * LSB_DEG)
    assert frame[2] == pytest.approx(-50.0 + 1.49, abs=2 * LSB_DEG)              # the offset goes on the wire
    assert rig.arm.read_joints()["arm_joint3"] == pytest.approx(-50.0, abs=2 * LSB_DEG)

    rig.arm.disconnect()
    assert rig.bus.closed
    assert not rig.arm.is_connected


def test_an_unmeasured_gripper_reads_back_exactly_the_command_the_twin_sent():
    """0..100 -> p_des -> 0..100 is not exact in floating point: an unmeasured gripper reads back the twin's
    own number, not the round trip."""
    rig = connected_open_loop()
    a1x = rig.arm.a1x
    v = 12.3
    assert 100.0 * a1x.gripper_opening(a1x.gripper_pdes(v / 100.0)) != v         # 12.300000000000002
    assert rig.arm.read_joints()["gripper"] == 0.0                             # gripper_start

    assert rig.arm.send_joints({"gripper": v}) == {"gripper": v}
    rig.run(0.1)
    assert rig.arm.read_joints()["gripper"] == v
    assert rig.arm.goal_positions() == {"gripper": v}


def test_stale_feedback_marks_every_arm_joint_stale_and_refuses_commands():
    rig = connected()
    rig.arm.send_joints({"arm_joint1": 20.0})
    rig.run(0.2)

    rig.bus.silent = True
    rig.run(0.2)

    assert rig.arm.link_silent
    assert rig.arm._link_silent                                                # the contract's older name
    assert set(rig.arm._stale) == set(ARM)                                     # the twin reads it
    faults = rig.arm.faults()
    assert len(faults) == 1
    assert faults[0].startswith("no response from the arm for")
    assert rig.arm.send_joints({"arm_joint1": 25.0}) == {}

    rig.bus.silent = False
    rig.run(0.3)

    assert not rig.arm.link_silent
    assert rig.arm._stale == []
    assert rig.arm.faults() == []
    assert rig.arm.send_joints({"arm_joint1": 25.0}) == {"arm_joint1": 25.0}


def test_a_deaf_arm_is_reported_from_the_command_measured_gap():
    rig = connected()
    rig.bus.deaf = True
    rig.arm.send_joints({"arm_joint2": 40.0})

    rig.run(0.6)
    assert rig.arm.faults() == []                                              # not `deaf_s` yet

    rig.run(1.0)
    faults = rig.arm.faults()
    assert len(faults) == 1
    assert "deaf" in faults[0]
    assert "blocked" in faults[0]
    assert fault_joint(faults[0]) == "arm_joint2"                              # an arm joint fault: the twin stops

    rig.bus.deaf = False                                                       # FF 1-5-6 helped: the arm moves
    rig.run(0.3)
    assert rig.arm.faults() == []


def test_stale_and_frozen_faults_are_arm_faults_for_the_twin():
    rig = connected()

    rig.bus.silent = True
    rig.run(0.3)

    faults = rig.arm.faults()
    assert len(faults) == 1
    assert "no response" in faults[0]
    assert fault_joint(faults[0]) is None
    rig.bus.state = "bus-off"
    assert "CAN bus: bus-off" in rig.arm.faults()


def test_limits_are_urdf_widened_to_the_start_pose():
    rig = make_rig(bus=FakeA1XBus((0.0, 20.0, 1.49, 0.0, 0.0, 0.0)))
    rig.bus.tick(n=60)
    rig.arm.connect()

    limits = rig.arm.joint_limits()

    assert set(limits) == set(ARM)                                             # no gripper
    assert limits["arm_joint3"][1] == pytest.approx(1.49, abs=0.02)
    assert limits["arm_joint3"][0] == pytest.approx(math.degrees(-3.316))
    assert limits["arm_joint1"] == pytest.approx((math.degrees(-2.880), math.degrees(2.880)))
    assert rig.arm.send_joints({"arm_joint2": -30.0})["arm_joint2"] == pytest.approx(0.0)   # URDF 0 > config -2
    assert rig.arm.send_joints({"arm_joint4": 95.0})["arm_joint4"] == pytest.approx(88.0)

    rig.arm.set_joint_limits({"arm_joint1": (-10.0, 10.0), "unknown": (0.0, 1.0)})
    assert "unknown" not in rig.arm.limits
    assert rig.arm.a1x.limits()[0] == (math.radians(-10.0), math.radians(10.0))   # handed to the driver
    assert rig.arm.send_joints({"arm_joint1": 20.0}) == {"arm_joint1": 10.0}


def test_gripper_0_100_maps_linearly_to_pdes_endpoints_in_a_ten_byte_frame():
    rig = connected()

    for v, want in ((100.0, OPEN), (0.0, CLOSED), (50.0, (OPEN + CLOSED) / 2)):
        assert rig.arm.send_joints({"gripper": v}) == {"gripper": v}
        rig.run(1.5)
        data = rig.bus.frames(P.GRIP_ID)[-1]
        assert len(data) == 10
        assert gripper_pdes(data) == pytest.approx(want, abs=LSB)
        assert struct.unpack_from(">h", data, 4)[0] == int(25.0 * 60)          # gripper_kp

    assert all(s.fd for s in rig.bus.tx if s.can_id == P.GRIP_ID)
    ps = rig.bus.grip_pdes()
    assert max(abs(b - a) for a, b in zip(ps, ps[1:])) <= GRIP_SLEW
    assert rig.arm.gripper_travel_s() == pytest.approx(1.215)


def test_gripper_stream_starts_only_after_a_real_command():
    rig = connected_open_loop()
    held = rig.arm.read_joints()
    assert held["gripper"] == 0.0                                              # gripper_start = closed, no feedback
    assert not rig.arm.gripper_known()

    rig.arm.send_joints(held)                                                  # holding the pose after Connect
    rig.run(0.3)
    assert rig.bus.frames(P.CMD_ID)
    assert not rig.bus.frames(P.GRIP_ID)

    rig.arm.send_joints({"gripper": 0.3})                                      # within 0.5 of the reported 0
    rig.run(0.1)
    assert not rig.bus.frames(P.GRIP_ID)

    rig.arm.send_joints({"gripper": 10.0})
    rig.run(0.1)
    assert rig.bus.grip_pdes()[0] == pytest.approx(rig.arm.a1x.gripper_pdes(0.1), abs=2 / P.POS_SCALE)
    assert rig.arm.gripper_known()


def test_assume_gripper_only_before_the_stream_starts():
    rig = connected_open_loop(gripper_start=100.0)
    assert rig.arm.read_joints()["gripper"] == 100.0
    assert rig.arm.assume_gripper(0.0) is True
    assert rig.arm.read_joints()["gripper"] == 0.0
    rig.run(0.1)
    assert not rig.bus.frames(P.GRIP_ID)                                       # the assumption sends nothing

    rig.arm.send_joints({"gripper": 40.0})
    rig.run(0.1)
    assert rig.bus.frames(P.GRIP_ID)
    assert rig.arm.assume_gripper(0.0) is False                                # the stream is the truth
    assert rig.arm.read_joints()["gripper"] == 40.0

    # The default A1X has no measurement before the first 0x051 (group 7 at rest is uncertain): it assumes.
    measured = connected()
    assert not measured.arm.gripper_measured()
    assert measured.arm.assume_gripper(0.0) is True
    measured.arm.send_joints({"gripper": 60.0})
    measured.run(0.1)
    assert measured.arm.gripper_measured()
    assert measured.arm.assume_gripper(0.0) is False                           # a measurement assumes nothing


def test_gripper_target_freezes_on_contact_but_send_joints_returns_the_command():
    rig = connected()
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(1.5)
    assert rig.arm.send_joints({"gripper": 0.0}) == {"gripper": 0.0}
    rig.run(0.4)

    rig.bus.grip_effort = 3.0                                                  # jaws on the cube (rest 0)
    rig.run(0.05)
    held = rig.bus.frames(P.GRIP_ID)[-1]
    rig.run(0.5)

    assert set(rig.bus.frames(P.GRIP_ID)[-100:]) == {held}                     # the target stands: grips, no crush
    p = gripper_pdes(held)
    assert OPEN < p < CLOSED
    assert rig.arm.send_joints({"gripper": 0.0}) == {"gripper": 0.0}           # the command, not the frozen p_des
    assert rig.arm.goal_positions()["gripper"] == 0.0
    rig.run(0.2)
    assert rig.bus.frames(P.GRIP_ID)[-1] == held

    rig.arm.send_joints({"gripper": 100.0})                                    # opening lifts the freeze
    rig.run(0.2)
    assert gripper_pdes(rig.bus.frames(P.GRIP_ID)[-1]) < p - 0.3


def test_a_measured_gripper_reads_0_to_100_through_the_feedback_stops():
    rig = connected()
    assert rig.arm.calibration_warnings() == []
    rig.arm.send_joints({"gripper": 50.0})
    rig.run(1.5)
    assert rig.arm.gripper_measured()

    rig.bus.grip_effort = 0.1                                                  # the fake: jaws stay where we put them
    for q, want in ((0.039, 0.0), (-1.992, 100.0), ((0.039 - 1.992) / 2, 50.0)):
        rig.bus.q[6] = q
        rig.run(0.02)
        assert rig.arm.read_joints()["gripper"] == pytest.approx(want, abs=0.02)   # 1 LSB = 0.01 units

    warnings = make_rig(**OPEN_LOOP).arm.calibration_warnings()
    assert len(warnings) == 1
    assert "feedback not measured" in warnings[0]


@pytest.mark.parametrize("settings", [{}, OPEN_LOOP], ids=["measured", "open-loop"])
def test_closing_never_sends_a_pdes_past_gripper_closed_pdes(settings):
    """Every 0x051 frame at most `gripper_closed_pdes` (0.08): +0.6 at kp 20 gave a stall current
    (|effort| 7.8) and overheated the gripper motor. Also with a command below 0 and in open loop."""
    rig = connected(**settings)

    for v in (100.0, 0.0, -50.0, 60.0, 0.0, -1.0):
        rig.arm.send_joints({"gripper": v})
        rig.run(1.5)

    ps = rig.bus.grip_pdes()
    assert ps
    assert max(ps) <= CLOSED + LSB
    assert ps[-1] == pytest.approx(CLOSED, abs=LSB)
    assert min(ps) >= OPEN - LSB


def test_gripper_over_xcan_follows_the_config_flag_or_the_env_var(monkeypatch):
    """a1x.gripper_over_xcan (default true) / A1X_GRIPPER_XCAN=1: XCAN drives the gripper; other ports don't care."""
    monkeypatch.delenv("A1X_GRIPPER_XCAN", raising=False)

    def arm(port: str, flag: bool = False) -> A1XArm:
        return A1XArm(context(port, gripper_over_xcan=flag), bus=FakeA1XBus(), threaded=False)

    assert A1XConfig().gripper_over_xcan is True
    for port in ("xcan", "xcan:6"):
        allowed = arm(port, flag=True)
        assert allowed.gripper_drivable(), port
        assert allowed.gripper_xcan_trial(), port
    socketcan = arm("can0", flag=True)
    assert socketcan.gripper_drivable()
    assert not socketcan.gripper_xcan_trial()

    refused = arm("xcan", flag=False)
    for value, on in (("1", True), ("true", True), ("TAK", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv("A1X_GRIPPER_XCAN", value)
        assert refused.gripper_drivable() is on, value
        assert refused.gripper_xcan_trial() is on, value
    monkeypatch.setenv("A1X_GRIPPER_XCAN", "1")
    assert not arm("can0", flag=False).gripper_xcan_trial()


def test_send_joints_is_refused_while_enabling_and_enable_refuses_a_silent_link():
    rig = connected()
    seen = []
    wait = rig.arm.a1x._wait

    def send_while_waiting(secs: float) -> None:
        seen.append(rig.arm.send_joints({"arm_joint1": 50.0}))
        wait(secs)

    rig.arm.a1x._wait = send_while_waiting
    assert rig.arm.enable() == "sent FF 1 -> 5 -> 6; the arm holds the measured pose"

    assert seen
    assert all(s == {} for s in seen)
    assert rig.arm.send_joints({"arm_joint1": 11.0}) == {"arm_joint1": 11.0}
    rig.bus.silent = True
    rig.run(0.3)
    with pytest.raises(RuntimeError, match="no fresh"):
        rig.arm.enable()


def test_efforts_and_velocities_come_from_fresh_feedback_for_the_six_arm_joints():
    rig = connected()
    rig.bus.eff[:6] = [1.5, -22.0, 3.0, 0.0, -0.5, 7.25]
    rig.bus.grip_effort = 9.0                                                  # must not show up in arm efforts
    rig.arm.send_joints({"arm_joint2": 50.0})                                  # 30 -> 50 deg
    rig.run(0.05)

    eff = rig.arm.efforts()
    vel = rig.arm.velocities()

    assert set(eff) == set(ARM)
    assert set(vel) == set(ARM)
    assert eff["arm_joint2"] == pytest.approx(-22.0, abs=1.0 / P.EFF_SCALE)
    assert eff["arm_joint6"] == pytest.approx(7.25, abs=1.0 / P.EFF_SCALE)
    assert eff["arm_joint1"] == pytest.approx(1.5, abs=0.002 + 1.0 / P.EFF_SCALE)   # + fake bus noise
    assert 9.0 not in eff.values()
    assert vel["arm_joint2"] == pytest.approx(90.0, abs=2.0)                   # backend slew rate 90 deg/s
    assert vel["arm_joint1"] == pytest.approx(0.0, abs=0.1)
    assert rig.arm.gripper_effort() == pytest.approx(9.0, abs=1.0 / P.EFF_SCALE)
    assert rig.arm.gripper_force == pytest.approx(1.2)

    rig.bus.silent = True
    rig.run(0.2)                                                               # > stale_ms
    assert rig.arm.efforts() is None
    assert rig.arm.velocities() is None


def test_info_names_the_link_and_is_never_simulated():
    unconnected = A1XArm(context(), bus=FakeA1XBus(), threaded=False)
    on_can0 = A1XArm(context("can0"), bus=FakeA1XBus(), threaded=False)

    assert unconnected.info == BackendInfo(name="Galaxea A1X (CAN)", description="link CAN", simulated=False)
    assert on_can0.info.description == "link can0"


def test_a_missing_transport_module_names_the_galaxeo_extra(monkeypatch):
    def no_module(iface: str) -> None:
        raise ImportError("No module named 'usb'")

    monkeypatch.setattr("galaxeo.bus.open_bus", no_module)
    xcan = A1XArm(context("xcan"), threaded=False)
    pcan = A1XArm(context("PCAN_USBBUS1"), threaded=False)

    with pytest.raises(ImportError, match=r'pip install "galaxeo\[xcan\]"'):
        xcan.connect()
    with pytest.raises(ImportError, match=r'pip install "galaxeo\[pcan\]"'):
        pcan.connect()


def test_the_backend_refuses_an_unknown_key_in_its_section():
    with pytest.raises(ValueError, match=r"a1x\.stream_rate"):
        A1XArm(context(stream_rate=100.0), bus=FakeA1XBus(), threaded=False)


@pytest.mark.parametrize("key", ["kp", "gripper_kp"])
def test_kp_zero_in_the_section_is_refused(key):
    with pytest.raises(ValueError, match="kp"):
        make_rig(**{key: 0.0})


def _load(target: str) -> Any:
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def test_the_entry_points_name_the_robot_folder_and_the_backend_factory():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    folder = Path(_load(project["entry-points"]["lerobot_mp.robots"]["a1x"]))
    factory = _load(project["entry-points"]["lerobot_mp.backends"]["a1x"])

    assert (folder / "robot.yaml").is_file()
    assert (folder / "a1x.xml").is_file()
    assert (folder / "demo.json").is_file()
    assert factory is create
    assert isinstance(factory(context()), A1XArm)


def test_package_data_ships_every_file_of_the_robot_folder():
    tomllib = pytest.importorskip("tomllib")
    setuptools = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]
    package = REPO / "galaxeo" / "robot"

    data = setuptools["package-data"]["galaxeo.robot"]
    listed = sorted(p for p in data if p.startswith("a1x/"))
    present = sorted(str(p.relative_to(package)) for p in (package / "a1x").iterdir())

    assert "galaxeo.robot" in setuptools["packages"]
    assert listed == present
    assert "README.md" in data, "the package README ships with the robot folder"
