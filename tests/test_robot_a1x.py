"""The twin's `a1x` backend without hardware: Galaxea A1X over CAN, driven like the twin drives it.

`FakeA1XBus` plays the arm on a virtual clock: it holds 7 groups (6 joints + gripper), moves toward
p_des at a finite speed, puts one 0x052 frame out on every `tick()` and records every frame sent.
Knobs: `silent`, `frozen`, `deaf`, `grip_effort`, `grip_rest_reading`, `state`.
"""

from __future__ import annotations

import importlib
import itertools
import math
import struct
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from galaxeo import protocol as P
from galaxeo.bus import Frame
from galaxeo.robot.backend import A1XArm, A1XConfig, BackendInfo, create, load_settings

REPO = Path(__file__).resolve().parents[1]
ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
START_DEG = (10.0, 30.0, -60.0, 5.0, 0.0, -20.0)
STEP = 0.005                                   # one stream tick (200 Hz)
LSB_DEG = math.degrees(1.0 / P.POS_SCALE)
LSB = 2 / P.POS_SCALE                          # p_des tolerance after encoding
#: The largest gripper p_des change per tick: 2 rad/s, plus encoding.
GRIP_SLEW = 2.0 * STEP + 2 / P.POS_SCALE
DEFAULTS = A1XConfig()
#: Gripper p_des endpoints of the default config (measured 2026-09-27: +0.08 closed, -1.95 open).
OPEN = DEFAULTS.gripper_open_pdes
CLOSED = DEFAULTS.gripper_closed_pdes
#: A gripper without feedback (open loop): the tests of a guessed opening (`gripper_start`, `assume_gripper`).
OPEN_LOOP = {"gripper_fb_closed": None, "gripper_fb_open": None}


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
class Clock:
    t: float = 100.0

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class Sent:
    can_id: int
    data: bytes
    fd: bool


def gripper_pdes(data: bytes) -> float:
    return struct.unpack_from(">h", data, 0)[0] / P.POS_SCALE


def largest_rise(values: Sequence[float]) -> float:
    return max(b - a for a, b in zip(values, values[1:]))


def largest_fall(values: Sequence[float]) -> float:
    return max(a - b for a, b in zip(values, values[1:]))


@dataclass
class FakeA1XBus:
    """An A1X on the bus: answers 0x052, obeys 0x050/0x051, records what is sent."""

    start_deg: InitVar[Sequence[float]] = START_DEG
    grip_q: InitVar[float] = 0.0
    #: Group 7 in the frames until any 0x051 arrives (as after power-on: an arbitrary value,
    #: 2026-09-27 -2.88 rad with the jaws at -1.49); None = the true jaw position.
    grip_rest_reading: float | None = None
    silent: bool = False
    frozen: bool = False
    deaf: bool = False
    grip_effort: float = 0.0
    state: str = "ok"
    #: Joints move toward p_des at this speed [rad/s].
    rate: float = math.radians(120.0)
    grip_rate: float = 2.0
    q: list[float] = field(init=False)
    v: list[float] = field(default_factory=lambda: [0.0] * 7)
    eff: list[float] = field(default_factory=lambda: [0.0] * 7)
    p_des: list[float] | None = None
    g_des: float | None = None
    rx: deque = field(default_factory=deque)
    tx: list[Sent] = field(default_factory=list)
    closed: bool = False
    ticks: int = 0
    frozen_payload: bytes | None = None

    def __post_init__(self, start_deg: Sequence[float], grip_q: float) -> None:
        self.q = [math.radians(x) for x in start_deg] + [grip_q]

    def recv(self, timeout: float = 0.0) -> Frame | None:
        return self.rx.popleft() if self.rx else None

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        data = bytes(data)
        self.tx.append(Sent(can_id=can_id, data=data, fd=fd))
        if can_id == P.CMD_ID and not self.deaf and not self.frozen:
            self.p_des = [struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE for j in range(6)]
        elif can_id == P.GRIP_ID and not self.frozen:
            self.g_des = gripper_pdes(data)

    def close(self) -> None:
        self.closed = True

    def tick(self, dt: float = STEP, n: int = 1) -> None:
        for _ in range(n):
            if self.p_des is not None:
                for j in range(6):
                    d = max(-self.rate * dt, min(self.rate * dt, self.p_des[j] - self.q[j]))
                    self.q[j] += d
                    self.v[j] = d / dt
            else:
                self.v[:6] = [0.0] * 6
            if self.g_des is not None and not self.grip_effort:
                self.q[6] += max(-self.grip_rate * dt, min(self.grip_rate * dt, self.g_des - self.q[6]))
            self.eff[6] = self.grip_effort
            self.ticks += 1
            if self.silent:
                continue

            q = list(self.q)
            if self.g_des is None and self.grip_rest_reading is not None:
                q[6] = self.grip_rest_reading
            if self.frozen:
                if self.frozen_payload is None:
                    self.frozen_payload = P.encode_feedback(q, self.v, self.eff)
                payload = self.frozen_payload
            else:
                self.frozen_payload = None
                eff = list(self.eff)
                eff[0] += 0.002 * (self.ticks % 2)      # measurement noise: live telemetry never repeats
                payload = P.encode_feedback(q, self.v, eff)
            self.rx.append(Frame(P.FB_ID, payload))

    def frames(self, can_id: int) -> list[bytes]:
        return [s.data for s in self.tx if s.can_id == can_id]

    def pdes_deg(self, data: bytes) -> list[float]:
        return [math.degrees(struct.unpack_from(">h", data, j * 10)[0] / P.POS_SCALE) for j in range(6)]

    def grip_pdes(self) -> list[float]:
        return [gripper_pdes(d) for d in self.frames(P.GRIP_ID)]


@dataclass
class TrickleBus(FakeA1XBus):
    """As on hardware: 0x052 frames arrive one by one - each drain of the queue sees at most one."""

    gave: bool = False

    def recv(self, timeout: float = 0.0) -> Frame | None:
        if self.gave:
            self.gave = False
            return None
        if not self.rx:
            self.tick()
        self.gave = True
        return self.rx.popleft() if self.rx else None


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

    def sleep(secs: float) -> None:
        n = max(1, round(secs / STEP))
        for _ in range(n):
            clock.t += secs / n
            bus.tick(secs / n)

    arm = A1XArm(context(**a1x), bus=bus, clock=clock, threaded=False, sleep=sleep, connect_timeout_s=0.05)
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


def test_connect_refuses_frozen_telemetry_arriving_one_frame_at_a_time():
    rig = make_rig(bus=TrickleBus())
    rig.bus.frozen = True

    with pytest.raises(RuntimeError, match="frozen"):
        rig.arm.connect()

    assert not rig.arm.is_connected
    assert rig.bus.tx == []
    assert rig.arm._identical == rig.arm.acfg.frozen_frames + 1


def test_first_frame_equals_the_measured_pose_and_slews_from_there():
    rig = connected()

    rig.arm.send_joints({"arm_joint1": 40.0})
    rig.run(0.6)

    frames = [rig.bus.pdes_deg(d) for d in rig.bus.frames(P.CMD_ID)]
    assert len(frames) == 120
    first = frames[0]
    assert first[0] == pytest.approx(START_DEG[0], abs=90.0 * STEP + 2 * LSB_DEG)   # one step from the measurement
    assert first[1:] == pytest.approx(list(START_DEG[1:]), abs=2 * LSB_DEG)          # the rest = the measurement
    assert max(abs(b[0] - a[0]) for a, b in zip(frames, frames[1:])) <= 90.0 * STEP + 2 * LSB_DEG
    assert frames[-1][0] == pytest.approx(40.0, abs=2 * LSB_DEG)
    assert all(len(d) == P.CMD_LEN for d in rig.bus.frames(P.CMD_ID))
    assert all(s.fd for s in rig.bus.tx if s.can_id == P.CMD_ID)
    # kp and kd from the config, never 0
    first_data = rig.bus.frames(P.CMD_ID)[0]
    assert all(struct.unpack_from(">h", first_data, j * 10 + 4)[0] == int(20.0 * 60) for j in range(6))


def test_stale_feedback_sets_link_silent_stops_transmit_and_reports_a_fault():
    rig = connected()
    rig.arm.send_joints({"arm_joint1": 20.0})
    rig.run(0.2)

    rig.bus.silent = True
    rig.run(0.2)

    assert rig.arm._link_silent
    assert set(rig.arm._stale) == set(ARM)
    faults = rig.arm.faults()
    assert len(faults) == 1
    assert faults[0].startswith("no response from the arm for")
    n = len(rig.bus.tx)
    rig.run(0.3)
    assert len(rig.bus.tx) == n
    assert rig.arm.send_joints({"arm_joint1": 25.0}) == {}

    # The next frame ends the silence, but the target from before the gap is gone: nothing goes out
    # until the twin (after its STOP and a confirmed measurement) sends a new command.
    rig.bus.silent = False
    rig.run(0.3)

    assert not rig.arm._link_silent
    assert rig.arm.faults() == []
    assert len(rig.bus.tx) == n
    q0 = math.degrees(rig.bus.q[0])
    rig.arm.send_joints({"arm_joint1": 25.0})
    rig.run(0.05)
    first = rig.bus.pdes_deg(rig.bus.tx[n].data)
    assert abs(first[0] - q0) <= 90.0 * STEP * 2 + 2 * LSB_DEG                  # the stream starts from the measurement


def test_frozen_payloads_are_reported_and_stop_transmit():
    rig = connected()
    rig.arm.send_joints({"arm_joint1": 12.0})
    rig.run(0.1)

    rig.bus.frozen = True
    rig.run(0.4)                                                               # 80 identical frames

    assert any("telemetry frozen" in f for f in rig.arm.faults())
    assert rig.arm._link_silent
    n = len(rig.bus.tx)
    rig.run(0.2)
    assert len(rig.bus.tx) == n


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


def test_gripper_stream_starts_only_after_a_real_command():
    rig = connected_open_loop()
    held = rig.arm.read_joints()
    assert held["gripper"] == 0.0                                              # gripper_start = closed, no feedback
    assert not rig.arm.gripper_known()

    rig.arm.send_joints(held)                                                  # holding the pose after Connect
    rig.run(0.3)
    assert rig.bus.frames(P.CMD_ID)
    assert not rig.bus.frames(P.GRIP_ID)

    rig.arm.send_joints({"gripper": 0.3})
    rig.run(0.1)
    assert not rig.bus.frames(P.GRIP_ID)

    rig.arm.send_joints({"gripper": 10.0})
    rig.run(0.1)
    # `gripper_start` is a guess: opening goes straight to the target (open jaws do not twitch,
    # closed ones do not snap shut in the first frame on a closed p_des).
    assert rig.bus.grip_pdes()[0] == pytest.approx(rig.arm._pdes(10.0), abs=2 / P.POS_SCALE)
    assert rig.arm.gripper_known()


def test_first_open_after_assume_goes_to_the_target_at_once_close_stays_a_ramp():
    """Assumed "closed" ("Set home = here"): opening starts at the target (an empty gripper loses nothing,
    and a wrong assumption does not snap the jaws shut from +0.6 in the first frame). The same without an
    assumption, from `gripper_start` alone: after Connect the jaws may stand open (the A1X keeps its last
    pose between sessions)."""
    rig = connected_open_loop()
    assert rig.arm.assume_gripper(0.0) is True
    assert rig.arm.gripper_known()
    assert rig.arm.read_joints()["gripper"] == 0.0
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(0.1)
    ps = rig.bus.grip_pdes()
    assert ps[0] == pytest.approx(OPEN, abs=LSB)
    assert set(ps) == {ps[0]}
    # Closing after it is an ordinary ramp.
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.1)
    ps = rig.bus.grip_pdes()
    assert largest_rise(ps) <= GRIP_SLEW
    assert ps[-1] < 0.0

    partly = connected_open_loop()
    assert partly.arm.assume_gripper(0.0)
    partly.arm.send_joints({"gripper": 30.0})
    partly.run(0.05)
    assert partly.bus.grip_pdes()[0] == pytest.approx(partly.arm._pdes(30.0), abs=2 / P.POS_SCALE)

    unassumed = connected_open_loop()                                          # no assumption, jaws open
    unassumed.bus.q[6] = OPEN
    unassumed.arm.send_joints({"gripper": 100.0})
    unassumed.run(0.05)
    ps = unassumed.bus.grip_pdes()
    assert ps[0] == pytest.approx(OPEN, abs=LSB)
    assert max(ps) < OPEN + 0.01                                               # no frame toward closed
    assert unassumed.bus.q[6] == pytest.approx(OPEN, abs=LSB)                  # the jaws did not twitch
    unassumed.arm.send_joints({"gripper": 0.0})
    unassumed.run(0.1)
    assert largest_rise(unassumed.bus.grip_pdes()) <= GRIP_SLEW                # closing: a ramp


def test_a_real_gripper_command_survives_a_link_dropout_and_the_next_one_ramps():
    """`_g_cmd` from a command that really went out is no guess: after a link gap the next command ramps
    from it (only `gripper_start` alone or an assumption starts at the target when opening)."""
    rig = connected_open_loop()
    rig.arm.send_joints({"gripper": 40.0})
    rig.run(1.5)

    rig.bus.silent = True
    rig.run(0.3)
    assert rig.arm._link_silent
    assert rig.arm._g_target is None

    rig.bus.silent = False
    rig.run(0.3)
    assert not rig.arm._link_silent

    n = len(rig.bus.frames(P.GRIP_ID))
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(0.1)
    ps = rig.bus.grip_pdes()[n:]
    assert ps[0] == pytest.approx(rig.arm._pdes(40.0) - 2.0 * STEP, abs=2 / P.POS_SCALE)
    assert largest_fall(ps) <= GRIP_SLEW


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


def test_gripper_force_is_measured_from_the_rest_effort_of_each_close():
    """Group 7's rest effort is sometimes a steady ~3 (> 1.2): a fixed threshold froze the target on the
    first tick of a close and the gripper stopped listening. The threshold counts from the rest of the
    first 0.1 s."""
    rig = connected()
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(1.5)

    rig.bus.grip_effort = 3.0
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(1.5)

    assert rig.bus.grip_pdes()[-1] == pytest.approx(CLOSED, abs=LSB)
    assert not rig.arm._g_frozen


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

    # ... and closing moves again (the rest is measured anew).
    rig.bus.grip_effort = 0.0
    low = rig.bus.grip_pdes()[-1]
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.5)
    assert rig.bus.grip_pdes()[-1] > low + 0.5


def test_opening_a_little_after_contact_keeps_the_rest_so_the_clamp_does_not_ratchet():
    """Opened by a notch after contact: the jaws are still on the cube. If the next close measured the
    rest anew it would take the pressure (3) for rest, and the threshold would grow by `gripper_force`
    with every cycle."""
    rig = connected()
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(1.5)
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.4)
    rig.bus.grip_effort = 3.0                                                  # the cube (rest 0)
    rig.run(0.1)
    assert rig.arm._g_frozen

    p = rig.bus.grip_pdes()[-1]
    at = (p - CLOSED) / (OPEN - CLOSED) * 100.0                                # the frozen p_des in 0..100
    rig.arm.send_joints({"gripper": at + 1.0})
    rig.run(0.2)
    assert not rig.arm._g_frozen
    assert rig.arm._g_rest == pytest.approx(0.0, abs=0.01)
    ajar = rig.bus.grip_pdes()[-1]
    assert ajar < p

    rig.arm.send_joints({"gripper": 0.0})                                      # the pressure is still 3
    rig.run(0.5)
    assert rig.arm._g_frozen
    assert max(rig.bus.grip_pdes()[-100:]) <= ajar + GRIP_SLEW                 # no further than one tick


def test_a_target_frozen_at_the_open_endpoint_is_released_by_an_equal_open_command():
    """A target frozen at the open end (`gripper_open_pdes`): a strict `want < target` let nothing through
    any more, while the twin showed every command as done."""
    rig = connected()
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(1.5)
    # As after a contact at the open end.
    rig.arm._g_target = rig.arm._g_stream
    rig.arm._g_frozen = True
    assert rig.arm._g_target == pytest.approx(OPEN)

    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.2)
    assert rig.bus.grip_pdes()[-1] == pytest.approx(OPEN, abs=LSB)             # frozen: closing stands

    rig.arm.send_joints({"gripper": 100.0})                                    # = the target: lifts the freeze
    assert not rig.arm._g_frozen
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(1.5)
    assert rig.bus.grip_pdes()[-1] == pytest.approx(CLOSED, abs=LSB)


@pytest.mark.parametrize("port, xcan", [("xcan", True), ("xcan:6", True), ("XCAN", True),
                                        ("can0", False), ("PCAN_USBBUS1", False)])
def test_gripper_is_drivable_over_xcan_by_default(monkeypatch, port, xcan):
    """Measured 2026-09-27: XCAN delivers 0x051. Every port drives the gripper unless the config says no."""
    monkeypatch.delenv("A1X_GRIPPER_XCAN", raising=False)

    arm = A1XArm(context(port), bus=FakeA1XBus(), threaded=False)
    off = A1XArm(context(port, gripper_over_xcan=False), bus=FakeA1XBus(), threaded=False)

    assert arm.gripper_drivable() is True
    assert arm.gripper_xcan_trial() is xcan
    assert off.gripper_drivable() is (not xcan)


def test_gripper_travel_time_covers_the_slew():
    rig = make_rig()

    assert rig.arm.gripper_travel_s() == pytest.approx(abs(OPEN - CLOSED) / 2.0 + 0.2)
    assert rig.arm.gripper_travel_s() == pytest.approx(1.215)                  # 2.03 / 2.0 + 0.2 margin


def test_default_a1x_gripper_is_measured_at_the_stops_found_on_2026_09_27():
    """The `--grip-test` measurement: group 7 of frame 0x052 is the jaw position in rad, like p_des. Closed
    stop +0.039 rad, open -1.992 rad; the closed p_des is 0.04 rad of pressure, not 0.56 rad as with +0.6."""
    assert DEFAULTS.gripper_fb_closed == 0.039
    assert DEFAULTS.gripper_fb_open == -1.992
    assert CLOSED == 0.08
    assert OPEN == -1.95
    assert CLOSED - DEFAULTS.gripper_fb_closed == pytest.approx(0.041)
    assert DEFAULTS.gripper_fb_open < OPEN

    rig = connected()
    assert rig.arm.calibration_warnings() == []
    assert not rig.arm.gripper_measured()                                      # nothing drives the gripper yet
    rig.arm.send_joints({"gripper": 50.0})
    rig.run(1.5)
    assert rig.arm.gripper_measured()
    assert rig.arm.gripper_known()

    rig.bus.grip_effort = 0.1                                                  # the fake: jaws stay where we put them
    for q, want in ((0.039, 0.0), (-1.992, 100.0), ((0.039 - 1.992) / 2, 50.0)):
        rig.bus.q[6] = q
        rig.run(0.02)
        assert rig.arm.read_joints()["gripper"] == pytest.approx(want, abs=0.02)   # 1 LSB = 0.01 units


def test_resting_group_7_after_power_up_is_not_trusted_until_the_gripper_is_driven():
    """2026-09-27: after power-on group 7 at rest read -164.9 deg (-2.88 rad), while the jaws stood at about
    -85 deg; it followed p_des only with the 0x051 stream running. Until the first 0x051 the backend reports
    its belief (`gripper_start` 0) and the guess rule holds (the first opening goes straight to the target),
    then the measurement."""
    rig = connected(bus=FakeA1XBus(grip_q=-1.49, grip_rest_reading=-2.88))
    assert rig.arm._fb.pos[6] == pytest.approx(-2.88, abs=LSB)
    assert not rig.arm.gripper_measured()
    assert not rig.arm.gripper_known()
    assert rig.arm.read_joints()["gripper"] == 0.0                             # gripper_start, not 142 from -2.88

    assert rig.arm.assume_gripper(0.0) is True
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(0.02)
    assert rig.bus.grip_pdes()[0] == pytest.approx(OPEN, abs=LSB)              # guessed: straight to the target
    rig.run(0.5)
    assert rig.arm.gripper_measured()
    assert rig.arm.read_joints()["gripper"] == pytest.approx(100.0, abs=2.5)   # -1.95 measured = about 98

    # A link loss (the arm may have been off): the measurement is invalid again until the next 0x051.
    rig.bus.silent = True
    rig.run(0.3, read=False)
    rig.bus.silent = False
    rig.run(0.05)
    assert not rig.arm.gripper_measured()
    assert rig.arm.read_joints()["gripper"] == 100.0                           # the last command


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


def test_enable_sequence_streams_before_between_and_after_each_code():
    rig = connected()
    latest = rig.bus.rx[-1].data if rig.bus.rx else P.encode_feedback(rig.bus.q)
    measured = [math.degrees(x) for x in P.decode_feedback(latest).pos[:6]]

    rig.arm.enable()

    ff_at = [k for k, s in enumerate(rig.bus.tx) if s.can_id == P.FF_ID]
    ff = [rig.bus.tx[k] for k in ff_at]
    assert [s.data for s in ff] == [b"\x01", b"\x05", b"\x06"]
    assert all(s.fd is False for s in ff)                                      # 1 byte, classic CAN
    bounds = [-1] + ff_at + [len(rig.bus.tx)]
    for a, b in zip(bounds, bounds[1:]):                                       # before, between and after
        arm_frames = [s for s in rig.bus.tx[a + 1:b] if s.can_id == P.CMD_ID]
        assert len(arm_frames) >= 50
    assert len(set(rig.bus.frames(P.CMD_ID))) == 1                             # p_des = the measurement throughout
    assert rig.bus.pdes_deg(rig.bus.frames(P.CMD_ID)[0]) == pytest.approx(measured, abs=LSB_DEG + 1e-9)
    assert not rig.bus.frames(P.GRIP_ID)


def test_link_lost_during_enable_leaves_no_target_to_resume_to():
    rig = connected()
    sleep = rig.arm._sleep
    calls = itertools.count(1)

    def sleep_then_go_silent(secs: float) -> None:
        if next(calls) == 70:                                                  # waiting before the second FF code
            rig.bus.silent = True
        sleep(secs)

    rig.arm._sleep = sleep_then_go_silent
    with pytest.raises(RuntimeError, match="went silent"):
        rig.arm.enable()
    rig.arm._sleep = sleep

    rig.bus.p_des = None
    n = len(rig.bus.tx)
    rig.bus.silent = False
    rig.run(0.5)

    assert len(rig.bus.tx) == n
    assert rig.arm.faults() == []


def test_send_joints_is_refused_while_enabling_and_enable_refuses_a_silent_link():
    rig = connected()
    seen = []
    wait = rig.arm._wait

    def send_while_waiting(secs: float) -> None:
        seen.append(rig.arm.send_joints({"arm_joint1": 50.0}))
        wait(secs)

    rig.arm._wait = send_while_waiting
    rig.arm.enable()

    assert seen
    assert all(s == {} for s in seen)
    assert rig.arm.send_joints({"arm_joint1": 11.0}) == {"arm_joint1": 11.0}
    rig.bus.silent = True
    rig.run(0.3)
    with pytest.raises(RuntimeError, match="no fresh"):
        rig.arm.enable()


@pytest.mark.parametrize("key", ["kp", "gripper_kp"])
def test_kp_zero_never_reaches_the_bus(key):
    with pytest.raises(ValueError, match="kp"):
        make_rig(**{key: 0.0})


def test_disconnect_never_sends_a_release_frame():
    rig = make_rig()
    rig.bus.tick(n=60)
    rig.arm.connect()
    rig.arm.send_joints({"arm_joint1": 15.0, "gripper": 20.0})
    rig.run(0.3)
    rig.arm.enable()
    rig.run(0.1)

    rig.arm.disconnect()

    assert rig.bus.closed
    assert not rig.arm.is_connected
    released = [s for s in rig.bus.tx if s.can_id == P.FF_ID and s.data[0] in P.RELEASE_CODES]
    assert released == []
    n = len(rig.bus.tx)
    rig.run(0.1, read=False)
    assert len(rig.bus.tx) == n


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

    # A stopped arm: zero velocity, the effort still from the latest frame.
    rig.run(0.5)
    assert max(abs(v) for v in rig.arm.velocities().values()) < 0.1
    rig.bus.eff[2] = 30.0
    rig.run(0.01)
    assert rig.arm.efforts()["arm_joint3"] == pytest.approx(30.0, abs=1.0 / P.EFF_SCALE)


def test_gripper_effort_is_the_absolute_group_7_effort_or_none_without_fresh_feedback():
    assert make_rig().arm.gripper_effort() is None                             # before connect

    rig = connected()
    rig.bus.grip_effort = -3.0
    rig.run(0.05)
    assert rig.arm.gripper_effort() == pytest.approx(3.0, abs=1.0 / P.EFF_SCALE)
    assert rig.arm.gripper_force == pytest.approx(1.2)

    rig.bus.silent = True
    rig.run(0.2)                                                               # > stale_ms
    assert rig.arm.gripper_effort() is None


def test_efforts_and_velocities_are_none_without_fresh_feedback():
    unconnected = make_rig()
    assert unconnected.arm.efforts() is None
    assert unconnected.arm.velocities() is None

    rig = connected()
    rig.run(0.05)
    assert rig.arm.efforts() is not None

    rig.bus.silent = True
    rig.run(0.2)                                                               # > stale_ms
    assert rig.arm.efforts() is None
    assert rig.arm.velocities() is None

    rig.bus.silent = False
    rig.run(0.05)
    assert rig.arm.efforts() is not None

    rig.bus.frozen = True
    rig.run(0.4)                                                               # 80 identical frames
    assert rig.arm.efforts() is None
    assert rig.arm.velocities() is None

    rig.arm.disconnect()
    assert rig.arm.efforts() is None


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


def test_settings_take_ints_for_floats_lists_for_tuples_and_null_for_unmeasured_feedback():
    cfg = load_settings(A1XConfig, {"kp": 30, "joint_offsets_deg": [0, 0, 1.49, 0, 0, 0],
                                    "gripper_fb_closed": None, "gripper_fb_open": None,
                                    "frozen_frames": 40, "enable_on_connect": True}, "a1x")

    assert cfg.kp == 30.0
    assert isinstance(cfg.kp, float)
    assert cfg.joint_offsets_deg == (0.0, 0.0, 1.49, 0.0, 0.0, 0.0)
    assert cfg.gripper_fb_closed is None
    assert cfg.frozen_frames == 40
    assert cfg.enable_on_connect is True
    assert cfg.stale_ms == DEFAULTS.stale_ms


def test_an_unknown_settings_key_is_refused_by_name():
    with pytest.raises(ValueError, match=r"a1x\.kpp"):
        load_settings(A1XConfig, {"kpp": 20.0}, "a1x")


def test_the_backend_refuses_an_unknown_key_in_its_section():
    with pytest.raises(ValueError, match=r"a1x\.stream_rate"):
        A1XArm(context(stream_rate=100.0), bus=FakeA1XBus(), threaded=False)


def test_a_missing_required_settings_key_is_refused_by_name():
    @dataclass
    class Needs:
        gain: float

    with pytest.raises(ValueError, match=r"a1x\.gain"):
        load_settings(Needs, {}, "a1x")


@pytest.mark.parametrize("key, value", [
    ("kp", True),                                   # bool is not a number here
    ("kp", "20"),
    ("enable_on_connect", 1),
    ("frozen_frames", 50.0),
    ("frozen_frames", True),
    ("joint_offsets_deg", 1.49),
    ("joint_offsets_deg", [0, 0, "1.49", 0, 0, 0]),
    ("joint_offsets_deg", [0, 0, False, 0, 0, 0]),
    ("gripper_fb_closed", "0.039"),
    ("gripper_fb_closed", False),
])
def test_a_settings_value_of_the_wrong_type_is_refused_by_name(key, value):
    with pytest.raises(TypeError, match=rf"a1x\.{key}"):
        load_settings(A1XConfig, {key: value}, "a1x")


def test_settings_that_are_not_a_mapping_are_refused():
    with pytest.raises(TypeError, match="a1x"):
        load_settings(A1XConfig, [("kp", 20.0)], "a1x")


def test_the_config_checks_still_run_after_loading():
    with pytest.raises(ValueError, match="set both or neither"):
        load_settings(A1XConfig, {"gripper_fb_closed": None}, "a1x")


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
