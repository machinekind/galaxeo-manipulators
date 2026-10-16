"""The A1X driver without hardware: `galaxeo.a1x.A1X` on a fake bus, in wire units.

Arm joints by index 0..5 in radians, the gripper as p_des. The fakes are in `a1x_fakes`; the twin's
backend on top of the driver has its own tests in `test_robot_a1x.py`.
"""

from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass
from typing import Any

import pytest
from a1x_fakes import (CLOSED, DEFAULTS, GRIP_SLEW, LSB, LSB_DEG, OPEN, OPEN_LOOP, START_DEG, STEP, Clock,
                       FakeA1XBus, TrickleBus, gripper_pdes, largest_fall, largest_rise, ticking_sleep)

from galaxeo import protocol as P
from galaxeo.a1x import GRIPPER_XCAN_ENV, A1X, A1XConfig, Fault, load_settings

rad = math.radians


@dataclass
class Rig:
    a1x: A1X
    bus: FakeA1XBus
    clock: Clock

    def run(self, secs: float, read: bool = True) -> None:
        for _ in range(int(round(secs / STEP))):
            self.clock.t += STEP
            self.bus.tick(STEP)
            self.a1x.io_step(self.clock.t)
            if read:
                self.a1x.read()


def make_rig(bus: FakeA1XBus | None = None, **settings: Any) -> Rig:
    clock = Clock()
    bus = bus or FakeA1XBus()
    a1x = A1X(A1XConfig(**settings), bus=bus, clock=clock, threaded=False, sleep=ticking_sleep(clock, bus),
              connect_timeout_s=0.05)
    return Rig(a1x=a1x, bus=bus, clock=clock)


def connected(**kw: Any) -> Rig:
    rig = make_rig(**kw)
    rig.bus.tick(n=60)
    rig.a1x.connect()
    return rig


def connected_open_loop(**kw: Any) -> Rig:
    """As `connected`, but the gripper without feedback (`gripper_fb_*` = None)."""
    return connected(**OPEN_LOOP, **kw)


def test_connect_refuses_frozen_telemetry_arriving_one_frame_at_a_time():
    rig = make_rig(bus=TrickleBus())
    rig.bus.frozen = True

    with pytest.raises(RuntimeError, match="frozen"):
        rig.a1x.connect()

    assert not rig.a1x.is_connected
    assert rig.bus.tx == []
    assert rig.a1x._identical == rig.a1x.cfg.frozen_frames + 1


def test_first_frame_equals_the_measured_pose_and_slews_from_there():
    rig = connected()

    rig.a1x.send({0: rad(40.0)})
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


def test_joint_offsets_are_subtracted_on_read_and_added_on_send():
    rig = connected(joint_offsets_deg=(0.0, 0.0, 1.49, 0.0, 0.0, 0.0))
    assert rig.a1x.read().joints[2] == pytest.approx(rad(START_DEG[2] - 1.49), abs=LSB)

    assert rig.a1x.send({2: rad(-50.0)}).joints == {2: rad(-50.0)}
    rig.run(1.0)

    assert rig.bus.pdes_deg(rig.bus.frames(P.CMD_ID)[-1])[2] == pytest.approx(-50.0 + 1.49, abs=2 * LSB_DEG)
    assert rig.a1x.read().joints[2] == pytest.approx(rad(-50.0), abs=LSB)
    with pytest.raises(ValueError, match="gripper is separate"):
        rig.a1x.send({6: 0.0})


def test_stale_feedback_sets_link_silent_stops_transmit_and_reports_a_fault():
    rig = connected()
    rig.a1x.send({0: rad(20.0)})
    rig.run(0.2)

    rig.bus.silent = True
    rig.run(0.2)

    assert rig.a1x.link_silent
    faults = rig.a1x.faults()
    assert len(faults) == 1
    assert faults[0].joint is None                                             # an arm fault, no joint
    assert faults[0].text.startswith("no response from the arm for")
    n = len(rig.bus.tx)
    rig.run(0.3)
    assert len(rig.bus.tx) == n
    assert rig.a1x.send({0: rad(25.0)}) is None

    # The next frame ends the silence, but the target from before the gap is gone: nothing goes out
    # until the caller (after its stop and a confirmed measurement) sends a new command.
    rig.bus.silent = False
    rig.run(0.3)

    assert not rig.a1x.link_silent
    assert rig.a1x.faults() == []
    assert len(rig.bus.tx) == n
    q0 = math.degrees(rig.bus.q[0])
    rig.a1x.send({0: rad(25.0)})
    rig.run(0.05)
    first = rig.bus.pdes_deg(rig.bus.tx[n].data)
    assert abs(first[0] - q0) <= 90.0 * STEP * 2 + 2 * LSB_DEG                  # the stream starts from the measurement


def test_frozen_payloads_are_reported_and_stop_transmit():
    rig = connected()
    rig.a1x.send({0: rad(12.0)})
    rig.run(0.1)

    rig.bus.frozen = True
    rig.run(0.4)                                                               # 80 identical frames

    assert any("telemetry frozen" in f.text for f in rig.a1x.faults())
    assert rig.a1x.link_silent
    n = len(rig.bus.tx)
    rig.run(0.2)
    assert len(rig.bus.tx) == n


def test_a_deaf_arm_is_reported_from_the_command_measured_gap():
    rig = connected()
    rig.bus.deaf = True
    rig.a1x.send({1: rad(40.0)})

    rig.run(0.6)
    assert rig.a1x.faults() == []                                              # not `deaf_s` yet

    rig.run(1.0)
    faults = rig.a1x.faults()
    assert len(faults) == 1
    assert faults[0].joint == 1                                                # J2: a joint fault
    assert "deaf" in faults[0].text
    assert "blocked" in faults[0].text

    rig.bus.deaf = False                                                       # FF 1-5-6 helped: the arm moves
    rig.run(0.3)
    assert rig.a1x.faults() == []


def test_a_bus_off_adapter_is_an_arm_fault():
    rig = connected()
    rig.bus.state = "bus-off"
    assert rig.a1x.faults() == [Fault(None, "CAN bus: bus-off")]


def test_bounds_are_urdf_widened_to_the_start_pose_and_send_clamps_to_them_and_the_limits():
    rig = make_rig(bus=FakeA1XBus((0.0, 20.0, 1.49, 0.0, 0.0, 0.0)))
    rig.bus.tick(n=60)
    rig.a1x.connect()

    bounds = rig.a1x.bounds()

    assert len(bounds) == 6                                                    # no gripper
    assert bounds[2][1] == pytest.approx(rad(1.49), abs=rad(0.02))
    assert bounds[2][0] == pytest.approx(-3.316)
    assert bounds[0] == pytest.approx((-2.880, 2.880))
    assert rig.a1x.limits() == ((-math.inf, math.inf),) * 6
    assert rig.a1x.send({1: rad(-30.0)}).joints[1] == pytest.approx(0.0)       # URDF 0
    rig.a1x.set_limits({3: (rad(-88.0), rad(88.0))})
    assert rig.a1x.limits()[3] == (rad(-88.0), rad(88.0))
    assert rig.a1x.send({3: rad(95.0)}).joints[3] == rad(88.0)                 # the caller's limit, inside URDF 90


def test_gripper_pdes_goes_out_in_a_ten_byte_frame_at_the_slew_rate():
    rig = connected()

    for want in (OPEN, CLOSED, (OPEN + CLOSED) / 2):
        assert rig.a1x.send(gripper=want).gripper == want
        rig.run(1.5)
        data = rig.bus.frames(P.GRIP_ID)[-1]
        assert len(data) == 10
        assert gripper_pdes(data) == pytest.approx(want, abs=LSB)
        assert struct.unpack_from(">h", data, 4)[0] == int(25.0 * 60)          # gripper_kp

    assert all(s.fd for s in rig.bus.tx if s.can_id == P.GRIP_ID)
    ps = rig.bus.grip_pdes()
    assert max(abs(b - a) for a, b in zip(ps, ps[1:])) <= GRIP_SLEW


def test_gripper_pdes_is_linear_in_the_opening():
    a1x = make_rig().a1x
    assert a1x.gripper_pdes(0.0) == CLOSED
    assert a1x.gripper_pdes(1.0) == pytest.approx(OPEN)
    assert a1x.gripper_pdes(0.5) == pytest.approx((OPEN + CLOSED) / 2)
    assert a1x.gripper_opening(a1x.gripper_pdes(0.3)) == pytest.approx(0.3)


def test_gripper_stream_starts_only_after_a_real_command():
    rig = connected_open_loop()
    held = rig.a1x.read()
    assert held.gripper == CLOSED                                              # gripper_start = closed, no feedback
    assert not held.gripper_measured
    assert not rig.a1x.gripper_known()

    took = rig.a1x.send(dict(enumerate(held.joints)), held.gripper)            # holding the pose after connect
    assert took.gripper is None
    rig.run(0.3)
    assert rig.bus.frames(P.CMD_ID)
    assert not rig.bus.frames(P.GRIP_ID)

    assert rig.a1x.send(gripper=rig.a1x.gripper_pdes(0.003)).gripper is None   # within GRIP_START_DELTA
    rig.run(0.1)
    assert not rig.bus.frames(P.GRIP_ID)

    rig.a1x.send(gripper=rig.a1x.gripper_pdes(0.1))
    rig.run(0.1)
    # `gripper_start` is a guess: opening goes straight to the target (open jaws do not twitch,
    # closed ones do not snap shut in the first frame on a closed p_des).
    assert rig.bus.grip_pdes()[0] == pytest.approx(rig.a1x.gripper_pdes(0.1), abs=2 / P.POS_SCALE)
    assert rig.a1x.gripper_known()


def test_first_open_after_assume_goes_to_the_target_at_once_close_stays_a_ramp():
    """Assumed "closed": opening starts at the target (an empty gripper loses nothing, and a wrong assumption
    does not snap the jaws shut from +0.6 in the first frame). The same without an assumption, from
    `gripper_start` alone: after `connect` the jaws may stand open (the A1X keeps its last pose between
    sessions)."""
    rig = connected_open_loop()
    assert rig.a1x.assume_gripper(CLOSED) is True
    assert rig.a1x.gripper_known()
    assert rig.a1x.read().gripper == CLOSED
    rig.a1x.send(gripper=OPEN)
    rig.run(0.1)
    ps = rig.bus.grip_pdes()
    assert ps[0] == pytest.approx(OPEN, abs=LSB)
    assert set(ps) == {ps[0]}
    # Closing after it is an ordinary ramp.
    rig.a1x.send(gripper=CLOSED)
    rig.run(0.1)
    ps = rig.bus.grip_pdes()
    assert largest_rise(ps) <= GRIP_SLEW
    assert ps[-1] < 0.0

    partly = connected_open_loop()
    assert partly.a1x.assume_gripper(CLOSED)
    partly.a1x.send(gripper=partly.a1x.gripper_pdes(0.3))
    partly.run(0.05)
    assert partly.bus.grip_pdes()[0] == pytest.approx(partly.a1x.gripper_pdes(0.3), abs=2 / P.POS_SCALE)

    unassumed = connected_open_loop()                                          # no assumption, jaws open
    unassumed.bus.q[6] = OPEN
    unassumed.a1x.send(gripper=OPEN)
    unassumed.run(0.05)
    ps = unassumed.bus.grip_pdes()
    assert ps[0] == pytest.approx(OPEN, abs=LSB)
    assert max(ps) < OPEN + 0.01                                               # no frame toward closed
    assert unassumed.bus.q[6] == pytest.approx(OPEN, abs=LSB)                  # the jaws did not twitch
    unassumed.a1x.send(gripper=CLOSED)
    unassumed.run(0.1)
    assert largest_rise(unassumed.bus.grip_pdes()) <= GRIP_SLEW                # closing: a ramp


def test_a_real_gripper_command_survives_a_link_dropout_and_the_next_one_ramps():
    """`_g_cmd` from a command that really went out is no guess: after a link gap the next command ramps
    from it (only `gripper_start` alone or an assumption starts at the target when opening)."""
    rig = connected_open_loop()
    forty = rig.a1x.gripper_pdes(0.4)
    rig.a1x.send(gripper=forty)
    rig.run(1.5)

    rig.bus.silent = True
    rig.run(0.3)
    assert rig.a1x.link_silent
    assert rig.a1x._g_target is None

    rig.bus.silent = False
    rig.run(0.3)
    assert not rig.a1x.link_silent

    n = len(rig.bus.frames(P.GRIP_ID))
    rig.a1x.send(gripper=OPEN)
    rig.run(0.1)
    ps = rig.bus.grip_pdes()[n:]
    assert ps[0] == pytest.approx(forty - 2.0 * STEP, abs=2 / P.POS_SCALE)
    assert largest_fall(ps) <= GRIP_SLEW


def test_assume_gripper_only_before_the_stream_starts():
    rig = connected_open_loop(gripper_start=100.0)
    assert rig.a1x.read().gripper == rig.a1x.gripper_pdes(1.0)                 # gripper_start 100 = open
    assert rig.a1x.assume_gripper(CLOSED) is True
    assert rig.a1x.read().gripper == CLOSED
    rig.run(0.1)
    assert not rig.bus.frames(P.GRIP_ID)                                       # the assumption sends nothing

    forty = rig.a1x.gripper_pdes(0.4)
    rig.a1x.send(gripper=forty)
    rig.run(0.1)
    assert rig.bus.frames(P.GRIP_ID)
    assert rig.a1x.assume_gripper(CLOSED) is False                             # the stream is the truth
    assert rig.a1x.read().gripper == forty

    # The default A1X has no measurement before the first 0x051 (group 7 at rest is uncertain): it assumes.
    measured = connected()
    assert not measured.a1x.gripper_measured()
    assert measured.a1x.assume_gripper(CLOSED) is True
    measured.a1x.send(gripper=measured.a1x.gripper_pdes(0.6))
    measured.run(0.1)
    assert measured.a1x.gripper_measured()
    assert measured.a1x.assume_gripper(CLOSED) is False                        # a measurement assumes nothing


def test_gripper_force_is_measured_from_the_rest_effort_of_each_close():
    """Group 7's rest effort is sometimes a steady ~3 (> 1.2): a fixed threshold froze the target on the
    first tick of a close and the gripper stopped listening. The threshold counts from the rest of the
    first 0.1 s."""
    rig = connected()
    rig.a1x.send(gripper=OPEN)
    rig.run(1.5)

    rig.bus.grip_effort = 3.0
    rig.a1x.send(gripper=CLOSED)
    rig.run(1.5)

    assert rig.bus.grip_pdes()[-1] == pytest.approx(CLOSED, abs=LSB)
    assert not rig.a1x._g_frozen


def test_gripper_target_freezes_on_contact_but_send_returns_the_command():
    rig = connected()
    rig.a1x.send(gripper=OPEN)
    rig.run(1.5)
    assert rig.a1x.send(gripper=CLOSED).gripper == CLOSED
    rig.run(0.4)

    rig.bus.grip_effort = 3.0                                                  # jaws on the cube (rest 0)
    rig.run(0.05)
    held = rig.bus.frames(P.GRIP_ID)[-1]
    rig.run(0.5)

    assert set(rig.bus.frames(P.GRIP_ID)[-100:]) == {held}                     # the target stands: grips, no crush
    p = gripper_pdes(held)
    assert OPEN < p < CLOSED
    assert rig.a1x.send(gripper=CLOSED).gripper == CLOSED                      # the command, not the frozen p_des
    rig.run(0.2)
    assert rig.bus.frames(P.GRIP_ID)[-1] == held

    rig.a1x.send(gripper=OPEN)                                                 # opening lifts the freeze
    rig.run(0.2)
    assert gripper_pdes(rig.bus.frames(P.GRIP_ID)[-1]) < p - 0.3

    # ... and closing moves again (the rest is measured anew).
    rig.bus.grip_effort = 0.0
    low = rig.bus.grip_pdes()[-1]
    rig.a1x.send(gripper=CLOSED)
    rig.run(0.5)
    assert rig.bus.grip_pdes()[-1] > low + 0.5


def test_opening_a_little_after_contact_keeps_the_rest_so_the_clamp_does_not_ratchet():
    """Opened by a notch after contact: the jaws are still on the cube. If the next close measured the
    rest anew it would take the pressure (3) for rest, and the threshold would grow by `gripper_force`
    with every cycle."""
    rig = connected()
    rig.a1x.send(gripper=OPEN)
    rig.run(1.5)
    rig.a1x.send(gripper=CLOSED)
    rig.run(0.4)
    rig.bus.grip_effort = 3.0                                                  # the cube (rest 0)
    rig.run(0.1)
    assert rig.a1x._g_frozen

    p = rig.bus.grip_pdes()[-1]
    rig.a1x.send(gripper=p + 0.01 * (OPEN - CLOSED))                           # 1 % of the stroke more open
    rig.run(0.2)
    assert not rig.a1x._g_frozen
    assert rig.a1x._g_rest == pytest.approx(0.0, abs=0.01)
    ajar = rig.bus.grip_pdes()[-1]
    assert ajar < p

    rig.a1x.send(gripper=CLOSED)                                               # the pressure is still 3
    rig.run(0.5)
    assert rig.a1x._g_frozen
    assert max(rig.bus.grip_pdes()[-100:]) <= ajar + GRIP_SLEW                 # no further than one tick


def test_a_target_frozen_at_the_open_endpoint_is_released_by_an_equal_open_command():
    """A target frozen at the open end (`gripper_open_pdes`): a strict `want < target` let nothing through
    any more, while the caller saw every command as done."""
    rig = connected()
    rig.a1x.send(gripper=OPEN)
    rig.run(1.5)
    # As after a contact at the open end.
    rig.a1x._g_target = rig.a1x._g_stream
    rig.a1x._g_frozen = True
    assert rig.a1x._g_target == pytest.approx(OPEN)

    rig.a1x.send(gripper=CLOSED)
    rig.run(0.2)
    assert rig.bus.grip_pdes()[-1] == pytest.approx(OPEN, abs=LSB)             # frozen: closing stands

    rig.a1x.send(gripper=OPEN)                                                 # = the target: lifts the freeze
    assert not rig.a1x._g_frozen
    rig.a1x.send(gripper=CLOSED)
    rig.run(1.5)
    assert rig.bus.grip_pdes()[-1] == pytest.approx(CLOSED, abs=LSB)


@pytest.mark.parametrize("port, xcan", [("xcan", True), ("xcan:6", True), ("XCAN", True),
                                        ("can0", False), ("PCAN_USBBUS1", False)])
def test_gripper_is_drivable_over_xcan_by_default(monkeypatch, port, xcan):
    """Measured 2026-09-27: XCAN delivers 0x051. Every port drives the gripper unless the config says no."""
    monkeypatch.delenv(GRIPPER_XCAN_ENV, raising=False)

    on = A1X(A1XConfig(), port, bus=FakeA1XBus(), threaded=False)
    off = A1X(A1XConfig(gripper_over_xcan=False), port, bus=FakeA1XBus(), threaded=False)

    assert on.gripper_drivable() is True
    assert on.gripper_xcan_trial() is xcan
    assert off.gripper_drivable() is (not xcan)


def test_gripper_travel_time_covers_the_slew():
    rig = make_rig()

    assert rig.a1x.gripper_travel_s() == pytest.approx(abs(OPEN - CLOSED) / 2.0 + 0.2)
    assert rig.a1x.gripper_travel_s() == pytest.approx(1.215)                  # 2.03 / 2.0 + 0.2 margin


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
    assert not rig.a1x.gripper_measured()                                      # nothing drives the gripper yet
    rig.a1x.send(gripper=rig.a1x.gripper_pdes(0.5))
    rig.run(1.5)
    assert rig.a1x.gripper_measured()
    assert rig.a1x.gripper_known()

    rig.bus.grip_effort = 0.1                                                  # the fake: jaws stay where we put them
    for q, want in ((0.039, 0.0), (-1.992, 1.0), ((0.039 - 1.992) / 2, 0.5)):
        rig.bus.q[6] = q
        rig.run(0.02)
        reading = rig.a1x.read()
        assert reading.gripper_measured
        assert rig.a1x.gripper_opening(reading.gripper) == pytest.approx(want, abs=0.0002)   # 1 LSB = 0.0001


def test_resting_group_7_after_power_up_is_not_trusted_until_the_gripper_is_driven():
    """2026-09-27: after power-on group 7 at rest read -164.9 deg (-2.88 rad), while the jaws stood at about
    -85 deg; it followed p_des only with the 0x051 stream running. Until the first 0x051 the driver reports
    its belief (`gripper_start`, closed) and the guess rule holds (the first opening goes straight to the
    target), then the measurement."""
    rig = connected(bus=FakeA1XBus(grip_q=-1.49, grip_rest_reading=-2.88))
    assert rig.a1x._fb.pos[6] == pytest.approx(-2.88, abs=LSB)
    assert not rig.a1x.gripper_measured()
    assert not rig.a1x.gripper_known()
    assert rig.a1x.read().gripper == CLOSED                                    # gripper_start, not -2.88 mapped

    assert rig.a1x.assume_gripper(CLOSED) is True
    rig.a1x.send(gripper=OPEN)
    rig.run(0.02)
    assert rig.bus.grip_pdes()[0] == pytest.approx(OPEN, abs=LSB)              # guessed: straight to the target
    rig.run(0.5)
    assert rig.a1x.gripper_measured()
    assert rig.a1x.gripper_opening(rig.a1x.read().gripper) == pytest.approx(1.0, abs=0.025)   # -1.95 = about 0.98

    # A link loss (the arm may have been off): the measurement is invalid again until the next 0x051.
    rig.bus.silent = True
    rig.run(0.3, read=False)
    rig.bus.silent = False
    rig.run(0.05)
    assert not rig.a1x.gripper_measured()
    assert rig.a1x.read().gripper == OPEN                                      # the last command


@pytest.mark.parametrize("settings", [{}, OPEN_LOOP], ids=["measured", "open-loop"])
def test_closing_never_sends_a_pdes_past_gripper_closed_pdes(settings):
    """Every 0x051 frame at most `gripper_closed_pdes` (0.08): +0.6 at kp 20 gave a stall current
    (|effort| 7.8) and overheated the gripper motor. Also with a target past the closed end and in open loop."""
    rig = connected(**settings)

    for opening in (1.0, 0.0, -0.5, 0.6, 0.0, -0.01):
        rig.a1x.send(gripper=rig.a1x.gripper_pdes(opening))
        rig.run(1.5)

    ps = rig.bus.grip_pdes()
    assert ps
    assert max(ps) <= CLOSED + LSB
    assert ps[-1] == pytest.approx(CLOSED, abs=LSB)
    assert min(ps) >= OPEN - LSB


def test_gripper_over_xcan_follows_the_config_flag_or_the_env_var(monkeypatch):
    """gripper_over_xcan (default true) / A1X_GRIPPER_XCAN=1: XCAN drives the gripper; other ports don't care."""
    monkeypatch.delenv(GRIPPER_XCAN_ENV, raising=False)

    def a1x(port: str, flag: bool = False) -> A1X:
        return A1X(A1XConfig(gripper_over_xcan=flag), port, bus=FakeA1XBus(), threaded=False)

    refused = a1x("xcan", flag=False)
    for value, on in (("1", True), ("true", True), ("TAK", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv(GRIPPER_XCAN_ENV, value)
        assert refused.gripper_drivable() is on, value
        assert refused.gripper_xcan_trial() is on, value
    monkeypatch.setenv(GRIPPER_XCAN_ENV, "1")
    assert not a1x("can0", flag=False).gripper_xcan_trial()


def test_enable_sequence_streams_before_between_and_after_each_code():
    rig = connected()
    latest = rig.bus.rx[-1].data if rig.bus.rx else P.encode_feedback(rig.bus.q)
    measured = [math.degrees(x) for x in P.decode_feedback(latest).pos[:6]]

    rig.a1x.enable()

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
    sleep = rig.a1x._sleep
    calls = itertools.count(1)

    def sleep_then_go_silent(secs: float) -> None:
        if next(calls) == 70:                                                  # waiting before the second FF code
            rig.bus.silent = True
        sleep(secs)

    rig.a1x._sleep = sleep_then_go_silent
    with pytest.raises(RuntimeError, match="went silent"):
        rig.a1x.enable()
    rig.a1x._sleep = sleep

    rig.bus.p_des = None
    n = len(rig.bus.tx)
    rig.bus.silent = False
    rig.run(0.5)

    assert len(rig.bus.tx) == n
    assert rig.a1x.faults() == []


def test_send_is_refused_while_enabling_and_enable_refuses_a_silent_link():
    rig = connected()
    seen = []
    wait = rig.a1x._wait

    def send_while_waiting(secs: float) -> None:
        seen.append(rig.a1x.send({0: rad(50.0)}))
        wait(secs)

    rig.a1x._wait = send_while_waiting
    rig.a1x.enable()

    assert seen
    assert all(s is None for s in seen)
    assert rig.a1x.send({0: rad(11.0)}).joints == {0: rad(11.0)}
    rig.bus.silent = True
    rig.run(0.3)
    with pytest.raises(RuntimeError, match="no fresh"):
        rig.a1x.enable()


@pytest.mark.parametrize("key", ["kp", "gripper_kp"])
def test_kp_zero_never_reaches_the_bus(key):
    with pytest.raises(ValueError, match="kp"):
        make_rig(**{key: 0.0})


def test_disconnect_never_sends_a_release_frame():
    rig = make_rig()
    rig.bus.tick(n=60)
    rig.a1x.connect()
    rig.a1x.send({0: rad(15.0)}, rig.a1x.gripper_pdes(0.2))
    rig.run(0.3)
    rig.a1x.enable()
    rig.run(0.1)

    rig.a1x.disconnect()

    assert rig.bus.closed
    assert not rig.a1x.is_connected
    released = [s for s in rig.bus.tx if s.can_id == P.FF_ID and s.data[0] in P.RELEASE_CODES]
    assert released == []
    n = len(rig.bus.tx)
    rig.run(0.1, read=False)
    assert len(rig.bus.tx) == n


def test_efforts_and_velocities_come_from_fresh_feedback_for_the_six_arm_joints():
    rig = connected()
    rig.bus.eff[:6] = [1.5, -22.0, 3.0, 0.0, -0.5, 7.25]
    rig.bus.grip_effort = 9.0                                                  # must not show up in arm efforts
    rig.a1x.send({1: rad(50.0)})                                               # 30 -> 50 deg
    rig.run(0.05)

    eff = rig.a1x.efforts()
    vel = rig.a1x.velocities()

    assert len(eff) == 6
    assert len(vel) == 6
    assert eff[1] == pytest.approx(-22.0, abs=1.0 / P.EFF_SCALE)
    assert eff[5] == pytest.approx(7.25, abs=1.0 / P.EFF_SCALE)
    assert eff[0] == pytest.approx(1.5, abs=0.002 + 1.0 / P.EFF_SCALE)        # + fake bus noise
    assert 9.0 not in eff
    assert vel[1] == pytest.approx(rad(90.0), abs=rad(2.0))                    # driver slew rate 90 deg/s
    assert vel[0] == pytest.approx(0.0, abs=rad(0.1))

    # A stopped arm: zero velocity, the effort still from the latest frame.
    rig.run(0.5)
    assert max(abs(v) for v in rig.a1x.velocities()) < rad(0.1)
    rig.bus.eff[2] = 30.0
    rig.run(0.01)
    assert rig.a1x.efforts()[2] == pytest.approx(30.0, abs=1.0 / P.EFF_SCALE)


def test_gripper_effort_is_the_absolute_group_7_effort_or_none_without_fresh_feedback():
    assert make_rig().a1x.gripper_effort() is None                             # before connect

    rig = connected()
    rig.bus.grip_effort = -3.0
    rig.run(0.05)
    assert rig.a1x.gripper_effort() == pytest.approx(3.0, abs=1.0 / P.EFF_SCALE)
    assert rig.a1x.gripper_force == pytest.approx(1.2)

    rig.bus.silent = True
    rig.run(0.2)                                                               # > stale_ms
    assert rig.a1x.gripper_effort() is None


def test_efforts_and_velocities_are_none_without_fresh_feedback():
    unconnected = make_rig()
    assert unconnected.a1x.efforts() is None
    assert unconnected.a1x.velocities() is None

    rig = connected()
    rig.run(0.05)
    assert rig.a1x.efforts() is not None

    rig.bus.silent = True
    rig.run(0.2)                                                               # > stale_ms
    assert rig.a1x.efforts() is None
    assert rig.a1x.velocities() is None

    rig.bus.silent = False
    rig.run(0.05)
    assert rig.a1x.efforts() is not None

    rig.bus.frozen = True
    rig.run(0.4)                                                               # 80 identical frames
    assert rig.a1x.efforts() is None
    assert rig.a1x.velocities() is None

    rig.a1x.disconnect()
    assert rig.a1x.efforts() is None


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
