"""The `a1x` backend: read-only gripper state, and command frames of another sender on the CAN link.

Everything runs on the `FakeA1XBus` of `test_robot_a1x.py` with its virtual clock; no CAN link is opened.
"""

from __future__ import annotations

import dataclasses

import pytest

from galaxeo import protocol as P
from galaxeo.bus import Frame
from galaxeo.robot.backend import (
    ECHO_KEEP,
    ECHO_WINDOW_S,
    FOREIGN_FAULT,
    FOREIGN_WINDOW_S,
    A1XArm,
    GripperState,
)
from test_robot_a1x import CLOSED, DEFAULTS, OPEN, OPEN_LOOP, STEP, Rig, connected, fault_joint, make_rig

#: Default stream: 2 p_des/s over the 2.03 p_des between closed (+0.08) and open (-1.95).
RATE = 2.0 * 100.0 / (CLOSED - OPEN)


def state(arm: A1XArm) -> GripperState:
    gs = arm.gripper_state()
    assert gs is not None
    return gs


def snapshot(arm: A1XArm) -> dict[str, str]:
    return {k: repr(v) for k, v in vars(arm).items()}


def open_stream(rig: Rig) -> None:
    """A stream really sent at 100. The opening is only guessed after Connect, so it goes there at once."""
    rig.arm.send_joints({"gripper": 100.0})
    rig.run(0.1)
    assert state(rig.arm).streamed == pytest.approx(100.0, abs=1e-9)


# ---- gripper state ----------------------------------------------------------------------------
def test_gripper_state_is_none_before_connect_and_after_disconnect():
    rig = make_rig()
    before = rig.arm.gripper_state()

    rig.bus.tick(n=60)
    rig.arm.connect()
    during = rig.arm.gripper_state()

    rig.arm.disconnect()
    after = rig.arm.gripper_state()

    assert before is None
    assert isinstance(during, GripperState)
    assert after is None


def test_before_any_command_only_the_reading_and_the_constants_are_known():
    rig = connected()
    rig.run(0.05)

    gs = state(rig.arm)

    assert gs.commanded is None
    assert gs.target is None
    assert gs.streamed is None
    assert gs.frozen is False
    assert gs.position is None
    assert gs.measured is False
    assert gs.effort == pytest.approx(0.0, abs=1.0 / P.EFF_SCALE)
    assert gs.t == rig.clock.t
    assert gs.age_s == pytest.approx(0.0, abs=1e-12)
    assert gs.rate == pytest.approx(RATE, rel=1e-12)
    assert gs.force_threshold == DEFAULTS.gripper_force


def test_the_state_is_frozen():
    gs = state(connected().arm)

    with pytest.raises(dataclasses.FrozenInstanceError):
        gs.commanded = 50.0                                                    # type: ignore[misc]


def test_commanded_is_none_until_a_command_passes_the_backends_gate():
    rig = connected()

    rig.arm.send_joints({"gripper": 0.0})                                      # equals the reported opening
    rig.run(0.1)
    held = state(rig.arm)
    held_frames = rig.bus.frames(P.GRIP_ID)

    rig.arm.send_joints({"gripper": 60.0})
    rig.run(STEP)
    moved = state(rig.arm)

    assert held.commanded is None
    assert held_frames == []
    assert moved.commanded == 60.0
    assert moved.target == pytest.approx(60.0, abs=1e-9)
    assert moved.streamed == pytest.approx(60.0, abs=1e-9)                    # a guessed opening: no ramp


def test_streamed_closes_at_the_stream_rate_while_commanded_is_already_zero():
    rig = connected()
    open_stream(rig)
    rig.arm.send_joints({"gripper": 0.0})

    streamed = []
    for _ in range(16):                                                        # 0.64 s of the 1.0 s stroke
        rig.run(0.04)
        gs = state(rig.arm)

        assert gs.commanded == 0.0
        assert gs.target == pytest.approx(0.0, abs=1e-9)
        streamed.append(gs.streamed)

    assert all(b < a for a, b in zip(streamed, streamed[1:]))
    assert streamed[-1] == pytest.approx(100.0 - RATE * 0.64, abs=1e-6)


def test_effort_above_the_threshold_while_closing_freezes_the_target_at_the_stream():
    rig = connected()
    open_stream(rig)
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.4)
    closing = state(rig.arm)

    rig.bus.grip_effort = 2.0
    rig.run(0.05)
    gs = state(rig.arm)
    rig.run(0.3)
    later = state(rig.arm)

    assert closing.frozen is False
    assert gs.frozen is True
    assert gs.target == gs.streamed
    assert 0.0 < gs.streamed < 100.0
    assert gs.commanded == 0.0
    assert gs.effort == pytest.approx(2.0, abs=1.0 / P.EFF_SCALE)
    assert gs.effort > gs.force_threshold
    assert later.streamed == gs.streamed                                       # the stream holds


def test_stale_feedback_gives_no_effort_and_age_grows_while_the_bus_is_silent():
    rig = connected()
    open_stream(rig)
    rig.arm.send_joints({"gripper": 0.0})
    rig.run(0.1)
    fresh = state(rig.arm)
    stale_s = DEFAULTS.stale_ms / 1000.0

    rig.bus.silent = True
    ages = []
    for _ in range(8):
        rig.run(0.04)
        gs = state(rig.arm)

        assert gs.t == fresh.t
        assert gs.age_s == pytest.approx(rig.clock.t - fresh.t, abs=1e-9)
        assert (gs.effort is None) == (gs.age_s > stale_s)
        ages.append(gs.age_s)

    assert fresh.effort is not None
    assert fresh.age_s == pytest.approx(0.0, abs=1e-12)
    assert all(b > a for a, b in zip(ages, ages[1:]))
    assert ages[-1] > stale_s
    assert gs.target is None                                                   # the stream was dropped
    assert gs.streamed is None
    assert gs.commanded == 0.0


def test_position_is_measured_only_once_the_gripper_is_driven():
    rig = connected()
    rig.run(0.05)
    idle = state(rig.arm)

    rig.arm.send_joints({"gripper": 60.0})
    rig.run(0.2)
    driven = state(rig.arm)
    span = DEFAULTS.gripper_fb_open - DEFAULTS.gripper_fb_closed

    assert idle.position is None
    assert idle.measured is False
    assert driven.measured is True
    assert driven.position == pytest.approx((rig.bus.q[6] - DEFAULTS.gripper_fb_closed) * 100.0 / span, abs=0.02)


def test_position_is_none_without_gripper_feedback_points():
    rig = connected(**OPEN_LOOP)
    rig.arm.send_joints({"gripper": 60.0})
    rig.run(0.2)

    gs = state(rig.arm)

    assert gs.position is None
    assert gs.measured is False
    assert gs.streamed == pytest.approx(60.0, abs=1e-9)


def test_a_thousand_calls_change_no_backend_field_and_send_no_frame():
    rig = connected()
    rig.arm.send_joints({"arm_joint1": 30.0, "gripper": 100.0})
    rig.run(0.3)

    def thousand_calls_change_nothing() -> None:
        rig.bus.tick(n=3)                                                      # queued frames stay queued
        before = snapshot(rig.arm)
        n_tx = len(rig.bus.tx)
        n_rx = len(rig.bus.rx)

        for _ in range(1000):
            rig.arm.gripper_state()

        assert snapshot(rig.arm) == before
        assert len(rig.bus.tx) == n_tx
        assert len(rig.bus.rx) == n_rx

    thousand_calls_change_nothing()

    # Stale feedback no `read_joints` has seen yet: unlike a read, this must not set `_link_silent`.
    rig.bus.silent = True
    rig.run(0.3, read=False)

    assert rig.arm._link_silent is False
    assert state(rig.arm).effort is None
    thousand_calls_change_nothing()
    assert rig.arm._link_silent is False


def test_gripper_state_never_raises_and_logs_the_error_once(monkeypatch, caplog):
    rig = connected()

    def broken() -> None:
        raise RuntimeError("zepsute")

    monkeypatch.setattr(rig.arm, "_fresh_fb", broken)
    with caplog.at_level("ERROR", logger="galaxeo.robot.backend"):
        first = rig.arm.gripper_state()
        second = rig.arm.gripper_state()

    assert first is None
    assert second is None
    assert len([r for r in caplog.records if "stanu chwytaka" in r.getMessage()]) == 1


# ---- another sender ---------------------------------------------------------------------------
def test_the_constants_of_the_sender_check():
    assert FOREIGN_WINDOW_S == 0.5
    assert ECHO_KEEP == 16
    assert ECHO_WINDOW_S == 0.25


def test_the_other_sender_fault_names_no_joint_so_the_twin_stops():
    assert fault_joint(FOREIGN_FAULT) is None


@pytest.mark.parametrize("can_id, payload", [
    (P.CMD_ID, P.encode_arm([0.1] * 6, 20.0, 1.0)),
    (P.GRIP_ID, P.encode_gripper(0.3, 25.0, 1.0)),
    (P.FF_ID, P.encode_ff(1)),
], ids=["arm", "gripper", "function"])
def test_a_command_frame_the_backend_never_sent_is_counted_and_reported_for_half_a_second(can_id, payload):
    rig = connected()
    rig.run(0.05)
    quiet_faults = rig.arm.faults()
    pose = rig.arm.read_joints()

    rig.bus.rx.append(Frame(can_id, payload))
    rig.run(STEP)
    t_seen = rig.clock.t

    assert quiet_faults == []
    assert rig.arm.foreign_frames() == 1
    assert rig.arm.faults() == [FOREIGN_FAULT]
    assert rig.arm.read_joints() == pose                                       # not decoded as feedback
    assert rig.bus.tx == []                                                    # and nothing was sent

    rig.run(FOREIGN_WINDOW_S - 0.01)

    assert rig.clock.t - t_seen < FOREIGN_WINDOW_S
    assert rig.arm.faults() == [FOREIGN_FAULT]

    rig.run(0.02)

    assert rig.arm.faults() == []
    assert rig.arm.foreign_frames() == 1                                       # counted since connecting

    rig.bus.rx.append(Frame(can_id, payload))
    rig.run(STEP)

    assert rig.arm.foreign_frames() == 2
    assert rig.arm.faults() == [FOREIGN_FAULT]


def stalled_echo(stall_s: float) -> Rig:
    """Our last 0x050 echoed back while the CAN thread stalls `stall_s` (the GIL, a garbage
    collection) before its next exchange."""
    rig = connected()
    open_stream(rig)
    rig.arm.send_joints({"arm_joint1": 40.0})
    rig.run(0.1)
    rig.bus.rx.append(Frame(P.CMD_ID, rig.bus.frames(P.CMD_ID)[-1]))
    rig.clock.t += stall_s
    rig.bus.tick(stall_s)
    rig.arm.io_step(rig.clock.t)
    return rig


def test_our_own_echo_drained_after_a_60_ms_stall_is_no_fault():
    assert stalled_echo(0.06).arm.faults() == []


def test_our_own_echo_drained_after_a_60_ms_stall_is_not_counted_as_another_sender():
    assert stalled_echo(0.06).arm.foreign_frames() == 0


def test_the_count_starts_again_on_the_next_connect():
    rig = connected()
    rig.bus.rx.append(Frame(P.FF_ID, P.encode_ff(1)))
    rig.run(STEP)

    rig.arm.disconnect()
    rig.bus.tick(n=60)
    rig.arm.connect()

    assert rig.arm.foreign_frames() == 0
    assert rig.arm.faults() == []


def test_an_echo_of_the_backends_own_payloads_is_not_another_sender():
    rig = connected()
    rig.arm.enable()                                                           # FF 1, 5, 6 went out
    rig.run(1.0)

    # The same function codes a second later are another sender, not an echo.
    rig.bus.rx.extend([Frame(P.FF_ID, P.encode_ff(5)), Frame(P.FF_ID, P.encode_ff(6))])
    rig.run(STEP)

    assert rig.arm.foreign_frames() == 2
    assert rig.arm.faults() == [FOREIGN_FAULT]

    rig.run(FOREIGN_WINDOW_S + 0.05)
    assert rig.arm.faults() == []

    open_stream(rig)
    rig.arm.send_joints({"arm_joint1": 40.0, "gripper": 0.0})
    rig.run(0.2)                                                               # moving and closing: payloads differ
    cmd = rig.bus.frames(P.CMD_ID)
    grip = rig.bus.frames(P.GRIP_ID)

    assert len(set(cmd[-ECHO_KEEP - 1:])) == ECHO_KEEP + 1

    echoes = [
        Frame(P.CMD_ID, cmd[-1]),
        Frame(P.GRIP_ID, grip[-1]),
        Frame(P.CMD_ID, cmd[-2] + bytes(64 - len(cmd[-2]))),                  # padded to an FD length
        Frame(P.GRIP_ID, grip[-2] + bytes(12 - len(grip[-2]))),
    ]
    rig.bus.rx.extend(echoes)
    rig.run(STEP)

    assert rig.arm.foreign_frames() == 2
    assert rig.arm.faults() == []

    # A second copy of an echoed payload, one sent more than the echo window ago, or one padded with
    # anything but zeros: another sender.
    stale = rig.bus.frames(P.CMD_ID)[-round((ECHO_WINDOW_S + 0.05) / STEP)]   # 0.3 s old at the next drain
    rig.bus.rx.extend([
        Frame(P.CMD_ID, cmd[-1]),
        Frame(P.CMD_ID, stale),
        Frame(P.GRIP_ID, rig.bus.frames(P.GRIP_ID)[-1] + b"\x00\x01"),
    ])
    rig.run(STEP)

    assert rig.arm.foreign_frames() == 5
    assert rig.arm.faults() == [FOREIGN_FAULT]
