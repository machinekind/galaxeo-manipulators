"""The A1X driver: read-only gripper state, and command frames of another sender on the CAN link.

Everything runs on the `FakeA1XBus` of `a1x_fakes` with its virtual clock, in wire units; no CAN link is
opened. The twin backend's view of the same (openings 0..100, faults as text) is in `test_robot_a1x.py`.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from a1x_fakes import CLOSED, DEFAULTS, OPEN, OPEN_LOOP, STEP
from test_a1x import Rig, connected, make_rig

from galaxeo import protocol as P
from galaxeo.a1x import A1X, ECHO_KEEP, ECHO_WINDOW_S, FOREIGN_FAULT, FOREIGN_WINDOW_S, Fault, GripperState
from galaxeo.bus import Frame

rad = math.radians

#: The stream's default speed [p_des/s].
RATE = DEFAULTS.gripper_rate_pdes_s
#: The other sender's fault: on the whole arm, no joint.
FOREIGN = Fault(None, FOREIGN_FAULT)


def state(a1x: A1X) -> GripperState:
    gs = a1x.gripper_state()
    assert gs is not None
    return gs


def snapshot(a1x: A1X) -> dict[str, str]:
    return {k: repr(v) for k, v in vars(a1x).items()}


def open_stream(rig: Rig) -> None:
    """A stream really sent at the open end. The opening is only guessed after connect, so it goes there at once."""
    rig.a1x.send(gripper=OPEN)
    rig.run(0.1)
    assert state(rig.a1x).streamed == pytest.approx(OPEN, abs=1e-9)


# ---- gripper state ----------------------------------------------------------------------------
def test_gripper_state_is_none_before_connect_and_after_disconnect():
    rig = make_rig()
    before = rig.a1x.gripper_state()

    rig.bus.tick(n=60)
    rig.a1x.connect()
    during = rig.a1x.gripper_state()

    rig.a1x.disconnect()
    after = rig.a1x.gripper_state()

    assert before is None
    assert isinstance(during, GripperState)
    assert after is None


def test_before_any_command_only_the_reading_and_the_constants_are_known():
    rig = connected()
    rig.run(0.05)

    gs = state(rig.a1x)

    assert gs.commanded is None
    assert gs.target is None
    assert gs.streamed is None
    assert gs.frozen is False
    assert gs.position is None
    assert gs.measured is False
    assert gs.effort == pytest.approx(0.0, abs=1.0 / P.EFF_SCALE)
    assert gs.t == rig.clock.t
    assert gs.age_s == pytest.approx(0.0, abs=1e-12)
    assert gs.rate == RATE
    assert gs.force_threshold == DEFAULTS.gripper_force


def test_the_state_is_frozen():
    gs = state(connected().a1x)

    with pytest.raises(dataclasses.FrozenInstanceError):
        gs.commanded = 0.0                                                     # type: ignore[misc]


def test_commanded_is_none_until_a_command_passes_the_drivers_gate():
    rig = connected()
    want = rig.a1x.gripper_pdes(0.6)

    rig.a1x.send(gripper=CLOSED)                                               # equals the reported position
    rig.run(0.1)
    held = state(rig.a1x)
    held_frames = rig.bus.frames(P.GRIP_ID)

    rig.a1x.send(gripper=want)
    rig.run(STEP)
    moved = state(rig.a1x)

    assert held.commanded is None
    assert held_frames == []
    assert moved.commanded == want
    assert moved.target == pytest.approx(want, abs=1e-9)
    assert moved.streamed == pytest.approx(want, abs=1e-9)                    # a guessed opening: no ramp


def test_streamed_closes_at_the_stream_rate_while_commanded_is_already_closed():
    rig = connected()
    open_stream(rig)
    rig.a1x.send(gripper=CLOSED)

    streamed = []
    for _ in range(16):                                                        # 0.64 s of the 1.0 s stroke
        rig.run(0.04)
        gs = state(rig.a1x)

        assert gs.commanded == CLOSED
        assert gs.target == pytest.approx(CLOSED, abs=1e-9)
        streamed.append(gs.streamed)

    assert all(b > a for a, b in zip(streamed, streamed[1:]))                 # more positive = tighter
    assert streamed[-1] == pytest.approx(OPEN + RATE * 0.64, abs=1e-6)


def test_effort_above_the_threshold_while_closing_freezes_the_target_at_the_stream():
    rig = connected()
    open_stream(rig)
    rig.a1x.send(gripper=CLOSED)
    rig.run(0.4)
    closing = state(rig.a1x)

    rig.bus.grip_effort = 2.0
    rig.run(0.05)
    gs = state(rig.a1x)
    rig.run(0.3)
    later = state(rig.a1x)

    assert closing.frozen is False
    assert gs.frozen is True
    assert gs.target == gs.streamed
    assert OPEN < gs.streamed < CLOSED
    assert gs.commanded == CLOSED
    assert gs.effort == pytest.approx(2.0, abs=1.0 / P.EFF_SCALE)
    assert gs.effort > gs.force_threshold
    assert later.streamed == gs.streamed                                       # the stream holds


def test_stale_feedback_gives_no_effort_and_age_grows_while_the_bus_is_silent():
    rig = connected()
    open_stream(rig)
    rig.a1x.send(gripper=CLOSED)
    rig.run(0.1)
    fresh = state(rig.a1x)
    stale_s = DEFAULTS.stale_ms / 1000.0

    rig.bus.silent = True
    ages = []
    for _ in range(8):
        rig.run(0.04)
        gs = state(rig.a1x)

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
    assert gs.commanded == CLOSED


def test_position_is_measured_only_once_the_gripper_is_driven():
    rig = connected()
    rig.run(0.05)
    idle = state(rig.a1x)

    rig.a1x.send(gripper=rig.a1x.gripper_pdes(0.6))
    rig.run(0.2)
    driven = state(rig.a1x)
    span = DEFAULTS.gripper_fb_open - DEFAULTS.gripper_fb_closed
    jaw = rig.a1x.gripper_pdes((rig.bus.q[6] - DEFAULTS.gripper_fb_closed) / span)

    assert idle.position is None
    assert idle.measured is False
    assert driven.measured is True
    assert driven.position == pytest.approx(jaw, abs=0.02 * (CLOSED - OPEN) / 100.0)


def test_position_is_none_without_gripper_feedback_points():
    rig = connected(**OPEN_LOOP)
    want = rig.a1x.gripper_pdes(0.6)
    rig.a1x.send(gripper=want)
    rig.run(0.2)

    gs = state(rig.a1x)

    assert gs.position is None
    assert gs.measured is False
    assert gs.streamed == pytest.approx(want, abs=1e-9)


def test_a_thousand_calls_change_no_driver_field_and_send_no_frame():
    rig = connected()
    rig.a1x.send({0: rad(30.0)}, gripper=OPEN)
    rig.run(0.3)

    def thousand_calls_change_nothing() -> None:
        rig.bus.tick(n=3)                                                      # queued frames stay queued
        before = snapshot(rig.a1x)
        n_tx = len(rig.bus.tx)
        n_rx = len(rig.bus.rx)

        for _ in range(1000):
            rig.a1x.gripper_state()

        assert snapshot(rig.a1x) == before
        assert len(rig.bus.tx) == n_tx
        assert len(rig.bus.rx) == n_rx

    thousand_calls_change_nothing()

    # Stale feedback no `read` has seen yet: unlike a read, this must not set `link_silent`.
    rig.bus.silent = True
    rig.run(0.3, read=False)

    assert rig.a1x.link_silent is False
    assert state(rig.a1x).effort is None
    thousand_calls_change_nothing()
    assert rig.a1x.link_silent is False


def test_gripper_state_never_raises_and_logs_the_error_once(monkeypatch, caplog):
    rig = connected()

    def broken() -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(rig.a1x, "_fresh_fb", broken)
    with caplog.at_level("ERROR", logger="galaxeo.a1x"):
        first = rig.a1x.gripper_state()
        second = rig.a1x.gripper_state()

    assert first is None
    assert second is None
    assert len([r for r in caplog.records if "gripper state" in r.getMessage()]) == 1


# ---- another sender ---------------------------------------------------------------------------
def test_the_constants_of_the_sender_check():
    assert FOREIGN_WINDOW_S == 0.5
    assert ECHO_KEEP == 16
    assert ECHO_WINDOW_S == 0.25


@pytest.mark.parametrize("can_id, payload", [
    (P.CMD_ID, P.encode_arm([0.1] * 6, 20.0, 1.0)),
    (P.GRIP_ID, P.encode_gripper(0.3, 25.0, 1.0)),
    (P.FF_ID, P.encode_ff(1)),
], ids=["arm", "gripper", "function"])
def test_a_command_frame_the_driver_never_sent_is_counted_and_reported_for_half_a_second(can_id, payload):
    rig = connected()
    rig.run(0.05)
    quiet_faults = rig.a1x.faults()
    pose = rig.a1x.read()

    rig.bus.rx.append(Frame(can_id, payload))
    rig.run(STEP)
    t_seen = rig.clock.t

    assert quiet_faults == []
    assert rig.a1x.foreign_frames() == 1
    assert rig.a1x.faults() == [FOREIGN]                                       # no joint: the caller stops
    assert rig.a1x.read() == pose                                              # not decoded as feedback
    assert rig.bus.tx == []                                                    # and nothing was sent

    rig.run(FOREIGN_WINDOW_S - 0.01)

    assert rig.clock.t - t_seen < FOREIGN_WINDOW_S
    assert rig.a1x.faults() == [FOREIGN]

    rig.run(0.02)

    assert rig.a1x.faults() == []
    assert rig.a1x.foreign_frames() == 1                                       # counted since connecting

    rig.bus.rx.append(Frame(can_id, payload))
    rig.run(STEP)

    assert rig.a1x.foreign_frames() == 2
    assert rig.a1x.faults() == [FOREIGN]


def stalled_echo(stall_s: float) -> Rig:
    """Our last 0x050 echoed back while the CAN thread stalls `stall_s` (the GIL, a garbage
    collection) before its next exchange."""
    rig = connected()
    open_stream(rig)
    rig.a1x.send({0: rad(40.0)})
    rig.run(0.1)
    rig.bus.rx.append(Frame(P.CMD_ID, rig.bus.frames(P.CMD_ID)[-1]))
    rig.clock.t += stall_s
    rig.bus.tick(stall_s)
    rig.a1x.io_step(rig.clock.t)
    return rig


def test_our_own_echo_drained_after_a_60_ms_stall_is_no_fault():
    assert stalled_echo(0.06).a1x.faults() == []


def test_our_own_echo_drained_after_a_60_ms_stall_is_not_counted_as_another_sender():
    assert stalled_echo(0.06).a1x.foreign_frames() == 0


def test_the_count_starts_again_on_the_next_connect():
    rig = connected()
    rig.bus.rx.append(Frame(P.FF_ID, P.encode_ff(1)))
    rig.run(STEP)

    rig.a1x.disconnect()
    rig.bus.tick(n=60)
    rig.a1x.connect()

    assert rig.a1x.foreign_frames() == 0
    assert rig.a1x.faults() == []


def test_an_echo_of_the_drivers_own_payloads_is_not_another_sender():
    rig = connected()
    rig.a1x.enable()                                                           # FF 1, 5, 6 went out
    rig.run(1.0)

    # The same function codes a second later are another sender, not an echo.
    rig.bus.rx.extend([Frame(P.FF_ID, P.encode_ff(5)), Frame(P.FF_ID, P.encode_ff(6))])
    rig.run(STEP)

    assert rig.a1x.foreign_frames() == 2
    assert rig.a1x.faults() == [FOREIGN]

    rig.run(FOREIGN_WINDOW_S + 0.05)
    assert rig.a1x.faults() == []

    open_stream(rig)
    rig.a1x.send({0: rad(40.0)}, gripper=CLOSED)
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

    assert rig.a1x.foreign_frames() == 2
    assert rig.a1x.faults() == []

    # A second copy of an echoed payload, one sent more than the echo window ago, or one padded with
    # anything but zeros: another sender.
    stale = rig.bus.frames(P.CMD_ID)[-round((ECHO_WINDOW_S + 0.05) / STEP)]   # 0.3 s old at the next drain
    rig.bus.rx.extend([
        Frame(P.CMD_ID, cmd[-1]),
        Frame(P.CMD_ID, stale),
        Frame(P.GRIP_ID, rig.bus.frames(P.GRIP_ID)[-1] + b"\x00\x01"),
    ])
    rig.run(STEP)

    assert rig.a1x.foreign_frames() == 5
    assert rig.a1x.faults() == [FOREIGN]
