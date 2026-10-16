"""bench_a1x.py on fakes only: the keyboard on a pipe, the station and the jaw link on a fake A1X with a
virtual clock (tests/bench_fakes.py). Nothing opens a CAN link."""

from __future__ import annotations

import io
import os
import struct
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

import bench_a1x as bench  # noqa: E402
from bench_fakes import BenchBus, bench_station, jaw_pdes  # noqa: E402
from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import JOINTS, MoveResult  # noqa: E402
from galaxeo.bus import Frame  # noqa: E402

START_DEG = (90.0, 120.0, -77.0, 47.0, 0.0, 0.0)
J6 = JOINTS[5]


def bench_bus(**kw) -> BenchBus:
    return BenchBus(start_deg=START_DEG, **kw)


def connected(bus: BenchBus) -> bench.Station:
    station = bench_station(bus, {})
    station.wait_feedback(1.0)
    return station


# ====================================================================== the jaw link
def test_after_a_stop_the_jaw_gets_no_command():
    bus = bench_bus()
    station = connected(bus)
    station.stopped.set()

    with pytest.raises(bench.Ended), station.jaw.phase():
        station.jaw.grip(bench.GRIP_OPEN, station.stopped)

    assert bus.frames(P.GRIP_ID) == []


def test_after_a_stop_the_simulated_jaw_gets_no_command():
    jaw = bench.SimJaw(lambda: 0.0, lambda secs: None)
    stopped = threading.Event()
    stopped.set()

    with pytest.raises(bench.Ended):
        jaw.grip(bench.GRIP_OPEN, stopped)

    assert jaw.gripper_state().commanded is None


def test_the_first_jaw_command_of_a_run_opens_the_jaw():
    station = connected(bench_bus())

    with pytest.raises(ValueError), station.jaw.phase():
        station.jaw.grip(40.0, station.stopped)


def test_asking_for_the_opening_the_jaw_has_returns_at_once(caplog):
    bus = bench_bus()
    station = connected(bus)
    with station.jaw.phase():
        station.jaw.grip(bench.GRIP_OPEN, station.stopped)
    sent = len(bus.frames(P.GRIP_ID))

    with station.jaw.phase():
        station.jaw.grip(bench.GRIP_OPEN, station.stopped)

    assert "did not reach" not in caplog.text
    assert len(bus.frames(P.GRIP_ID)) == sent


def test_the_next_phase_starts_the_jaw_where_the_last_one_left_it():
    bus = bench_bus()
    station = connected(bus)
    with station.jaw.phase():
        station.jaw.grip(bench.GRIP_OPEN, station.stopped)

    with station.jaw.phase():
        station.jaw.grip(40.0, station.stopped)

    assert station.jaw.opening == pytest.approx(40.0)
    assert struct.unpack_from(">h", bus.frames(P.GRIP_ID)[-1], 0)[0] / P.POS_SCALE == pytest.approx(
        jaw_pdes(40.0), abs=2 / P.POS_SCALE)


def test_the_last_arm_frames_echo_is_no_other_sender():
    bus = bench_bus(echo=True)
    station = connected(bus)
    station.arm.set_armed(True)
    goal = np.radians(START_DEG) + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    result = station.move(goal, 20.0)

    with station.jaw.phase():
        station.jaw.wait(0.05)
        foreign = station.jaw.foreign_frames()

    assert result
    assert foreign == 0


def test_a_command_frame_of_another_sender_is_refused_before_anything_moves():
    bus = bench_bus()
    station = connected(bus)
    bus.on_tick = lambda: bus.rx.append(Frame(P.FF_ID, P.encode_ff(1)))

    with pytest.raises(bench.Refusal, match="busy"):
        bench.listen_for_other_senders(station)


def test_ctrl_c_in_a_phase_that_drives_the_arm_holds_the_measured_pose_not_the_stream():
    # The arm follows at 30 deg/s while the backend's stream slews at 90: the stream runs ahead.
    bus = bench_bus(rate=np.radians(30.0))
    station = connected(bus)
    base_j6 = START_DEG[5]
    fired: list[bool] = []

    def ctrl_c() -> None:
        if not fired and np.degrees(bus.q[5]) - base_j6 > 0.5:
            fired.append(True)
            raise KeyboardInterrupt

    bus.on_tick = ctrl_c
    link = station.jaw
    with pytest.raises(KeyboardInterrupt), link.phase():
        targets = link.app_degrees(np.radians(START_DEG))
        targets[J6] += 3.0
        link.send(targets)
        link.wait(1.0)
    last = bus.frames(P.CMD_ID)[-1]

    assert fired
    assert np.degrees(struct.unpack_from(">h", last, 50)[0] / P.POS_SCALE) - base_j6 < 1.0


def test_a_phase_samples_the_measured_arm_in_app_degrees():
    bus = bench_bus()
    station = connected(bus)

    with station.jaw.phase():
        station.jaw.wait(0.02)
        sample = station.jaw.sample()
        efforts = station.jaw.efforts()
        velocities = station.jaw.velocities()

    assert sample.fresh
    assert [sample.joints[j] for j in JOINTS] == pytest.approx(list(START_DEG), abs=0.01)
    assert station.jaw.app_degrees(np.radians(START_DEG)) == pytest.approx(dict(zip(JOINTS, START_DEG)))
    assert set(efforts) == set(velocities) == set(JOINTS)


def test_the_backend_never_closes_the_benchs_link():
    bus = bench_bus()
    station = connected(bus)

    with station.jaw.phase():
        station.jaw.wait(0.01)

    assert not bus.closed
    station.close()
    assert bus.closed


def test_the_backend_sees_the_urdf_limits_in_degrees_and_the_gripper_in_0_to_100():
    ctx = bench.bench_context("fake", {"gripper_rate_pdes_s": 1.5})

    assert ctx.joints == (*JOINTS, "gripper")
    assert ctx.limits["gripper"] == (0.0, 100.0)
    assert ctx.limits["arm_joint1"] == pytest.approx(tuple(np.degrees(P.URDF_LIMITS_RAD[0])))
    assert ctx.settings == {"gripper_rate_pdes_s": 1.5}


def test_holding_the_measured_pose_sends_one_frame_at_it():
    bus = bench_bus()
    station = connected(bus)

    q = station.hold_measured()

    sent = bus.frames(P.CMD_ID)
    p_des = [struct.unpack_from(">h", sent[0], j * 10)[0] / P.POS_SCALE for j in range(6)]
    assert len(sent) == 1
    assert q == pytest.approx(np.radians(START_DEG), abs=2 / P.POS_SCALE)
    assert p_des == pytest.approx(list(q), abs=2 / P.POS_SCALE)


def test_a_settled_reading_is_the_measured_pose():
    station = connected(bench_bus())

    reading = station.settled_reading()

    assert reading.q == pytest.approx(np.radians(START_DEG), abs=2 / P.POS_SCALE)


# ====================================================================== the twin's config
def test_a_config_without_an_a1x_section_changes_nothing(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "local.yaml"
    config.write_text("safety:\n  velocity_scale: 0.5\n", encoding="utf-8")

    assert bench.a1x_settings(str(config)) == {}


def test_a_config_that_enables_the_motors_on_connect_is_refused(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "local.yaml"
    config.write_text("a1x:\n  enable_on_connect: true\n", encoding="utf-8")

    with pytest.raises(bench.Refusal, match="enable on connect"):
        bench.a1x_settings(str(config))


# ====================================================================== moves
def test_a_long_move_is_cut_into_segments_under_the_limit():
    start = np.zeros(6)
    goal = np.radians([100.0, 0.0, -30.0, 0.0, 0.0, 0.0])

    parts = bench.segments(start, goal)

    assert len(parts) == 3
    assert np.array_equal(parts[-1], goal)
    steps = np.diff(np.vstack([start, *parts]), axis=0)
    assert np.degrees(np.abs(steps).max()) <= bench.SEGMENT_DEG


def test_a_move_of_nothing_is_one_segment_at_the_goal():
    q = np.radians(START_DEG)

    assert len(bench.segments(q, q)) == 1


@pytest.mark.parametrize("result, stopped", [
    (MoveResult("hold", "stop"), True),
    (MoveResult("rejected", "not armed"), True),
    (MoveResult("hold", "impact effort arm_joint2"), False),
    (MoveResult("rejected", "jump"), False),
])
def test_a_stop_is_told_from_other_endings(result, stopped):
    assert bench.stopped_by(result) is stopped
    assert (bench.ended_by(result).ending == "stop") is stopped


def test_an_ending_names_the_state_reason_and_detail():
    ended = bench.ended_by(MoveResult("hold", "stale", "no 0x052 for 0.2 s"))

    assert (ended.ending, ended.detail) == ("hold", "stale, no 0x052 for 0.2 s")


@pytest.mark.parametrize("result, met", [
    (MoveResult("backoff", "impact lag arm_joint3"), True),
    (MoveResult("hold", "impact effort arm_joint2"), True),
    (MoveResult("hold", "not arrived"), True),
    (MoveResult("hold", "stop"), False),
    (MoveResult("reached"), False),
])
def test_a_move_that_met_something_tripped(result, met):
    assert bench.tripped(result) is met


def test_exit_codes_say_whether_anything_moved(capsys):
    def refused() -> int:
        raise bench.Refusal("i-know", "add --i-know")

    def ended() -> int:
        raise bench.Ended("stop", "Ctrl-C")

    assert bench.run(lambda: 0) == 0
    assert bench.run(refused) == 2
    assert bench.run(ended) == 1
    assert "Refused: i-know (add --i-know)" in capsys.readouterr().err


# ====================================================================== the keyboard
class Tty:
    """The read end of a pipe that says it is a terminal."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def readline(self) -> str:
        return self._stream.readline()

    def isatty(self) -> bool:
        return True


@pytest.fixture
def terminal(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    writer = os.fdopen(write_fd, "w")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    keys = bench.Keys(Tty(reader))
    keys.start()
    yield keys, writer
    writer.close()


def type_line(writer, text: str) -> None:
    writer.write(text + "\n")
    writer.flush()


def queued(keys: bench.Keys, n: int) -> None:
    end = time.monotonic() + 5.0
    while len(keys._lines) < n and time.monotonic() < end:
        time.sleep(0.005)


def test_a_bare_enter_once_armed_is_a_stop(terminal):
    keys, writer = terminal
    keys.arm()

    type_line(writer, "")

    assert keys.stop.wait(5.0)


def test_a_line_typed_before_the_prompt_does_not_go_on(terminal):
    keys, writer = terminal
    keys.arm()
    type_line(writer, bench.CONTINUE_WORD)
    queued(keys, 1)

    assert keys.proceed("go on? ") is False
    assert keys.stop.is_set()


def test_the_word_typed_after_the_prompt_goes_on(terminal):
    keys, writer = terminal
    keys.arm()
    answer = threading.Timer(0.2, type_line, args=(writer, bench.CONTINUE_WORD))
    answer.start()

    assert keys.proceed("go on? ") is True
    answer.join()


def test_without_a_terminal_nothing_is_confirmed():
    keys = bench.Keys(io.StringIO(""))

    assert keys.confirm("Enter = go ") is False
