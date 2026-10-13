"""grasp_bench.py on fakes only: the A1X on a fake CAN bus with a virtual clock, the simulated arm
(SimTransport), scripted keys and a fake wrist camera (tests/grasp_bench_fakes.py). Nothing opens a CAN
link or a camera; every file goes under tmp_path."""

from __future__ import annotations

import io
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
yaml = pytest.importorskip("yaml")
cv2 = pytest.importorskip("cv2")

import grasp_bench as gb  # noqa: E402
from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import Kinematics  # noqa: E402
from galaxeo.grasp.contract import TIP_CLEAR, TOUCH_EXPECTED_Z, TOUCH_TOLERANCE  # noqa: E402
from galaxeo.grasp.camera import HAND_BODY, wrist_intrinsics  # noqa: E402
from galaxeo.grasp.camera import WristCamera as GeometryCamera  # noqa: E402
from galaxeo.grasp.hover import joint_box, solve_hover  # noqa: E402
from galaxeo.grasp.rig import Rig  # noqa: E402
from galaxeo.robot import ROBOT_DIR  # noqa: E402
from grasp_bench_fakes import (  # noqa: E402
    BenchBus,
    ScriptedKeys,
    TimedPolicy,
    Timing,
    WristCamera,
    deps,
    jaw_pdes,
)

KIN = Kinematics()
FLOOR = 0.036                                  # TCP z where the fake's fingertips stop
START_Z = 0.07
START = gb.start_pose(KIN, 0.0, 0.33, START_Z, 0.0)
#: From START_Z to the floor in 1 mm steps below 6 cm, two more steps on it (the height rule), then
#: two steps back down after the 2 mm backoff, and store.
TO_THE_FLOOR = ["d 5", "d 5"] + ["d 1"] * 26
STEP_DOWN_AND_STORE = ["d 1", "d 1", "ok"]
HOVER_POSE = gb.hover_over(KIN, [0.0, 0.33, 0.034 + TIP_CLEAR], 0.0)
POLICY = TimedPolicy(Timing(first_ms=20.0, median_ms=10.0, p95_ms=12.0))


def bench_bus(**kw) -> BenchBus:
    return BenchBus(start_deg=np.degrees(START.q), **kw)


def real(tool: str, rig: Path | None, *extra: str) -> list[str]:
    argv = [tool, "--i-know", "--iface", "fake", *extra]
    return argv if rig is None else [*argv, "--rig", str(rig)]


def ids(bus: BenchBus) -> set[int]:
    return {s.can_id for s in bus.tx}


def last_jaw_pdes(bus: BenchBus) -> float:
    return struct.unpack_from(">h", bus.frames(P.GRIP_ID)[-1], 0)[0] / P.POS_SCALE


def runs_of(bus: BenchBus, can_id: int) -> int:
    """Unbroken runs of `can_id` frames in what went out."""
    runs = 0
    inside = False
    for sent in bus.tx:
        if sent.can_id == can_id and not inside:
            runs += 1
        inside = sent.can_id == can_id
    return runs


def taught_rig(path: Path) -> Rig:
    rig = Rig.load(path)
    rig.table.update({"touch_tcp_z": 0.034, "tcp": [0.0, 0.33, 0.034], "taught_at": gb.now_iso()})
    rig.grasp.update({"tcp": [0.0, 0.33, 0.04], "yaw": 0.0, "taught_at": gb.now_iso(), "object_height": 0.03})
    rig.save()
    return rig


def no_station(a, settings):
    raise AssertionError("the link was opened")


# ====================================================================== the rig file, no arm
def test_dot_writes_the_error_into_the_rig_file(tmp_path):
    path = tmp_path / "rig.json"

    code = gb.main(["dot", "--rig", str(path), "--error-px", "3.5"], deps(None, ScriptedKeys()))

    assert code == 0
    assert Rig.load(path).wrist["dot_error_px"] == 3.5


def test_a_dot_error_over_the_limit_is_written_and_refused_by_the_rig(tmp_path):
    path = tmp_path / "rig.json"

    gb.main(["dot", "--rig", str(path), "--error-px", "9"], deps(None, ScriptedKeys()))

    assert "dot off" in Rig.load(path).missing("shadow", simulated=False)


def test_dot_without_a_rig_file_is_refused(tmp_path):
    code = gb.main(["dot", "--error-px", "3.5"], deps(None, ScriptedKeys()))

    assert code == 2


def test_a_negative_dot_error_is_refused_and_nothing_written(tmp_path):
    path = tmp_path / "rig.json"

    code = gb.main(["dot", "--rig", str(path), "--error-px", "-1"], deps(None, ScriptedKeys()))

    assert code == 2
    assert not path.exists()


def test_show_lists_what_each_mode_misses(tmp_path, capsys):
    path = tmp_path / "rig.json"

    gb.main(["show", "--rig", str(path)], deps(None, ScriptedKeys()))

    out = capsys.readouterr().out
    assert "table not taught" in out
    assert "jaw stroke not measured" in out


def test_set_freeze_records_the_decision(tmp_path):
    path = tmp_path / "rig.json"

    code = gb.main(["gripper", "--rig", str(path), "--set-freeze", "100"], deps(None, ScriptedKeys()))

    assert code == 0
    assert Rig.load(path).freeze_threshold() == 100.0


def test_gap_mm_records_the_ruler_readings(tmp_path):
    path = tmp_path / "rig.json"

    gb.main(["gripper", "--rig", str(path), "--gap-mm", "0:3", "100:103"], deps(None, ScriptedKeys()))

    assert Rig.load(path).gripper["gap_mm"] == {"0": 3.0, "100": 103.0}


def test_a_gap_off_the_jaw_model_is_called_out(tmp_path, capsys):
    path = tmp_path / "rig.json"

    gb.main(["gripper", "--rig", str(path), "--gap-mm", "50:60"], deps(None, ScriptedKeys()))

    assert "UWAGA" in capsys.readouterr().out


def test_a_malformed_gap_is_refused(tmp_path):
    path = tmp_path / "rig.json"

    code = gb.main(["gripper", "--rig", str(path), "--gap-mm", "50-60"], deps(None, ScriptedKeys()))

    assert code == 2


# ====================================================================== refusals before anything moves
@pytest.mark.parametrize("tool", ["gripper", "table", "point", "latency"])
def test_the_real_arm_needs_i_know_before_the_link_opens(tool, tmp_path):
    rig = tmp_path / "rig.json"
    taught_rig(rig)
    argv = [tool, "--iface", "fake", "--rig", str(rig)]
    if tool == "latency":
        argv += ["--student", "x.pt"]

    code = gb.main(argv, gb.Deps(open_station=no_station, keys=ScriptedKeys))

    assert code == 2


def test_latency_on_the_simulated_arm_is_refused(tmp_path):
    rig = tmp_path / "rig.json"
    taught_rig(rig)

    code = gb.main(["latency", "--sim", "--rig", str(rig), "--student", "x.pt"], gb.Deps(open_station=no_station))

    assert code == 2


def test_latency_without_a_taught_point_is_refused(tmp_path):
    code = gb.main(real("latency", tmp_path / "rig.json", "--student", "x.pt"), gb.Deps(open_station=no_station))

    assert code == 2


def test_an_unreachable_start_point_is_refused_before_anything_is_sent():
    bus = bench_bus()

    code = gb.main(real("table", None, "--xy", "0.9", "0.9"), deps(bus, ScriptedKeys()))

    assert code == 2
    assert bus.tx == []


def test_without_the_operators_enter_nothing_moves():
    bus = bench_bus()

    code = gb.main(real("table", None, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(confirms=False)))

    assert code == 2
    assert P.CMD_ID not in ids(bus)


# ====================================================================== jaw
def cube_at_the_prompt(bus: BenchBus) -> ScriptedKeys:
    """The operator puts a 3 cm cube between the plates when asked: the jaw stops at opening 30."""
    return ScriptedKeys(on_proceed=lambda: setattr(bus, "cube_pdes", jaw_pdes(30.0)))


@pytest.fixture
def jaw_run(tmp_path):
    bus = bench_bus()
    path = tmp_path / "rig.json"
    code = gb.main(real("gripper", path), deps(bus, cube_at_the_prompt(bus)))
    return code, Rig.load(path), bus


def test_the_jaw_tool_ends_cleanly(jaw_run):
    assert jaw_run[0] == 0


def test_the_close_stroke_is_the_streams_full_stroke(jaw_run):
    # The default stream: 2 p_des/s over the 2.03 p_des between closed and open.
    assert jaw_run[1].gripper["stroke_close_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


def test_the_open_stroke_is_recorded(jaw_run):
    assert jaw_run[1].gripper["stroke_open_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


def test_the_free_and_loaded_effort_peaks_are_recorded(jaw_run):
    assert jaw_run[1].gripper["free_effort_peak"] == pytest.approx(0.3, abs=0.01)
    assert jaw_run[1].gripper["loaded_effort_peak"] == pytest.approx(5.0, abs=0.01)


def test_a_cube_that_stands_out_suggests_a_freeze_threshold(jaw_run):
    assert jaw_run[1].gripper["suggested_threshold"] == pytest.approx(2.65, abs=0.01)


def test_the_rig_reads_the_jaw_stroke_back(jaw_run):
    assert jaw_run[1].stroke_s() == pytest.approx(2.03 / 2.0, abs=0.03)


def test_the_jaw_tool_never_commands_the_arm(jaw_run):
    assert ids(jaw_run[2]) == {P.GRIP_ID}


def test_the_jaw_tool_keeps_the_raw_traces_next_to_the_rig_file(jaw_run, tmp_path):
    assert len(list(tmp_path.glob("rig_gripper_*.npz"))) == 1


def test_the_jaw_tool_without_a_rig_file_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bus = bench_bus()

    gb.main(real("gripper", None), deps(bus, cube_at_the_prompt(bus)))

    assert list(tmp_path.iterdir()) == []


def test_a_stop_mid_close_leaves_the_jaw_where_its_stream_was(tmp_path):
    bus = bench_bus()
    path = tmp_path / "rig.json"
    opened: list[gb.Station] = []

    def stop_past_half() -> None:
        # What a bare Enter or Ctrl-C ends in, at the tick the close passes opening 50.
        if bus.g_des is not None and bus.g_des > jaw_pdes(50.0) and not opened[0].stopped.is_set():
            opened[0].stop()

    bus.on_tick = stop_past_half
    code = gb.main(real("gripper", path), deps(bus, ScriptedKeys(), opened=opened))

    assert code == 1
    assert last_jaw_pdes(bus) < jaw_pdes(30.0)       # more open than 30: not closed
    assert Rig.load(path).gripper == {}


def test_the_jaw_tool_runs_on_the_simulated_arm(tmp_path):
    path = tmp_path / "rig.json"

    code = gb.main(["gripper", "--sim", "--rig", str(path)], deps(None, ScriptedKeys()))

    assert code == 0
    assert Rig.load(path).gripper["stroke_close_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


# ====================================================================== teaching by jogging
@pytest.fixture
def table_run(tmp_path, capsys):
    bus = bench_bus(floor_z=FLOOR)
    path = tmp_path / "rig.json"
    keys = ScriptedKeys(lines=TO_THE_FLOOR + STEP_DOWN_AND_STORE)
    code = gb.main(real("table", path, "--start-z", str(START_Z)), deps(bus, keys))
    return code, Rig.load(path), bus, capsys.readouterr().out


def test_teaching_the_table_ends_with_ok(table_run):
    assert table_run[0] == 0


def test_the_table_touch_is_where_the_fingertips_stopped(table_run):
    assert table_run[1].touch_tcp_z() == pytest.approx(FLOOR, abs=0.001)


def test_the_taught_table_is_no_longer_missing(table_run):
    assert "table not taught" not in table_run[1].missing("shadow", simulated=False, now=time.time())


def test_the_height_rule_calls_the_contact(table_run):
    assert "Kontakt prawdopodobny (wysokosc" in table_run[3]


def test_teaching_sends_no_function_frame(table_run):
    assert P.FF_ID not in ids(table_run[2])


def test_a_contact_backs_off_two_millimetres():
    bus = bench_bus(floor_z=FLOOR)

    gb.main(real("table", None, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=TO_THE_FLOOR + ["q"])))

    assert bus.tcp_z() == pytest.approx(FLOOR + gb.JOG_BACKOFF, abs=0.0005)


def test_pressing_on_the_table_trips_the_impact_guard(capsys):
    bus = bench_bus(floor_z=FLOOR, press_effort=15.0)

    gb.main(real("table", None, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=TO_THE_FLOOR + ["q"])))

    assert "straznik uderzen: impact effort arm_joint2" in capsys.readouterr().out


def test_a_touch_far_from_the_expected_height_is_not_saved(tmp_path):
    bus = bench_bus(floor_z=0.05)
    path = tmp_path / "rig.json"
    lines = ["d 5", "d 5"] + ["d 1"] * 12 + STEP_DOWN_AND_STORE + ["q"]

    code = gb.main(real("table", path, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=lines)))

    assert code == 2
    assert Rig.load(path).table == {}


def test_force_saves_a_far_touch_with_the_operators_note(tmp_path):
    bus = bench_bus(floor_z=0.05)
    path = tmp_path / "rig.json"
    lines = ["d 5", "d 5"] + ["d 1"] * 12 + STEP_DOWN_AND_STORE

    gb.main(real("table", path, "--start-z", str(START_Z), "--force"), deps(bus, ScriptedKeys(lines=lines)))

    assert Rig.load(path).overrides["table"].startswith("--force")


def test_a_bare_enter_stops_the_teaching(tmp_path):
    bus = bench_bus()
    path = tmp_path / "rig.json"

    code = gb.main(real("table", path, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=["d 5", ""])))

    assert code == 1
    assert not path.exists()


def test_g_moves_the_jaw_while_the_arm_holds():
    bus = bench_bus()

    gb.main(real("point", None, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=["g 40", "q"])))

    assert last_jaw_pdes(bus) == pytest.approx(jaw_pdes(40.0), abs=2 / P.POS_SCALE)


def test_the_arm_never_streams_during_a_jaw_phase():
    bus = bench_bus()

    gb.main(real("point", None, "--start-z", str(START_Z)), deps(bus, ScriptedKeys(lines=["d 5", "g 40", "d 5", "q"])))

    assert runs_of(bus, P.GRIP_ID) == 2                            # opening at the start pose, then g 40


def test_teaching_the_point_stores_the_tcp_yaw_and_object(tmp_path):
    bus = bench_bus()
    path = tmp_path / "rig.json"

    gb.main(real("point", path, "--start-z", str(START_Z), "--object-height", "0.025"),
        deps(bus, ScriptedKeys(lines=["d 5", "ok"])))

    rig = Rig.load(path)
    assert rig.grasp["tcp"][2] == pytest.approx(START_Z - 0.005, abs=0.001)
    assert abs(gb.wrap(rig.grasp_yaw() - START.yaw)) < 0.01
    assert rig.object_height() == 0.025
    assert rig.grasp["joint_offsets_deg"] == [0.0] * 6


def test_the_simulated_arm_finds_the_table(tmp_path):
    path = tmp_path / "rig.json"
    lines = ["d 5"] * 13 + ["d 1"] * 30 + STEP_DOWN_AND_STORE

    code = gb.main(["table", "--sim", "--rig", str(path)], deps(None, ScriptedKeys(lines=lines)))

    assert code == 0
    assert Rig.load(path).touch_tcp_z() == pytest.approx(TOUCH_EXPECTED_Z, abs=TOUCH_TOLERANCE / 5)


def test_the_simulated_arm_writes_nothing_without_a_rig_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    code = gb.main(["table", "--sim"], deps(None, ScriptedKeys(lines=["d 5", "ok"])))

    assert code == 0
    assert list(tmp_path.iterdir()) == []


# ====================================================================== latency
@pytest.fixture
def latency_run(tmp_path):
    bus = BenchBus(start_deg=np.degrees(HOVER_POSE.q))
    camera = WristCamera(bus)
    path = tmp_path / "rig.json"
    taught_rig(path)
    code = gb.main(real("latency", path, "--student", "x.pt", "--camera", "0"),
               deps(bus, ScriptedKeys(), camera=camera, policy=POLICY))
    return code, Rig.load(path), camera


def test_latency_ends_cleanly(latency_run):
    assert latency_run[0] == 0


def test_the_camera_latency_is_the_cameras_delay_within_a_frame(latency_run):
    camera = latency_run[2]
    assert camera.delay - 0.01 <= latency_run[1].latency["camera_s"] <= camera.delay + camera.period + 0.01


def test_the_arm_latency_is_a_few_stream_ticks(latency_run):
    assert latency_run[1].latency["actuation_s"] < 0.03


def test_the_inference_p95_is_the_policys(latency_run):
    assert latency_run[1].latency["infer_p95_s"] == 0.012


def test_the_total_is_the_sum_of_the_three(latency_run):
    lat = latency_run[1].latency
    assert lat["total_s"] == pytest.approx(lat["camera_s"] + lat["infer_p95_s"] + lat["actuation_s"], abs=2e-4)


def test_the_measured_latency_satisfies_the_rig(latency_run):
    assert "latency not measured" not in latency_run[1].missing("shadow", simulated=False)


def test_latency_closes_the_camera(latency_run):
    assert latency_run[2].closed


def test_a_camera_that_is_not_full_hd_is_refused_before_the_arm_moves(tmp_path):
    bus = BenchBus(start_deg=np.degrees(HOVER_POSE.q))
    path = tmp_path / "rig.json"
    taught_rig(path)

    code = gb.main(real("latency", path, "--student", "x.pt", "--camera", "0"),
               deps(bus, ScriptedKeys(), camera=WristCamera(bus, size=(1280, 720)), policy=POLICY))

    assert code == 2
    assert P.CMD_ID not in ids(bus)


def test_the_jaw_runs_as_the_twins_config_says(tmp_path):
    config = tmp_path / "local.yaml"
    config.write_text("a1x:\n  gripper_rate_pdes_s: 1.5\n", encoding="utf-8")
    bus = bench_bus(grip_rate=1.5)
    path = tmp_path / "rig.json"

    gb.main(real("gripper", path, "--config", str(config)), deps(bus, cube_at_the_prompt(bus)))

    assert Rig.load(path).gripper["stroke_close_s_effort"] == pytest.approx(2.03 / 1.5, abs=0.03)


def test_a_config_without_an_a1x_section_changes_nothing(tmp_path):
    config = tmp_path / "local.yaml"
    config.write_text("safety:\n  velocity_scale: 0.5\n", encoding="utf-8")

    assert gb.a1x_settings(str(config)) == {}


# ====================================================================== dot capture
@pytest.fixture
def dot_run(tmp_path):
    bus = BenchBus(start_deg=np.degrees(HOVER_POSE.q))
    path = tmp_path / "rig.json"
    taught_rig(path)
    out = tmp_path / "dot" / "dot.png"
    code = gb.main(real("dot", path, "--out", str(out), "--camera", "0"),
                   deps(bus, ScriptedKeys(), camera=WristCamera(bus)))
    return code, out, bus, path


def test_the_dot_capture_ends_cleanly(dot_run):
    assert dot_run[0] == 0


def test_the_dot_image_is_the_model_input_four_times_enlarged(dot_run):
    assert cv2.imread(str(dot_run[1])).shape == (480, 640, 3)


def test_the_dot_image_marks_the_believed_grasp_where_it_projects(dot_run):
    bus = dot_run[2]
    rig = Rig.load(dot_run[3])
    hand = KIN.link_pose(np.array(bus.q[:6]), HAND_BODY)
    u, v = GeometryCamera(wrist_intrinsics(rig)).believed_uv(hand, rig.believed())
    pixel = cv2.imread(str(dot_run[1]))[round((v + 0.5) * 4 - 0.5), round((u + 0.5) * 4 - 0.5)]

    assert int(pixel[1]) - max(int(pixel[0]), int(pixel[2])) > 200      # the cross's green, antialiased


def test_the_dot_capture_records_nothing(dot_run):
    assert "dot_error_px" not in Rig.load(dot_run[3]).wrist


def test_the_dot_capture_needs_an_output_file(tmp_path):
    path = tmp_path / "rig.json"
    taught_rig(path)

    code = gb.main(real("dot", path, "--camera", "0"), gb.Deps(open_station=no_station))

    assert code == 2


def test_the_dot_capture_moves_the_arm_only_with_i_know(tmp_path):
    path = tmp_path / "rig.json"
    taught_rig(path)

    code = gb.main(["dot", "--iface", "fake", "--rig", str(path), "--out", str(tmp_path / "dot.png")],
                   gb.Deps(open_station=no_station))

    assert code == 2


def test_the_dot_capture_on_the_simulated_arm_is_refused(tmp_path):
    path = tmp_path / "rig.json"
    taught_rig(path)

    code = gb.main(["dot", "--sim", "--rig", str(path), "--out", str(tmp_path / "dot.png")],
                   gb.Deps(open_station=no_station))

    assert code == 2


def test_the_dot_capture_needs_the_taught_point(tmp_path):
    code = gb.main(real("dot", tmp_path / "rig.json", "--out", str(tmp_path / "dot.png"), "--camera", "0"),
                   gb.Deps(open_station=no_station))

    assert code == 2


# ====================================================================== the pieces
def test_the_camera_reads_frames_through_opencv(tmp_path):
    video = tmp_path / "wrist.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for k in range(10):
        writer.write(np.full((48, 64, 3), 20 * k, np.uint8))
    writer.release()

    camera = gb.Camera(str(video), time.monotonic)
    end = time.monotonic() + 5.0
    while camera.latest() is None and time.monotonic() < end:
        time.sleep(0.01)
    frame = camera.latest()
    camera.close()

    assert frame.size == (64, 48)
    assert frame.image.shape == (gb.FRAME_SMALL[1], gb.FRAME_SMALL[0])


def test_the_bench_hovers_where_the_strategy_does():
    limits = yaml.safe_load((ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))["limits"]
    strategy = solve_hover(KIN, [0.0, 0.33, 0.04], 0.0, joint_box(KIN, limits))

    np.testing.assert_allclose(gb.hover_over(KIN, [0.0, 0.33, 0.04], 0.0).q, strategy.q)


def test_a_stroke_of_the_whole_range_in_a_second_is_a_one_second_stroke():
    t = np.arange(-0.5, 2.0, 0.005)
    streamed = np.clip(100.0 - 100.0 * t, 0.0, 100.0)
    log = gb.EffortLog(t=t, effort=np.zeros_like(t), streamed=streamed, commanded=np.zeros_like(t),
                       frozen=np.zeros(len(t), bool), rate_hz=200.0, valid=True, target=0.0, t0=0.0)

    assert gb.stroke_from_log(log).stream_s == pytest.approx(1.0, abs=0.01)


def test_a_move_in_which_the_freeze_fired_does_not_count():
    t = np.arange(-0.5, 2.0, 0.005)
    effort = np.where((t > 0.0) & (t < 0.6), 0.3, 0.0)
    log = gb.EffortLog(t=t, effort=effort, streamed=np.clip(100.0 - 100.0 * t, 40.0, 100.0),
                       commanded=np.zeros_like(t), frozen=t > 0.6, rate_hz=200.0, valid=False, target=0.0, t0=0.0)

    assert gb.gripper_summary([log], []) == {}


def test_the_operators_stop_time_is_recorded_per_direction():
    t = np.arange(-0.5, 2.0, 0.005)
    log = gb.EffortLog(t=t, effort=np.zeros_like(t), streamed=np.clip(100.0 - 100.0 * t, 0.0, 100.0),
                       commanded=np.zeros_like(t), frozen=np.zeros(len(t), bool), rate_hz=200.0, valid=True,
                       target=0.0, t0=0.0, operator_s=1.3)

    assert gb.gripper_summary([log], [])["stroke_close_s_operator"] == 1.3


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
    keys = gb.Keys(Tty(reader))
    keys.start()
    yield keys, writer
    writer.close()


def type_line(writer, text: str) -> None:
    writer.write(text + "\n")
    writer.flush()


def queued(keys: gb.Keys, n: int) -> None:
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
    type_line(writer, gb.CONTINUE_WORD)
    queued(keys, 1)

    assert keys.proceed("dalej? ") is False
    assert keys.stop.is_set()


def test_the_word_typed_after_the_prompt_goes_on(terminal):
    keys, writer = terminal
    keys.arm()
    answer = threading.Timer(0.2, type_line, args=(writer, gb.CONTINUE_WORD))
    answer.start()

    assert keys.proceed("dalej? ") is True
    answer.join()
