"""gripper_a1x.py on fakes only: the A1X on a fake CAN bus with a virtual clock, the simulated jaw and
scripted keys (tests/bench_fakes.py). Nothing opens a CAN link; every file goes under tmp_path."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

import gripper_a1x as gt  # noqa: E402
from bench_fakes import BenchBus, ScriptedKeys, bench_station, jaw_pdes  # noqa: E402
from galaxeo import protocol as P  # noqa: E402

START_DEG = (90.0, 120.0, -77.0, 47.0, 0.0, 0.0)


def bench_bus(**kw) -> BenchBus:
    return BenchBus(start_deg=START_DEG, **kw)


def real(*extra: str) -> list[str]:
    return ["--i-know", "--iface", "fake", *extra]


def deps(bus: BenchBus | None, keys: ScriptedKeys, *, opened: list | None = None) -> gt.Deps:
    """The tool's outside world as fakes; without a bus the default station (the simulated arm) runs.
    `opened` collects the stations the run opens."""
    def station(a, settings):
        st = bench_station(bus, settings) if bus is not None else gt.open_station(a, settings)
        if opened is not None:
            opened.append(st)
        return st

    return gt.Deps(open_station=station, keys=lambda: keys)


def no_station(a, settings):
    raise AssertionError("the link was opened")


def summary(path: Path) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["summary"]))


def last_jaw_pdes(bus: BenchBus) -> float:
    return struct.unpack_from(">h", bus.frames(P.GRIP_ID)[-1], 0)[0] / P.POS_SCALE


def cube_at_the_prompt(bus: BenchBus) -> ScriptedKeys:
    """The operator puts a 3 cm cube between the plates when asked: the jaw stops at opening 30."""
    return ScriptedKeys(on_proceed=lambda: setattr(bus, "cube_pdes", jaw_pdes(30.0)))


# ====================================================================== refusals before anything moves
def test_the_real_arm_needs_i_know_before_the_link_opens():
    code = gt.main(["--iface", "fake"], gt.Deps(open_station=no_station, keys=ScriptedKeys))

    assert code == 2


def test_a_config_that_enables_the_motors_on_connect_is_refused_before_the_link_opens(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "local.yaml"
    config.write_text("a1x:\n  enable_on_connect: true\n", encoding="utf-8")

    code = gt.main(real("--config", str(config)), gt.Deps(open_station=no_station, keys=ScriptedKeys))

    assert code == 2


def test_without_the_operators_enter_nothing_moves():
    bus = bench_bus()

    code = gt.main(real(), deps(bus, ScriptedKeys(confirms=False)))

    assert code == 2
    assert bus.frames(P.GRIP_ID) == []


# ====================================================================== a full run
@pytest.fixture
def jaw_run(tmp_path):
    bus = bench_bus()
    out = tmp_path / "jaw.npz"
    code = gt.main(real("--out", str(out)), deps(bus, cube_at_the_prompt(bus)))
    return code, summary(out), bus


def test_the_jaw_tool_ends_cleanly(jaw_run):
    assert jaw_run[0] == 0


def test_the_close_stroke_is_the_streams_full_stroke(jaw_run):
    # The default stream: 2 p_des/s over the 2.03 p_des between closed and open.
    assert jaw_run[1]["stroke_close_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


def test_the_open_stroke_is_recorded(jaw_run):
    assert jaw_run[1]["stroke_open_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


def test_the_free_and_loaded_effort_peaks_are_recorded(jaw_run):
    assert jaw_run[1]["free_effort_peak"] == pytest.approx(0.3, abs=0.01)
    assert jaw_run[1]["loaded_effort_peak"] == pytest.approx(5.0, abs=0.01)


def test_a_cube_that_stands_out_suggests_a_freeze_threshold(jaw_run):
    assert jaw_run[1]["separable"] is True
    assert jaw_run[1]["suggested_threshold"] == pytest.approx(2.65, abs=0.01)


def test_the_jaw_tool_never_commands_the_arm(jaw_run):
    assert {s.can_id for s in jaw_run[2].tx} == {P.GRIP_ID}


def test_the_traces_hold_every_move(tmp_path):
    bus = bench_bus()
    out = tmp_path / "traces" / "jaw"

    gt.main(real("--out", str(out)), deps(bus, cube_at_the_prompt(bus)))

    with np.load(tmp_path / "traces" / "jaw.npz") as z:
        assert z["part"].tolist() == ["free"] * 6 + ["cube"] * 2
        assert z["target"].tolist() == [*gt.GRIPPER_FREE_MOVES, *gt.GRIPPER_LOADED_MOVES]
        assert len(z["effort_7"]) == len(z["t_7"]) > 0


def test_the_summary_is_printed(tmp_path, capsys):
    bus = bench_bus()

    gt.main(real(), deps(bus, cube_at_the_prompt(bus)))

    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("Summary: "))
    assert json.loads(line.removeprefix("Summary: "))["suggested_threshold"] == pytest.approx(2.65, abs=0.01)


def test_the_jaw_tool_without_out_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bus = bench_bus()

    gt.main(real(), deps(bus, cube_at_the_prompt(bus)))

    assert list(tmp_path.iterdir()) == []


def test_the_freeze_is_off_while_measuring_whatever_the_config_says(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "local.yaml"
    config.write_text("a1x:\n  gripper_force: 1.2\n", encoding="utf-8")
    bus = bench_bus()
    opened: list = []

    gt.main(real("--config", str(config)), deps(bus, cube_at_the_prompt(bus), opened=opened))

    assert opened[0].jaw.backend.a1x.cfg.gripper_force == gt.FREEZE_OFF


def test_the_jaw_runs_as_the_twins_config_says(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "local.yaml"
    config.write_text("a1x:\n  gripper_rate_pdes_s: 1.5\n", encoding="utf-8")
    bus = bench_bus(grip_rate=1.5)
    out = tmp_path / "jaw.npz"

    gt.main(real("--config", str(config), "--out", str(out)), deps(bus, cube_at_the_prompt(bus)))

    assert summary(out)["stroke_close_s_effort"] == pytest.approx(2.03 / 1.5, abs=0.03)


def test_a_stop_mid_close_leaves_the_jaw_where_its_stream_was(tmp_path):
    bus = bench_bus()
    out = tmp_path / "jaw.npz"
    opened: list = []

    def stop_past_half() -> None:
        # What a bare Enter or Ctrl-C ends in, at the tick the close passes opening 50.
        if bus.g_des is not None and bus.g_des > jaw_pdes(50.0) and not opened[0].stopped.is_set():
            opened[0].stop()

    bus.on_tick = stop_past_half
    code = gt.main(real("--out", str(out)), deps(bus, ScriptedKeys(), opened=opened))

    assert code == 1
    assert last_jaw_pdes(bus) < jaw_pdes(30.0)       # more open than 30: not closed
    assert "stroke_close_s_effort" not in summary(out)


def test_the_jaw_tool_runs_on_the_simulated_arm(tmp_path):
    out = tmp_path / "jaw.npz"

    code = gt.main(["--sim", "--out", str(out)], deps(None, ScriptedKeys()))

    assert code == 0
    assert summary(out)["stroke_close_s_effort"] == pytest.approx(2.03 / 2.0, abs=0.03)


# ====================================================================== the arithmetic
def test_a_stroke_of_the_whole_range_in_a_second_is_a_one_second_stroke():
    t = np.arange(-0.5, 2.0, 0.005)
    streamed = np.clip(100.0 - 100.0 * t, 0.0, 100.0)
    log = gt.EffortLog(t=t, effort=np.zeros_like(t), streamed=streamed, commanded=np.zeros_like(t),
                       frozen=np.zeros(len(t), bool), rate_hz=200.0, valid=True, target=0.0, t0=0.0)

    assert gt.stroke_from_log(log).stream_s == pytest.approx(1.0, abs=0.01)


def test_a_move_in_which_the_freeze_fired_does_not_count():
    t = np.arange(-0.5, 2.0, 0.005)
    effort = np.where((t > 0.0) & (t < 0.6), 0.3, 0.0)
    log = gt.EffortLog(t=t, effort=effort, streamed=np.clip(100.0 - 100.0 * t, 40.0, 100.0),
                       commanded=np.zeros_like(t), frozen=t > 0.6, rate_hz=200.0, valid=False, target=0.0, t0=0.0)

    assert gt.gripper_summary([log], []) == {}


def test_the_operators_stop_time_is_recorded_per_direction():
    t = np.arange(-0.5, 2.0, 0.005)
    log = gt.EffortLog(t=t, effort=np.zeros_like(t), streamed=np.clip(100.0 - 100.0 * t, 0.0, 100.0),
                       commanded=np.zeros_like(t), frozen=np.zeros(len(t), bool), rate_hz=200.0, valid=True,
                       target=0.0, t0=0.0, operator_s=1.3)

    assert gt.gripper_summary([log], [])["stroke_close_s_operator"] == 1.3


def test_a_freeze_flag_left_from_the_last_close_is_not_a_freeze_in_this_move():
    t = np.arange(-0.5, 2.0, 0.005)

    assert gt._froze(t, t < 0.1) is False
    assert gt._froze(t, t > 0.6) is True
