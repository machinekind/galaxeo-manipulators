"""The per-rig file: reading, writing and the refusal reasons."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from galaxeo.grasp.contract import (
    FREEZE_DISABLED,
    OBJECT_HEIGHT_DEFAULT,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)
from galaxeo.grasp.rig import MODES, Rig, RigError

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc).timestamp()
ORDER = ["table not taught", "table touch implausible", "grasp point not taught", "table touch far",
         "teaching old", "object height not set", "dot not checked", "dot off", "latency not measured",
         "latency high", "jaw stroke not measured", "gripper freeze undecided", "no shadow episodes"]


def stamp(hours_ago: float) -> str:
    t = datetime.fromtimestamp(NOW, timezone.utc) - timedelta(hours=hours_ago)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def complete() -> dict:
    """A rig file on which every mode may run."""
    return {
        "version": 1,
        "table": {"touch_tcp_z": 0.0342, "tcp": [0.01, 0.30, 0.0342], "joints": [90, 120, -80, 40, 0, 0],
                  "taught_at": stamp(1.0)},
        "grasp": {"tcp": [0.0, 0.33, 0.036], "yaw": 0.1, "joints": [90, 110, -75, 50, 0, 5],
                  "taught_at": stamp(0.5), "object_height": 0.030, "joint_offsets_deg": [0.0] * 6},
        "gripper": {"stroke_close_s_effort": 1.28, "stroke_close_s_operator": 1.35, "freeze_threshold": 100.0,
                    "gap_mm": {}},
        "wrist": {"source": "/dev/video2", "K": None, "dist": None, "upside_down": True, "dot_error_px": 3.1},
        "latency": {"camera_s": 0.045, "actuation_s": 0.03, "infer_p95_s": 0.02, "total_s": 0.11},
        "guard": {"effort_abs": 12.0},
        "overrides": {"shadow_done": "10 clean shadow episodes, 2026-09-27"},
    }


def write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def rig_from(data: dict) -> Rig:
    return Rig(**{k: copy.deepcopy(v) for k, v in data.items() if k != "version"})


def in_order(reasons: list[str]) -> bool:
    return [r for r in ORDER if r in reasons] == reasons and len(set(reasons)) == len(reasons)


# ------------------------------------------------------------------ the file

def test_a_missing_file_is_an_empty_rig_with_that_path(tmp_path):
    path = tmp_path / "nowhere" / "grasp.json"
    rig = Rig.load(path)
    assert rig.path == path
    assert rig.version == 1
    assert rig.to_dict() == {"version": 1, "table": {}, "grasp": {}, "gripper": {}, "wrist": {},
                             "latency": {}, "guard": {}, "overrides": {}}
    assert not path.parent.exists()


def test_save_and_load_round_trip_including_the_version(tmp_path):
    path = tmp_path / "rig" / "grasp.json"
    rig = rig_from(complete())
    rig.path = path
    rig.save()
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    back = Rig.load(path)
    assert back.to_dict() == complete()
    assert rig.to_dict() == complete()
    assert back.path == path


def test_values_measured_with_numpy_are_saved_as_plain_numbers(tmp_path):
    rig = Rig(path=tmp_path / "grasp.json")
    rig.table = {"touch_tcp_z": np.float32(0.0342), "tcp": np.array([0.01, 0.30, 0.0342]),
                 "joints": {"arm_joint1": np.float64(90.0)}, "taught_at": stamp(1.0)}
    rig.grasp = {"tcp": np.array([0.0, 0.33, 0.036]), "yaw": np.float64(0.1), "taught_at": stamp(0.5),
                 "object_height": 0.03, "joint_offsets_deg": np.zeros(6)}
    rig.wrist = {"source": np.int64(2), "upside_down": np.bool_(True), "dot_error_px": np.float32(3.0)}
    rig.save()
    back = Rig.load(rig.path)
    assert back.table["tcp"] == [0.01, 0.30, 0.0342]
    assert back.table["joints"] == {"arm_joint1": 90.0}
    assert back.touch_tcp_z() == pytest.approx(0.0342, abs=1e-8)
    assert back.wrist_source() == "2"
    assert back.grasp["joint_offsets_deg"] == [0.0] * 6
    assert back.upside_down() is True
    assert back.to_dict() == rig.to_dict()


def test_to_dict_is_a_copy():
    rig = rig_from(complete())
    d = rig.to_dict()
    d["table"]["touch_tcp_z"] = 1.0
    d["grasp"]["tcp"][0] = 5.0
    assert rig.touch_tcp_z() == 0.0342
    assert rig.grasp["tcp"][0] == 0.0


def test_an_unknown_top_level_key_raises(tmp_path):
    data = complete()
    data["reach"] = {}
    with pytest.raises(RigError, match="reach"):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_rig_error_is_a_value_error():
    assert issubclass(RigError, ValueError)


@pytest.mark.parametrize("version", [2, 0, "1", 1.5, True, None])
def test_a_file_with_another_version_raises(tmp_path, version):
    data = complete()
    data["version"] = version
    with pytest.raises(RigError, match="wersja"):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_file_without_a_version_is_read_as_version_1(tmp_path):
    data = complete()
    del data["version"]
    rig = Rig.load(write(tmp_path / "grasp.json", data))
    assert rig.version == 1
    assert rig.to_dict() == complete()


def test_save_refuses_a_malformed_value_and_keeps_the_old_file(tmp_path):
    path = write(tmp_path / "grasp.json", complete())
    before = path.read_bytes()
    rig = Rig.load(path)
    rig.table["touch_tcp_z"] = float("nan")
    with pytest.raises(RigError):
        rig.save()
    rig = Rig.load(path)
    rig.version = 2
    with pytest.raises(RigError):
        rig.save()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grasp.json"]


@pytest.mark.parametrize("section,key,value", [
    ("table", "touch_tcp_z", "0.034"), ("table", "touch_tcp_z", float("nan")), ("grasp", "tcp", [0.0, 0.33]),
    ("grasp", "yaw", True), ("grasp", "taught_at", "yesterday"), ("wrist", "upside_down", "yes"),
    ("wrist", "K", [[1, 0, 0], [0, 1, 0]]), ("gripper", "stroke_close_s_effort", 0.0),
    ("grasp", "joint_offsets_deg", [0.0] * 5), ("latency", "total_s", "fast"),
])
def test_a_malformed_value_raises_when_loading(tmp_path, section, key, value):
    data = complete()
    data[section][key] = value
    with pytest.raises(RigError):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_section_that_is_not_an_object_raises(tmp_path):
    data = complete()
    data["table"] = [0.034]
    with pytest.raises(RigError, match="musi byc obiektem"):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_file_that_is_not_json_raises(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RigError, match="JSON"):
        Rig.load(tmp_path / "bad.json")


def test_a_rig_without_a_path_is_never_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RigError, match="bez sciezki"):
        Rig.for_sim().save()
    with pytest.raises(RigError, match="bez sciezki"):
        Rig().save()
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ refusals

def test_the_modes_are_shadow_and_real():
    assert MODES == ("shadow", "real")


@pytest.mark.parametrize("mode", ["dry", "sim", "fast"])
def test_an_unknown_mode_raises(mode):
    with pytest.raises(ValueError, match="unknown mode"):
        Rig().missing(mode, simulated=False)


@pytest.mark.parametrize("simulated", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_a_complete_rig_may_run_every_mode(mode, simulated):
    assert rig_from(complete()).missing(mode, simulated=simulated, now=NOW) == []


def test_an_empty_rig_on_a_real_arm_misses_for_shadow_in_the_listed_order():
    assert Rig().missing("shadow", simulated=False, now=NOW) == [
        "table not taught", "grasp point not taught", "object height not set", "dot not checked",
        "latency not measured"]


def test_an_empty_rig_on_a_real_arm_misses_for_real_in_the_listed_order():
    assert Rig().missing("real", simulated=False, now=NOW) == [
        "table not taught", "grasp point not taught", "object height not set", "dot not checked",
        "latency not measured", "jaw stroke not measured", "gripper freeze undecided", "no shadow episodes"]


@pytest.mark.parametrize("mode", MODES)
def test_an_empty_rig_on_a_simulated_arm_misses_nothing(mode):
    assert Rig().missing(mode, simulated=True, now=NOW) == []


def test_the_wrist_camera_source_is_not_asked_for():
    data = complete()
    data["wrist"]["source"] = None
    assert rig_from(data).missing("real", simulated=False, now=NOW) == []


def test_every_limit_at_once_gives_every_reason_in_the_listed_order():
    data = complete()
    data["table"].update(touch_tcp_z=0.06, tcp=[0.13, 0.30, 0.06], taught_at=stamp(13.0))
    data["grasp"].update(yaw=None, object_height=0.2)
    data["gripper"].update(stroke_close_s_effort=None, stroke_close_s_operator=None, freeze_threshold=None)
    data["wrist"].update(dot_error_px=9.0)
    data["latency"].update(total_s=0.2)
    data["overrides"] = {}
    rig = rig_from(data)
    want = ["table touch implausible", "grasp point not taught", "table touch far", "teaching old",
            "object height not set", "dot off", "latency high", "jaw stroke not measured",
            "gripper freeze undecided", "no shadow episodes"]
    assert in_order(want)
    assert rig.missing("real", simulated=False, now=NOW) == want
    assert rig.missing("shadow", simulated=False, now=NOW) == want[:7]
    assert rig.missing("real", simulated=True, now=NOW) == []


def test_overrides_lift_the_matching_limits():
    data = complete()
    data["table"]["touch_tcp_z"] = 0.06
    data["table"]["taught_at"] = stamp(30.0)
    data["wrist"]["dot_error_px"] = 12.0
    data["latency"]["total_s"] = 0.3
    rig = rig_from(data)
    assert rig.missing("real", simulated=False, now=NOW) == [
        "table touch implausible", "teaching old", "dot off", "latency high"]
    rig.overrides.update(table="mount moved, checked by hand", taught="same day", dot="known offset",
                         latency="slow camera, accepted")
    assert rig.missing("real", simulated=False, now=NOW) == []


def test_an_empty_override_note_is_no_override():
    data = complete()
    data["wrist"]["dot_error_px"] = 12.0
    rig = rig_from(data)
    rig.overrides["dot"] = ""
    assert rig.missing("real", simulated=False, now=NOW) == ["dot off"]


def test_only_real_needs_the_shadow_episodes():
    rig = rig_from(complete())
    rig.overrides.pop("shadow_done")
    assert rig.missing("real", simulated=False, now=NOW) == ["no shadow episodes"]
    assert rig.missing("shadow", simulated=False, now=NOW) == []


def test_teaching_older_than_twelve_hours_is_refused():
    data = complete()
    data["grasp"]["taught_at"] = stamp(11.9)
    assert rig_from(data).missing("shadow", simulated=False, now=NOW) == []
    data["grasp"]["taught_at"] = stamp(12.1)
    assert rig_from(data).missing("shadow", simulated=False, now=NOW) == ["teaching old"]


def test_old_teaching_is_not_judged_without_a_clock_or_with_the_taught_override():
    data = complete()
    data["grasp"]["taught_at"] = stamp(12.1)
    rig = rig_from(data)
    assert rig.missing("shadow", simulated=False) == []
    rig.overrides["taught"] = "operator: point unchanged since yesterday"
    assert rig.missing("shadow", simulated=False, now=NOW) == []


def test_a_taught_table_without_its_stamp_counts_as_old():
    data = complete()
    data["table"]["taught_at"] = None
    assert rig_from(data).missing("shadow", simulated=False, now=NOW) == ["teaching old"]


def test_taught_age_reads_iso_stamps_with_or_without_a_zone():
    rig = rig_from(complete())
    assert rig.taught_age_s(NOW) == {"table": pytest.approx(3600.0), "grasp": pytest.approx(1800.0)}
    rig.table["taught_at"] = "2026-09-27T11:00:00+00:00"
    rig.grasp["taught_at"] = "2026-09-27T10:00:00"            # naive stamps are UTC
    assert rig.taught_age_s(NOW) == {"table": pytest.approx(3600.0), "grasp": pytest.approx(7200.0)}
    assert Rig().taught_age_s(NOW) == {"table": None, "grasp": None}


@pytest.mark.parametrize("table_tcp,far", [([0.12, 0.33, 0.0342], True), ([0.0, 0.24, 0.0342], False),
                                           (None, True)], ids=["0.12 m in x", "0.09 m in y", "no position"])
def test_a_table_taught_far_from_the_grasp_point_is_refused(table_tcp, far):
    data = complete()
    data["table"]["tcp"] = table_tcp
    want = ["table touch far"] if far else []
    assert rig_from(data).missing("shadow", simulated=False, now=NOW) == want


@pytest.mark.parametrize("height,ok", [(None, False), (0.004, False), (0.005, True), (0.03, True),
                                       (0.10, True), (0.11, False)])
@pytest.mark.parametrize("mode", MODES)
def test_object_height_must_be_set_within_limits(mode, height, ok):
    data = complete()
    data["grasp"]["object_height"] = height
    want = [] if ok else ["object height not set"]
    assert rig_from(data).missing(mode, simulated=False, now=NOW) == want


# ------------------------------------------------------------------ readers

def test_the_believed_point_is_the_taught_xy_at_touch_plus_tip_clearance():
    rig = rig_from(complete())
    np.testing.assert_array_equal(rig.believed(), [0.0, 0.33, 0.0342 + TIP_CLEAR])
    assert TIP_CLEAR == 0.006
    assert rig.grasp_yaw() == 0.1
    assert rig.touch_tcp_z() == 0.0342


def test_there_is_no_believed_point_without_the_table_or_the_grasp_point():
    rig = rig_from(complete())
    rig.table["touch_tcp_z"] = None
    assert rig.believed() is None
    assert Rig(table={"touch_tcp_z": 0.034}).believed() is None


def test_upside_down_and_object_height_are_read_from_the_file(tmp_path):
    data = complete()
    data["wrist"]["upside_down"] = False
    data["grasp"]["object_height"] = 0.045
    rig = Rig.load(write(tmp_path / "grasp.json", data))
    assert rig.upside_down() is False
    assert rig.object_height() == 0.045


def test_upside_down_defaults_to_true_and_object_height_to_none():
    assert Rig().upside_down() is True
    assert Rig().object_height() is None


def test_readers_give_the_measured_values():
    rig = rig_from(complete())
    assert rig.stroke_s() == 1.35
    assert rig.jaw_units_s() == pytest.approx(100.0 / 1.35)
    assert rig.freeze_threshold() == 100.0
    assert rig.wrist_source() == "/dev/video2"
    assert rig.camera_latency_s() == 0.045
    assert rig.effort_abs() == 12.0


def test_readers_of_an_empty_rig_give_defaults_and_none():
    empty = Rig()
    assert empty.stroke_s() is None
    assert empty.jaw_units_s() is None
    assert empty.freeze_threshold() is None
    assert empty.camera_latency_s() == 0.0
    assert empty.effort_abs() == 12.0
    assert empty.wrist_source() is None
    assert empty.touch_tcp_z() is None
    assert empty.grasp_yaw() is None


def test_the_wrist_source_is_a_string_and_a_blank_one_is_none():
    assert Rig(wrist={"source": 2}).wrist_source() == "2"
    assert Rig(wrist={"source": "  "}).wrist_source() is None


def test_one_recorded_stroke_is_the_stroke():
    assert Rig(gripper={"stroke_close_s_effort": 1.3}).stroke_s() == 1.3


def test_intrinsics_default_to_the_contract_and_are_a_copy():
    lens = Rig().intrinsics()
    np.testing.assert_array_equal(lens.K, WRIST_K_2026_09_26)
    np.testing.assert_array_equal(lens.dist, WRIST_DIST_2026_09_26)
    lens.K[0, 0] = 1.0
    assert WRIST_K_2026_09_26[0, 0] != 1.0


def test_intrinsics_are_the_rigs_own_pair():
    own = Rig(wrist={"K": [[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]], "dist": [0.1, 0.0, 0.0, 0.0]})
    lens = own.intrinsics()
    assert lens.K.shape == (3, 3)
    assert lens.K[0, 2] == 960
    assert lens.dist.tolist() == [0.1, 0.0, 0.0, 0.0]


def test_half_a_calibration_is_refused():
    with pytest.raises(RigError, match="razem"):
        Rig(wrist={"K": [[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]]}).intrinsics()


def test_for_sim_holds_every_value_a_simulated_arm_reads_and_nothing_measured():
    rig = Rig.for_sim()
    assert rig.path is None
    assert rig.touch_tcp_z() == TOUCH_EXPECTED_Z
    assert rig.object_height() == OBJECT_HEIGHT_DEFAULT
    assert rig.camera_latency_s() == 0.0
    assert "stroke_close_s_operator" in rig.gripper
    assert rig.stroke_s() is None
    assert rig.freeze_threshold() == FREEZE_DISABLED
    assert rig.wrist == {}
    assert rig.overrides == {}
    assert rig.believed() is None
    assert rig.grasp_yaw() is None


@pytest.mark.parametrize("mode", MODES)
def test_for_sim_runs_every_mode_on_a_simulated_arm(mode):
    assert Rig.for_sim().missing(mode, simulated=True, now=NOW) == []
