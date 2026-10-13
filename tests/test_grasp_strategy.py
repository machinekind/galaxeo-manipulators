"""The grasp strategy against doubles of the twin: observation, run info, frames, objects and an arm.

Nothing here imports the twin. `Observation`, `RunInfo`, `Frame` and `ObjectPose` are dataclass
doubles with the attributes the strategy reads. `FakeArm` follows the targets the way the
twin's supervisor (rate limit into the command) and a servo (the joint at the command) would.
The policy is a scripted double, except in the replay of training's recording.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest
from grasp_recording import npz

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cv2")

from galaxeo.arm import CollisionChecker, Kinematics  # noqa: E402
from galaxeo.grasp import strategy as S  # noqa: E402
from galaxeo.grasp.command import GraspCommand  # noqa: E402
from galaxeo.grasp.contract import (  # noqa: E402
    ARM_BASE_WORLD,
    CHECKPOINT_SHA256,
    GAP_CLOSED,
    GAP_PER_UNIT,
    MAX_TICKS,
    RAW_H,
    RAW_W,
    SHADOW_FLOOR_ABOVE_BELIEVED,
    TICK_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
)
from galaxeo.grasp.hover import Hover, joint_box, solve_hover  # noqa: E402
from galaxeo.grasp.policy import GraspStudentPolicy  # noqa: E402
from galaxeo.grasp.strategy import Grasp, GraspRefused, GraspSettings, JointHistory, Move, Phase  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_ENV = "GALAXEO_GRASP_STUDENT"

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
JOINTS = (*ARM, "gripper")
LIMITS = {"arm_joint1": (-160.0, 160.0), "arm_joint2": (3.0, 177.0), "arm_joint3": (-187.0, -3.0),
          "arm_joint4": (-85.0, 85.0), "arm_joint5": (-85.0, 85.0), "arm_joint6": (-160.0, 160.0),
          "gripper": (0.0, 100.0)}
MAX_VEL = {**dict.fromkeys(ARM, 90.0), "gripper": 300.0}
HOME = {"arm_joint1": 0.0, "arm_joint2": 57.3, "arm_joint3": -91.7, "arm_joint4": 34.4, "arm_joint5": 0.0,
        "arm_joint6": 0.0, "gripper": 0.0}
CUBE_XY = (0.0, 0.33)
CUBE_YAW = 0.2
BELIEVED = np.array([CUBE_XY[0], CUBE_XY[1], TOUCH_EXPECTED_Z + TIP_CLEAR])
DT = 0.01
SIM_IMAGE = np.zeros((120, 160, 3), np.uint8)
RAW_IMAGE = np.zeros((RAW_H, RAW_W, 3), np.uint8)

STILL = [0.0, 0.0, 0.0, 0.0, 1.0]
DOWN = [0.0, 0.0, -1.0, 0.0, 1.0]
CLOSE = [0.0, 0.0, 0.0, 0.0, -1.0]
UP_CLOSED = [0.0, 0.0, 1.0, 0.0, -1.0]
YAW = [0.0, 0.0, 0.0, 1.0, 0.2]


# ================================================================== doubles

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    age_s: float


@dataclass(frozen=True)
class ObjectPose:
    position: np.ndarray
    rotation: np.ndarray
    source: str = "symulacja"
    age_s: float = 0.0


@dataclass(frozen=True)
class RunInfo:
    robot: str = "a1x"
    joints: tuple[str, ...] = JOINTS
    gripper: str | None = "gripper"
    limits: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(LIMITS))
    max_vel: Mapping[str, float] = field(default_factory=lambda: dict(MAX_VEL))
    folder: Path = Path(".")
    simulated: bool = True


@dataclass(frozen=True)
class Observation:
    t: float
    step: int
    joints: Mapping[str, float]
    command: Mapping[str, float]
    goal: np.ndarray | None = None
    objects: Mapping[str, ObjectPose] = field(default_factory=dict)
    frames: Mapping[str, Frame] = field(default_factory=dict)


@dataclass(frozen=True)
class Call:
    image: np.ndarray
    obs: np.ndarray
    uv: np.ndarray


class ScriptedPolicy:
    """`act` from a script of the policy tick number; it keeps every input it was given."""

    def __init__(self, script: Callable[[int], list[float]]) -> None:
        self.script = script
        self.calls: list[Call] = []

    def act(self, image: np.ndarray, obs: np.ndarray, uv: np.ndarray) -> np.ndarray:
        self.calls.append(Call(np.array(image), np.array(obs), np.array(uv)))
        return np.asarray(self.script(len(self.calls) - 1), np.float32)

    def inputs(self, image: np.ndarray, uv: np.ndarray) -> np.ndarray:
        return np.zeros((4, 120, 160), np.float32)


def segments(*parts: tuple[int, list[float]]) -> Callable[[int], list[float]]:
    """A script of (ticks, action) segments; the last action repeats."""
    def script(n: int) -> list[float]:
        for ticks, action in parts:
            if n < ticks:
                return action
            n -= ticks
        return parts[-1][1]
    return script


@dataclass
class FakeArm:
    """The supervisor moves the command towards the target at `command_deg_s` (else `MAX_VEL`), the
    servo puts the joint at the command, `blocked` joints never move, `offset` is added to what the
    arm measures."""

    measured: dict[str, float]
    command: dict[str, float] = field(default_factory=dict)
    target: dict[str, float] = field(default_factory=dict)
    command_deg_s: dict[str, float] = field(default_factory=dict)
    blocked: frozenset[str] = frozenset()
    offset: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.command = dict(self.command or self.measured)
        self.target = dict(self.target or self.command)

    def joints(self) -> dict[str, float]:
        return {j: v + self.offset.get(j, 0.0) for j, v in self.measured.items()}

    def follow(self, move: Move, dt: float) -> None:
        self.target.update(move.target)
        for j in JOINTS:
            step = self.command_deg_s.get(j, MAX_VEL[j]) * dt
            self.command[j] += min(max(self.target[j] - self.command[j], -step), step)
            if j not in self.blocked:
                self.measured[j] = self.command[j]


@dataclass(frozen=True)
class Tick:
    obs: Observation
    move: Move
    phase: Phase
    calls: int                      # policy calls so far, this tick's included


def cube(xy: tuple[float, float] = CUBE_XY, yaw: float = CUBE_YAW, z: float = 0.015) -> ObjectPose:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return ObjectPose(position=np.array([xy[0], xy[1], z]), rotation=rotation)


def observe(arm: FakeArm, t: float, step: int, *, frame_age: float = 0.003, image: np.ndarray = SIM_IMAGE,
            objects: Mapping[str, ObjectPose] | None = None, frames: bool = True) -> Observation:
    return Observation(t=t, step=step, joints=arm.joints(), command=dict(arm.command),
                       objects=dict(objects if objects is not None else {"cube": cube()}),
                       frames={"wrist": Frame(image, frame_age)} if frames else {})


def drive(strategy: Grasp, arm: FakeArm, policy: ScriptedPolicy, *, seconds: float = 20.0,
          times: Callable[[int], float] | None = None, each: Callable[[int, float], dict[str, Any]] | None = None,
          **observation: Any) -> list[Tick]:
    """Ticks until `done` or `seconds`; `times(k)` gives each tick's `obs.t` (default k * DT) and
    `each(k, t)` per-tick changes of the observation's keyword arguments."""
    ticks: list[Tick] = []
    k = 0
    t = 0.0
    while t <= seconds:
        t = times(k) if times is not None else k * DT
        extra = dict(observation)
        if each is not None:
            extra.update(each(k, t))
        obs = observe(arm, t, k, **extra)
        move = strategy.act(obs)
        ticks.append(Tick(obs=obs, move=move, phase=strategy.phase, calls=len(policy.calls)))
        arm.follow(move, DT)
        if move.done:
            break
        k += 1
    return ticks


# ================================================================== set-up

@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


@pytest.fixture(scope="module")
def hover(kin: Kinematics) -> Hover:
    """The hover the strategy solves over the default cube: the solve is deterministic."""
    return solve_hover(kin, BELIEVED, CUBE_YAW, joint_box(kin, LIMITS))


def at_hover(hover: Hover, **arm: Any) -> FakeArm:
    joints = {j: math.degrees(float(v)) for j, v in zip(ARM, hover.q)}
    return FakeArm(measured={**joints, "gripper": 100.0}, **arm)


@pytest.fixture
def use_policy(monkeypatch: pytest.MonkeyPatch) -> Callable[[ScriptedPolicy], ScriptedPolicy]:
    def install(policy: ScriptedPolicy) -> ScriptedPolicy:
        monkeypatch.setattr(GraspStudentPolicy, "load", classmethod(lambda cls, path, sha256, *, threads=2: policy))
        return policy
    return install


def settings(**kw: Any) -> GraspSettings:
    return GraspSettings(student_pt=kw.pop("student_pt", "student.pt"), **kw)


def start(policy: ScriptedPolicy, arm: FakeArm, *, run: RunInfo | None = None, image: np.ndarray = SIM_IMAGE,
          objects: Mapping[str, ObjectPose] | None = None, frames: bool = True, **kw: Any) -> Grasp:
    strategy = Grasp(settings(**kw))
    strategy.start(run or RunInfo(), observe(arm, 0.0, 0, image=image, objects=objects, frames=frames))
    policy.calls.clear()                 # the warm-up call of `start`
    return strategy


def refusal(policy: ScriptedPolicy, arm: FakeArm, **kw: Any) -> str:
    with pytest.raises(GraspRefused) as e:
        start(policy, arm, **kw)
    return str(e.value)


@dataclass(frozen=True)
class Ran:
    strategy: Grasp
    ticks: list[Tick]


def run_at_hover(use_policy, hover: Hover, script: Callable[[int], list[float]], *,
                 arm: dict[str, Any] | None = None, setup: dict[str, Any] | None = None,
                 seconds: float = 10.0, **observation: Any) -> Ran:
    """A run from the hover with the jaw open; `arm` goes to `FakeArm`, `setup` to `start`."""
    policy = use_policy(ScriptedPolicy(script))
    fake = at_hover(hover, **(arm or {}))
    strategy = start(policy, fake, **(setup or {}))
    ticks = drive(strategy, fake, policy, seconds=seconds, **observation)
    return Ran(strategy=strategy, ticks=ticks)


def records(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (folder / "ticks.jsonl").read_text(encoding="utf-8").splitlines()]


def policy_ticks(ticks: list[Tick]) -> list[Tick]:
    """The ticks on which the policy ran."""
    out = []
    previous = 0
    for tick in ticks:
        if tick.calls > previous:
            out.append(tick)
        previous = tick.calls
    return out


def complete_rig(path: Path, **change: Any) -> Path:
    """A per-rig file that lets every mode run on a real arm."""
    stamp = datetime.now(timezone.utc).isoformat()
    data = {
        "version": 1,
        "table": {"touch_tcp_z": TOUCH_EXPECTED_Z, "tcp": [CUBE_XY[0], CUBE_XY[1], TOUCH_EXPECTED_Z],
                  "taught_at": stamp},
        "grasp": {"tcp": [CUBE_XY[0], CUBE_XY[1], BELIEVED[2]], "yaw": CUBE_YAW, "object_height": 0.03,
                  "taught_at": stamp},
        "gripper": {"stroke_close_s_operator": 1.3, "freeze_threshold": 100.0},
        "wrist": {"dot_error_px": 3.0},
        "latency": {"camera_s": 0.0, "total_s": 0.1},
        "overrides": {"shadow_done": "10 clean episodes, test"},
    }
    data.update(change)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def sim_rig(path: Path, *, touch: float = TOUCH_EXPECTED_Z, xy: tuple[float, float] = CUBE_XY,
            yaw: float = CUBE_YAW, camera_s: float = 0.0) -> Path:
    path.write_text(json.dumps({
        "version": 1,
        "table": {"touch_tcp_z": touch},
        "grasp": {"tcp": [xy[0], xy[1], touch + TIP_CLEAR], "yaw": yaw, "object_height": 0.03},
        "latency": {"camera_s": camera_s},
    }), encoding="utf-8")
    return path


# ================================================================== the plugin contract

def test_settings_fields_are_text_numbers_or_switches():
    hints = get_type_hints(GraspSettings)
    assert {hints[f.name] for f in dataclasses.fields(GraspSettings)} <= {str, int, float, bool}


def test_settings_have_the_names_the_documentation_uses():
    assert [f.name for f in dataclasses.fields(GraspSettings)] == [
        "student_pt", "student_sha256", "rig_json", "camera", "object_name", "shadow", "dump_dir", "dump_every"]


def test_the_checkpoint_path_has_no_default():
    assert dataclasses.fields(GraspSettings)[0].default is dataclasses.MISSING


def test_the_published_sha256_is_the_default():
    assert settings().student_sha256 == CHECKPOINT_SHA256


def test_importing_the_strategy_imports_nothing_of_the_twin():
    code = ("import sys, galaxeo.grasp.strategy; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot_mp'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_entry_point_names_the_strategy_class():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["entry-points"]["lerobot_mp.strategies"] == {"grasp": "galaxeo.grasp.strategy:Grasp"}


def test_the_strategy_runs_at_100_hz():
    assert Grasp.hz == 100.0


def test_the_settings_class_is_the_strategys():
    assert Grasp.settings is GraspSettings


def test_the_time_limit_covers_the_slowest_staged_plan_and_the_episode():
    # Three stages at 20 deg/s over the A1X's widest ranges alone take over 40 s.
    assert Grasp(settings()).time_limit_s > 40.0 + MAX_TICKS * TICK_S


def test_a_negative_dump_every_is_refused():
    with pytest.raises(ValueError, match="dump_every"):
        Grasp(settings(dump_every=-1))


def test_dumps_without_a_directory_are_refused():
    with pytest.raises(ValueError, match="dump_dir"):
        Grasp(settings(dump_every=5))


# ================================================================== refusals in start

def test_another_robot_is_refused(use_policy, hover):
    so101 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(robot="so101", joints=so101))
    assert "Galaxea A1X" in message


def test_an_arm_without_a_gripper_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(gripper=None, joints=ARM))
    assert "chwytaka" in message


def test_a_supervisor_slower_than_the_policy_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(max_vel={**MAX_VEL, "arm_joint3": 60.0}))
    assert "arm_joint3 60 st./s" in message


def test_a_model_with_other_joint_ranges_is_refused(use_policy, hover, monkeypatch):
    class Wider(Kinematics):
        def __init__(self) -> None:
            super().__init__()
            self.lo = self.lo - math.radians(1.0)

    monkeypatch.setattr(S, "Kinematics", Wider)
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover))
    assert "zakresy stawow" in message


def test_a_real_arm_without_a_rig_file_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(simulated=False), image=RAW_IMAGE)
    assert "rig_json" in message


def test_a_real_arm_with_an_untaught_rig_is_refused_with_the_missing_steps(use_policy, hover, tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text('{"version": 1}', encoding="utf-8")
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(simulated=False), image=RAW_IMAGE, rig_json=str(rig))
    assert "table not taught" in message


def test_shadow_asks_the_rig_for_the_shadow_steps_only(use_policy, hover, tmp_path):
    rig = complete_rig(tmp_path / "rig.json", overrides={})
    strategy = start(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                     run=RunInfo(simulated=False), image=RAW_IMAGE, rig_json=str(rig), shadow=True)
    assert strategy.phase is Phase.PLAN


def test_real_mode_needs_the_shadow_episodes(use_policy, hover, tmp_path):
    rig = complete_rig(tmp_path / "rig.json", overrides={})
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(simulated=False), image=RAW_IMAGE, rig_json=str(rig))
    assert "no shadow episodes" in message


def test_a_malformed_rig_file_is_refused_in_polish(use_policy, hover, tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text("{not json", encoding="utf-8")
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover), rig_json=str(rig))
    assert "nie da sie odczytac" in message


def test_a_missing_checkpoint_is_refused(hover, tmp_path):
    message = refusal(ScriptedPolicy(segments((1, STILL))), at_hover(hover), student_pt=str(tmp_path / "none.pt"))
    assert "nie wczytano modelu studenta" in message


def test_a_checkpoint_with_another_hash_is_refused(hover, tmp_path):
    junk = tmp_path / "student.pt"
    junk.write_bytes(b"not the student")
    message = refusal(ScriptedPolicy(segments((1, STILL))), at_hover(hover), student_pt=str(junk),
                      student_sha256="0" * 64)
    assert "sha256" in message


def test_a_camera_without_a_frame_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover), frames=False)
    assert "brak klatki z kamery 'wrist'" in message


def test_a_real_arm_delivering_the_policy_size_is_refused(use_policy, hover, tmp_path):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(simulated=False), rig_json=str(complete_rig(tmp_path / "rig.json")))
    assert f"{RAW_W}x{RAW_H}" in message


def test_a_real_arm_delivering_the_raw_frame_starts(use_policy, hover, tmp_path):
    strategy = start(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                     run=RunInfo(simulated=False), image=RAW_IMAGE,
                     rig_json=str(complete_rig(tmp_path / "rig.json")))
    assert strategy.phase is Phase.PLAN


def test_a_simulated_arm_delivering_a_raw_frame_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover), image=RAW_IMAGE)
    assert "160x120" in message


def test_without_an_object_or_a_taught_point_the_run_is_refused(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover), objects={})
    assert "nie wiadomo, gdzie chwytac" in message


def test_an_object_in_front_of_the_arm_is_outside_training(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      objects={"cube": cube(xy=(0.33, 0.0))})
    assert "poza obszarem treningu" in message


def test_an_object_without_room_on_j6_has_no_hover(use_policy, hover):
    message = refusal(use_policy(ScriptedPolicy(segments((1, STILL)))), at_hover(hover),
                      run=RunInfo(limits={**LIMITS, "arm_joint6": (-130.0, 70.0)}),
                      objects={"cube": cube(yaw=math.radians(-60.0))})
    assert "brak zawisu" in message


# ================================================================== where the grasp is believed

def first_command(tmp_path: Path, use_policy, arm: FakeArm, **kw: Any) -> list[float]:
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    strategy = start(policy, arm, dump_dir=str(tmp_path / "dump"), **kw)
    drive(strategy, arm, policy, seconds=3.0, objects=kw.get("objects"))
    strategy.stop("test")
    return records(tmp_path / "dump")[0]["p_cmd"]


def test_without_an_observed_object_the_taught_point_is_the_belief(use_policy, hover, tmp_path):
    rig = sim_rig(tmp_path / "rig.json", xy=(0.02, 0.32))
    p = first_command(tmp_path, use_policy, at_hover(hover), objects={}, rig_json=str(rig))
    assert p[:2] == pytest.approx([0.02, 0.32], abs=1e-12)


def test_an_observed_object_wins_over_the_taught_point(use_policy, hover, tmp_path):
    rig = sim_rig(tmp_path / "rig.json", xy=(0.02, 0.32))
    p = first_command(tmp_path, use_policy, at_hover(hover), rig_json=str(rig))
    assert p[:2] == pytest.approx(list(CUBE_XY), abs=1e-12)


# ================================================================== a run from home

@pytest.fixture(scope="module")
def from_home() -> dict[str, Any]:
    """One run from the home pose with a script that goes down, closes and lifts; the observed cube
    stays on the table, so the run ends at the tick cap."""
    policy = ScriptedPolicy(segments((3, STILL), (10, DOWN), (8, CLOSE), (60, UP_CLOSED)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GraspStudentPolicy, "load", classmethod(lambda cls, path, sha256, *, threads=2: policy))
        arm = FakeArm(measured=dict(HOME))
        strategy = start(policy, arm)
        ticks = drive(strategy, arm, policy, seconds=40.0)
        strategy.stop("test")
    return {"ticks": ticks, "strategy": strategy, "policy": policy}


def phase_ticks(ticks: list[Tick], phase: Phase) -> list[Tick]:
    return [t for t in ticks if t.phase is phase]


def test_a_run_from_home_goes_through_the_phases_in_order(from_home):
    seen: list[Phase] = []
    for tick in from_home["ticks"]:
        if not seen or seen[-1] is not tick.phase:
            seen.append(tick.phase)
    assert seen == [Phase.RAMPS, Phase.JAW, Phase.ARRIVE, Phase.SAG, Phase.EPISODE, Phase.DONE]


def test_the_ramps_move_no_joint_faster_than_the_smoothstep_peak_of_the_preposition_speed(from_home):
    targets = [t.move.target for t in phase_ticks(from_home["ticks"], Phase.RAMPS)]
    fastest = max(abs(b[j] - a[j]) for a, b in zip(targets, targets[1:]) for j in ARM) / DT
    assert fastest <= 1.5 * S.PREPOSITION_SPEED_DEG_S + 1e-6


def test_the_jaw_keeps_its_command_during_the_ramps(from_home):
    assert {t.move.target["gripper"] for t in phase_ticks(from_home["ticks"], Phase.RAMPS)} == {HOME["gripper"]}


def test_the_jaw_opens_at_the_hover_with_99_then_100(from_home):
    first_policy_tick = policy_ticks(from_home["ticks"])[0]
    changes: list[float] = []
    for tick in from_home["ticks"]:
        if tick is first_policy_tick:
            break
        grip = tick.move.target.get("gripper")
        if grip is not None and (not changes or changes[-1] != grip):
            changes.append(grip)
    assert changes == [HOME["gripper"], 99.0, 100.0]


def test_the_policy_first_runs_with_the_arm_standing_at_the_hover(from_home, hover):
    first = policy_ticks(from_home["ticks"])[0]
    off = max(abs(first.obs.joints[j] - math.degrees(float(v))) for j, v in zip(ARM, hover.q))
    assert off <= S.ARRIVE_DEG


def test_the_policy_steps_every_50_ms(from_home):
    times = np.array([t.obs.t for t in policy_ticks(from_home["ticks"])])
    np.testing.assert_allclose(np.diff(times), TICK_S, atol=1e-9)


def test_the_ticks_between_policy_ticks_send_no_target(from_home):
    ran = {id(t) for t in policy_ticks(from_home["ticks"])}
    between = [t for t in phase_ticks(from_home["ticks"], Phase.EPISODE) if id(t) not in ran]
    assert all(t.move.target == {} for t in between)


def test_the_run_ends_at_the_tick_cap(from_home):
    assert from_home["strategy"].ending.code == "tick cap"


def test_the_tick_cap_is_not_a_known_success(from_home):
    assert from_home["ticks"][-1].move.success is None


def test_an_ending_holds_the_arm_where_it_measures(from_home):
    last = from_home["ticks"][-1]
    assert {j: last.move.target[j] for j in ARM} == {j: last.obs.joints[j] for j in ARM}


def test_after_the_tick_cap_the_jaw_keeps_its_closed_command(from_home):
    assert from_home["ticks"][-1].move.target["gripper"] == 0.0


def test_the_ending_note_is_polish_for_the_panel(from_home):
    assert from_home["ticks"][-1].move.note == f"chwyt: limit {MAX_TICKS} tikow"


# ================================================================== pre-positioning

@dataclass(frozen=True)
class Retreat:
    start_xy: np.ndarray
    low_targets: list[np.ndarray]          # commanded TCP positions under the hover height


@pytest.fixture
def retreat(use_policy, hover, kin) -> Retreat:
    """The tool 5 cm under the hover, pointing down: the arm leaves it straight up."""
    low = kin.ik_step(hover.p - np.array([0.0, 0.0, 0.05]), kin.fk(hover.q)[:3, :3], hover.q, iters=200)
    arm = FakeArm(measured={**{j: math.degrees(float(v)) for j, v in zip(ARM, low.q)}, "gripper": 100.0})
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    strategy = start(policy, arm)
    ticks = drive(strategy, arm, policy, seconds=10.0)
    commanded = [kin.fk(np.radians([t.move.target[j] for j in ARM]))[:3, 3] for t in phase_ticks(ticks, Phase.RAMPS)]
    return Retreat(start_xy=kin.fk(low.q)[:2, 3], low_targets=[p for p in commanded if p[2] < hover.p[2] - 1e-3])


def test_a_tool_low_near_the_hover_is_commanded_up_from_where_it_stands(retreat):
    assert len(retreat.low_targets) > 10


def test_the_way_up_stays_on_the_vertical_through_the_tool(retreat):
    assert max(float(np.hypot(*(p[:2] - retreat.start_xy))) for p in retreat.low_targets) <= 1e-3


@pytest.fixture
def blocked_base(use_policy) -> Ran:
    """J1 does not move while the staged plan drives it."""
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    arm = FakeArm(measured=dict(HOME), blocked=frozenset({"arm_joint1"}))
    strategy = start(policy, arm)
    return Ran(strategy=strategy, ticks=drive(strategy, arm, policy, seconds=40.0))


def test_a_blocked_joint_during_the_ramps_ends_impact_lag(blocked_base):
    assert blocked_base.strategy.ending.code == "impact lag arm_joint1"


def test_an_impact_holds_the_measured_pose_jaw_included(blocked_base):
    last = blocked_base.ticks[-1]
    assert last.move.target == dict(last.obs.joints)


@pytest.fixture
def blocked_way(use_policy, monkeypatch) -> Ran:
    monkeypatch.setattr(CollisionChecker, "path_clear", lambda self, *a, **kw: "kolizja: test")
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    arm = FakeArm(measured=dict(HOME))
    strategy = start(policy, arm)
    return Ran(strategy=strategy, ticks=drive(strategy, arm, policy, seconds=1.0))


def test_a_collision_on_the_way_ends_the_run_on_its_first_tick(blocked_way):
    assert len(blocked_way.ticks) == 1


def test_a_collision_on_the_way_is_the_ending(blocked_way):
    assert blocked_way.strategy.ending.code == "collision"


def test_a_collision_on_the_way_sends_no_target_but_the_measured_pose(blocked_way):
    first = blocked_way.ticks[0]
    assert first.move.target == dict(first.obs.joints)


def test_an_arm_that_stops_short_of_the_hover_ends_not_arrived(use_policy, hover):
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    arm = at_hover(hover, offset={"arm_joint1": 3.0})
    strategy = start(policy, arm)
    drive(strategy, arm, policy, seconds=10.0)
    assert strategy.ending.code == "not arrived"


def test_on_a_real_arm_j5_off_its_zero_at_the_hover_ends_the_run(use_policy, hover, tmp_path):
    policy = use_policy(ScriptedPolicy(segments((1, STILL))))
    arm = at_hover(hover, offset={"arm_joint5": 1.1})
    strategy = start(policy, arm, run=RunInfo(simulated=False), image=RAW_IMAGE,
                     rig_json=str(complete_rig(tmp_path / "rig.json")))
    drive(strategy, arm, policy, seconds=10.0, image=RAW_IMAGE)
    assert strategy.ending.code == "j5 offset"


# ================================================================== pairing and the schedule

def test_the_history_pairs_with_the_sample_nearest_the_time():
    history = JointHistory()
    for k in range(10):
        history.add(k * 0.01, {"arm_joint1": float(k)})
    assert history.nearest(0.042).joints["arm_joint1"] == 4.0


def test_the_history_keeps_two_seconds():
    history = JointHistory()
    for k in range(300):
        history.add(k / 64, {"arm_joint1": float(k)})        # times exact in binary
    assert history.nearest(0.0).t == (299 - 128) / 64


def test_the_vector_is_built_from_the_joints_at_the_frames_exposure(use_policy, hover):
    policy = use_policy(ScriptedPolicy(segments((40, DOWN))))
    arm = at_hover(hover)
    strategy = start(policy, arm)
    ticks = drive(strategy, arm, policy, seconds=3.0, frame_age=0.03)
    by_time = {round(t.obs.t, 6): t.obs.joints for t in ticks}
    lo = np.asarray(S.JOINT_LO)
    hi = np.asarray(S.JOINT_HI)
    errors = []
    for tick, call in zip(policy_ticks(ticks), policy.calls):
        joints = by_time[round(tick.obs.t - 0.03, 6)]
        want = 2.0 * (np.radians([joints[j] for j in ARM]) - lo) / (hi - lo) - 1.0
        errors.append(float(np.max(np.abs(call.obs[:6] - want))))
    assert max(errors) <= 1e-6


def test_a_frame_with_no_joint_sample_near_its_exposure_is_skipped(use_policy, hover, tmp_path):
    policy = use_policy(ScriptedPolicy(segments((40, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm, rig_json=str(sim_rig(tmp_path / "rig.json", camera_s=0.1)),
                     dump_dir=str(tmp_path / "dump"))

    # The strategy is not called for 0.3 s, then sees a frame exposed 0.1 s ago: no sample near it.
    def times(k: int) -> float:
        return k * DT if k < 200 else k * DT + 0.3

    drive(strategy, arm, policy, seconds=2.6, times=times)
    strategy.stop("test")
    assert "pairing" in [r["skip"] for r in records(tmp_path / "dump")]


def test_the_policy_steps_on_a_fixed_grid_whatever_the_tick_period(use_policy, hover):
    policy = use_policy(ScriptedPolicy(segments((200, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm)
    period = 0.0103
    ticks = drive(strategy, arm, policy, seconds=4.0, times=lambda k: k * period)
    times = [t.obs.t for t in policy_ticks(ticks)]
    off_grid = [t for n, t in enumerate(times) if not times[0] + n * TICK_S <= t < times[0] + n * TICK_S + period]
    assert off_grid == []


def test_a_late_tick_restarts_the_grid_after_it(use_policy, hover):
    policy = use_policy(ScriptedPolicy(segments((200, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm)

    def times(k: int) -> float:
        return k * DT if k < 150 else k * DT + 0.12

    ticks = drive(strategy, arm, policy, seconds=3.0, times=times)
    late = [t.obs.t for t in policy_ticks(ticks) if t.obs.t > 150 * DT]
    np.testing.assert_allclose(np.diff(late), TICK_S, atol=1e-9)


def test_a_policy_tick_more_than_a_period_late_is_counted(use_policy, hover):
    policy = use_policy(ScriptedPolicy(segments((200, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm)

    def times(k: int) -> float:
        return k * DT if k < 150 else k * DT + 0.12

    drive(strategy, arm, policy, seconds=3.0, times=times)
    strategy.stop("test")
    assert strategy.summary["loop_gaps"] == 1


# ================================================================== soft skips

@pytest.fixture
def stale_frames(use_policy, hover) -> Ran:
    return run_at_hover(use_policy, hover, segments((40, STILL)), seconds=5.0, frame_age=0.2)


def test_stale_frames_end_the_episode_with_camera(stale_frames):
    assert stale_frames.strategy.ending.code == "camera"


def test_stale_frames_are_skipped_five_times_first(stale_frames):
    assert stale_frames.strategy.summary["skipped"] == 5


def test_a_skipped_tick_resends_the_previous_target(stale_frames):
    episode = [t.move.target for t in phase_ticks(stale_frames.ticks, Phase.EPISODE) if t.move.target]
    assert all(target == episode[0] for target in episode)


@pytest.fixture
def old_frame(use_policy, hover) -> Ran:
    return run_at_hover(use_policy, hover, segments((40, STILL)), seconds=5.0, frame_age=0.6)


def test_a_frame_older_than_half_a_second_ends_with_camera(old_frame):
    assert old_frame.strategy.ending.code == "camera"


def test_a_frame_older_than_half_a_second_ends_without_a_skip(old_frame):
    assert old_frame.strategy.summary["skipped"] == 0


@pytest.fixture
def silent_camera(use_policy, hover) -> Ran:
    policy = use_policy(ScriptedPolicy(segments((40, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm)
    ticks = drive(strategy, arm, policy, seconds=5.0,
                  each=lambda k, t: {"frames": strategy.phase is not Phase.EPISODE})
    return Ran(strategy=strategy, ticks=ticks)


def test_a_camera_that_stops_delivering_ends_with_camera(silent_camera):
    assert silent_camera.strategy.ending.code == "camera"


def test_a_camera_that_stops_delivering_is_skipped_five_times_first(silent_camera):
    assert silent_camera.strategy.summary["skipped"] == 5


@pytest.fixture
def slow_supervisor(use_policy, hover) -> Ran:
    """The supervisor drives J6 at 2 deg/s while the policy asks for 5 degrees a tick."""
    return run_at_hover(use_policy, hover, segments((40, YAW)), arm={"command_deg_s": {"arm_joint6": 2.0}},
                        seconds=5.0)


def test_a_supervisor_far_behind_the_command_ends_with_lag(slow_supervisor):
    assert slow_supervisor.strategy.ending.code == "lag"


def test_lag_ends_the_episode_after_ten_skips(slow_supervisor):
    assert slow_supervisor.strategy.summary["skipped"] == 10


def test_a_target_the_supervisor_has_not_reached_by_the_next_tick_counts_as_rate_limited(slow_supervisor):
    assert slow_supervisor.strategy.summary["rate_limited"] >= 1


# ================================================================== the bookkeeping

@pytest.fixture
def refused_yaw(use_policy, hover, tmp_path) -> dict[str, Any]:
    """J6 has 15.5 degrees of room below the hover: three 5 degree yaw steps fit, the fourth is refused."""
    j6 = math.degrees(float(hover.q[5]))
    run = RunInfo(limits={**LIMITS, "arm_joint6": (j6 - 15.5, 160.0)})
    policy = use_policy(ScriptedPolicy(segments((6, YAW))))
    arm = at_hover(hover)
    strategy = start(policy, arm, run=run, dump_dir=str(tmp_path / "dump"))
    ticks = drive(strategy, arm, policy, seconds=1.6)
    strategy.stop("test")
    return {"records": records(tmp_path / "dump"), "ticks": policy_ticks(ticks), "policy": policy}


def test_the_fourth_yaw_step_is_refused_by_the_ik(refused_yaw):
    assert [r["ik_ok"] for r in refused_yaw["records"][:4]] == [True, True, True, False]


def test_a_refused_step_keeps_the_commanded_yaw(refused_yaw):
    rec = refused_yaw["records"]
    assert rec[3]["yaw_cmd"] == rec[2]["yaw_cmd"]


def test_a_refused_step_keeps_the_commanded_tcp(refused_yaw):
    rec = refused_yaw["records"]
    assert rec[3]["p_cmd"] == rec[2]["p_cmd"]


def test_a_refused_step_commits_the_jaw_command(refused_yaw):
    assert refused_yaw["records"][3]["g_cmd"] == pytest.approx(0.6)


def test_the_tick_after_a_refusal_sees_the_refused_action_as_the_previous_one(refused_yaw):
    assert refused_yaw["policy"].calls[4].obs[13:18].tolist() == pytest.approx(YAW)


def test_a_refused_step_resends_the_previous_joints(refused_yaw):
    targets = [t.move.target for t in refused_yaw["ticks"]]
    assert {j: targets[3][j] for j in ARM} == {j: targets[2][j] for j in ARM}


def test_the_command_integrates_from_itself_not_from_the_measured_pose(use_policy, hover, tmp_path):
    policy = use_policy(ScriptedPolicy(segments((4, DOWN), (10, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm, dump_dir=str(tmp_path / "dump"))

    def droop(k: int, t: float) -> dict[str, Any]:
        # The shoulder sags 1.5 degrees under the command from the first policy tick on.
        if policy.calls:
            arm.offset["arm_joint2"] = 1.5
        return {}

    drive(strategy, arm, policy, seconds=1.4, each=droop)
    strategy.stop("test")
    want = GraspCommand(hover.p, hover.yaw, hover.q, floor_z=0.0)
    expected = [want.state.p_cmd]
    for _ in range(4):
        want.commit(want.propose(np.asarray(DOWN, np.float32)), want.state.q_cmd)
        expected.append(want.state.p_cmd)
    got = [r["p_cmd"] for r in records(tmp_path / "dump")[:4]]
    assert got == [p.tolist() for p in expected[1:]]


# ================================================================== the measured floor and the table

@pytest.fixture
def at_the_table(use_policy, hover) -> Ran:
    policy = use_policy(ScriptedPolicy(segments((40, DOWN))))
    arm = at_hover(hover)
    strategy = start(policy, arm)

    def droop(k: int, t: float) -> dict[str, Any]:
        # 2.5 degrees on J2 put the measured TCP about 14 mm under the command.
        if policy.calls:
            arm.offset["arm_joint2"] = 2.5
        return {}

    return Ran(strategy=strategy, ticks=drive(strategy, arm, policy, seconds=5.0, each=droop))


def test_a_measured_tool_at_the_table_ends_table(at_the_table):
    assert at_the_table.strategy.ending.code == "table"


def test_the_table_ending_also_stops_the_jaw_where_it_measures(at_the_table):
    last = at_the_table.ticks[-1]
    assert last.move.target == dict(last.obs.joints)


# ================================================================== shadow

@pytest.fixture
def shadow_run(use_policy, hover, tmp_path) -> dict[str, Any]:
    policy = use_policy(ScriptedPolicy(segments((10, DOWN), (8, CLOSE), (60, UP_CLOSED))))
    arm = at_hover(hover)
    strategy = start(policy, arm, shadow=True, dump_dir=str(tmp_path / "dump"))
    risen = cube(z=0.015 + 0.06)

    def lifted_cube(k: int, t: float) -> dict[str, Any]:
        return {"objects": {"cube": risen if len(policy.calls) > 20 else cube()}}

    ticks = drive(strategy, arm, policy, seconds=10.0, each=lifted_cube)
    strategy.stop("test")
    return {"ticks": ticks, "strategy": strategy, "records": records(tmp_path / "dump")}


def test_shadow_keeps_the_jaw_open_in_every_episode_target(shadow_run):
    grips = {t.move.target["gripper"] for t in phase_ticks(shadow_run["ticks"], Phase.EPISODE) if t.move.target}
    assert grips == {100.0}


def test_shadow_keeps_the_command_floor_above_the_believed_point(shadow_run):
    floor = min(r["floor_z"] for r in shadow_run["records"])
    assert floor >= BELIEVED[2] + SHADOW_FLOOR_ABOVE_BELIEVED


def test_shadow_never_reports_a_success(shadow_run):
    assert shadow_run["strategy"].ending.success is None


# ================================================================== success

GRASP_AND_LIFT = segments((10, DOWN), (8, CLOSE), (60, UP_CLOSED))


@pytest.fixture
def lifted_object(use_policy, hover) -> Ran:
    policy = use_policy(ScriptedPolicy(GRASP_AND_LIFT))
    arm = at_hover(hover)
    strategy = start(policy, arm)
    risen = cube(z=0.015 + 0.05)
    ticks = drive(strategy, arm, policy, seconds=10.0,
                  each=lambda k, t: {"objects": {"cube": risen if len(policy.calls) > 20 else cube()}})
    return Ran(strategy=strategy, ticks=ticks)


def test_an_observed_object_risen_with_the_jaw_closed_ends_with_success(lifted_object):
    assert lifted_object.strategy.ending.code == "success"


def test_the_success_is_reported_to_the_twin(lifted_object):
    assert lifted_object.ticks[-1].move.success is True


def test_a_success_holds_the_jaw_closed(lifted_object):
    assert lifted_object.ticks[-1].move.target["gripper"] == 0.0


@pytest.fixture
def lifted_blind(use_policy, hover, tmp_path) -> Ran:
    """The taught point and no observed object: the strategy cannot see what it holds."""
    return run_at_hover(use_policy, hover, GRASP_AND_LIFT, objects={},
                        setup={"objects": {}, "rig_json": str(sim_rig(tmp_path / "rig.json"))})


def test_without_an_observed_object_a_closed_jaw_lifted_4_cm_ends_lifted(lifted_blind):
    assert lifted_blind.strategy.ending.code == "lifted"


def test_a_lift_without_an_observed_object_is_not_a_known_success(lifted_blind):
    assert lifted_blind.ticks[-1].move.success is None


# ================================================================== dumps

@pytest.fixture
def dumped(use_policy, hover, tmp_path) -> Path:
    """A few policy ticks, then the twin stops the run from outside, as its STOP does."""
    policy = use_policy(ScriptedPolicy(segments((5, STILL))))
    arm = at_hover(hover)
    strategy = start(policy, arm, dump_dir=str(tmp_path / "dump"), dump_every=2)
    drive(strategy, arm, policy, seconds=1.5)
    strategy.stop("STOP")
    return tmp_path / "dump"


def test_model_input_dumps_are_written_every_n_policy_ticks(dumped):
    assert sorted(p.name for p in dumped.glob("[0-9][0-9][0-9][0-9].png"))[:3] == ["0000.png", "0002.png", "0004.png"]


def test_each_dump_has_its_heat_channel(dumped):
    images = {p.stem for p in dumped.glob("[0-9][0-9][0-9][0-9].png")}
    assert {p.stem.removesuffix("_heat") for p in dumped.glob("*_heat.png")} == images


def test_every_policy_tick_is_a_line_of_the_tick_log(dumped):
    assert len(records(dumped)) == json.loads((dumped / "summary.json").read_text())["ticks"]


def test_the_summary_names_the_ending(dumped):
    assert json.loads((dumped / "summary.json").read_text())["ending"] == "stopped"


# ================================================================== training's recording, tick by tick

@dataclass(frozen=True)
class Replayed:
    calls: list[Call]
    recorded: dict[str, np.ndarray]           # this episode's rows of shard.npz


class RecordedPolicy(ScriptedPolicy):
    """The recorded action of each policy tick."""

    def __init__(self, actions: np.ndarray) -> None:
        super().__init__(lambda n: actions[n])


SHARD_KEYS = ("img", "q", "gap", "act", "obs", "obs_fk", "uv", "uv_fk", "believed_pos", "believed_yaw", "episode",
              "g_cmd", "prev_action")
JAW_ENTRY = 12
ATTEMPTS_ENTRY = 18


def replay(episode: int, tmp_path: Path) -> Replayed:
    """Training's ticks of one episode through the strategy's own tick: the recorded joints and jaw
    as the measurement, the recorded frames as the wrist camera, the object where training believed it."""
    z = npz("shard.npz", *SHARD_KEYS)
    rows = z["episode"] == episode
    rec = {k: z[k][rows] for k in SHARD_KEYS}
    believed = rec["believed_pos"][0] - np.asarray(ARM_BASE_WORLD)
    touch = float(believed[2]) - TIP_CLEAR
    yaw = float(rec["believed_yaw"][0])
    seen = {"cube": cube(xy=(float(believed[0]), float(believed[1])), yaw=yaw)}

    def recorded_joints(i: int) -> dict[str, float]:
        joints = {j: math.degrees(float(v)) for j, v in zip(ARM, rec["q"][i])}
        return {**joints, "gripper": (float(rec["gap"][i]) - GAP_CLOSED) / GAP_PER_UNIT}

    policy = RecordedPolicy(rec["act"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GraspStudentPolicy, "load", classmethod(lambda cls, path, sha256, *, threads=2: policy))
        arm = FakeArm(measured=recorded_joints(0))
        strategy = start(policy, arm, objects=seen, rig_json=str(sim_rig(tmp_path / "rig.json", touch=touch)))

        k = 0
        replaying = False
        while len(policy.calls) < len(rec["act"]) and k < 3000:
            i = len(policy.calls)
            # From the sag on, the arm is training's: its measured joints, its frames.
            replaying = replaying or strategy.phase in (Phase.SAG, Phase.EPISODE)
            if replaying:
                arm = FakeArm(measured=recorded_joints(i))
            obs = observe(arm, k * DT, k, frame_age=0.0, image=rec["img"][min(i, len(rec["img"]) - 1)], objects=seen)
            move = strategy.act(obs)
            assert not move.done, move.note
            arm.follow(move, DT)
            k += 1
        strategy.stop("test")
    return Replayed(calls=policy.calls, recorded=rec)


@pytest.fixture(scope="module", params=[50000, 50001], ids=["episode 50000", "episode 50001"])
def replayed(request, tmp_path_factory) -> Replayed:
    return replay(request.param, tmp_path_factory.mktemp("replay"))


def forced_close(rec: dict[str, np.ndarray]) -> np.ndarray:
    """Ticks where training's jaw command is not the previous action's: its forced early close
    (`force_p`), which slams the jaw shut for 6 ticks whatever the policy says. A deployed strategy
    has no such perturbation. The first tick has no previous action."""
    own = np.clip(0.5 * (rec["prev_action"][:, 4].astype(np.float64) + 1.0), 0.0, 1.0)
    forced = np.abs(rec["g_cmd"] - own) > 1e-6
    forced[0] = False
    return forced


def test_every_recorded_tick_reaches_the_policy(replayed):
    assert len(replayed.calls) == len(replayed.recorded["act"])


def test_training_recorded_the_episode_with_a_forced_early_close(replayed):
    assert int(forced_close(replayed.recorded).sum()) == 6


def test_the_recorded_frame_reaches_the_policy_unchanged(replayed):
    assert all(np.array_equal(c.image, img) for c, img in zip(replayed.calls, replayed.recorded["img"]))


def test_outside_the_forced_close_the_strategys_vector_is_trainings_at_the_recorded_joints(replayed):
    free = ~forced_close(replayed.recorded)
    got = np.stack([c.obs for c in replayed.calls])[free]
    assert np.abs(got - replayed.recorded["obs_fk"][free]).max() <= 1e-5


def test_during_the_forced_close_only_the_jaw_command_and_the_attempts_differ(replayed):
    forced = forced_close(replayed.recorded)
    others = np.r_[0:JAW_ENTRY, JAW_ENTRY + 1:ATTEMPTS_ENTRY]
    got = np.stack([c.obs for c in replayed.calls])[forced]
    assert np.abs(got[:, others] - replayed.recorded["obs_fk"][forced][:, others]).max() <= 1e-5


def test_the_strategys_believed_pixel_is_trainings_at_the_recorded_joints(replayed):
    got = np.stack([c.uv for c in replayed.calls])
    assert np.abs(got - replayed.recorded["uv_fk"]).max() <= 1e-3


@pytest.fixture(scope="module")
def network() -> GraspStudentPolicy:
    value = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not value:
        pytest.skip(f"no checkpoint: {CHECKPOINT_ENV}=<path to the published student.pt>")
    return GraspStudentPolicy.load(Path(value).expanduser(), CHECKPOINT_SHA256)


def test_with_trainings_lagging_tcp_and_pixel_the_policy_gives_the_recorded_actions_exactly(replayed, network):
    got = np.stack([network.act(c.image, obs, uv) for c, obs, uv in
                    zip(replayed.calls, replayed.recorded["obs"], replayed.recorded["uv"])])
    assert np.array_equal(got, replayed.recorded["act"])


def test_the_policy_on_the_strategys_own_inputs_stays_within_the_substep_lag_of_the_recorded_actions(replayed,
                                                                                                    network):
    forced = forced_close(replayed.recorded)
    got = []
    for c, obs, was_forced in zip(replayed.calls, replayed.recorded["obs"], forced):
        vector = c.obs.copy()
        if was_forced:
            vector[[JAW_ENTRY, ATTEMPTS_ENTRY]] = obs[[JAW_ENTRY, ATTEMPTS_ENTRY]]
        got.append(network.act(c.image, vector, c.uv))
    assert np.abs(np.stack(got) - replayed.recorded["act"]).max() <= 0.01
