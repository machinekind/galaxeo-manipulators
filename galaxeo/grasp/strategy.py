"""The wrist-camera grasp student as a strategy of the digital twin.

    [project.entry-points."lerobot_mp.strategies"]
    grasp = "galaxeo.grasp.strategy:Grasp"

The twin builds `GraspSettings` from the operator's values, creates `Grasp(settings)`, calls
`start(run, obs)` once (a refusal raises `GraspRefused` with a Polish message), then `act(obs)`
at `hz` until an answer says `done`, and `stop(reason)` once. Nothing here imports the twin: the
answers are this module's `Move`, the observation and the run are read by their attributes.

One run is one episode, in phases inside `act`:

  plan      on the first tick: the straight-up retreat (when the tool stands low near the
            hover) and the staged plan to the hover, each stage's path collision-checked once
  ramps     smoothstep ramps through those poses at PREPOSITION_SPEED_DEG_S, the jaw as it was
  jaw       open at the hover (99, then 100), until the measured jaw is open
  arrive    until the arm stands still at the hover
  sag       the hover's sag, averaged over SAG_AVERAGE_S
  episode   the policy on a 50 ms schedule by `obs.t`; the ticks between send nothing
  done      the arm holds where it measures

The episode tick is training's step: the 19-vector from the newest joints and the believed pixel
from the joints paired with the frame, as in PR 25, the policy, one bookkeeping step, one warm IK
step accepted only when it
converged, jumped at most IK_MAX_JUMP_DEG and is clear of the table and the object. A refused
step keeps the commanded TCP, yaw and joints and commits the rest; the command never
integrates from the measured pose. The supervisor rate-limits each target to `max_vel`.

The jaw gap the policy reads is the measured gripper, `GAP_CLOSED + GAP_PER_UNIT * opening`: the
A1X measures its jaw, the simulated arm reports the scene's finger slide. In shadow the real jaw
stays open, the floor stays above the object and nothing is a success; the gap there comes from
a virtual jaw that follows the policy's own command (`_VirtualJaw`) at the jaw speed measured on
the rig (`Rig.jaw_units_s`), else at the contract's `STREAM_UNITS_S`.

Every ending answers `done=True`. An ending that may hold the object (success, lifted, tick or
time cap) keeps the last episode target, jaw included; any other holds where the arm measures.

Units: the observation and the targets in degrees and the gripper 0..100, `galaxeo.arm` and the
19-vector in radians.

The A1X backend's joint offsets (`a1x.joint_offsets_deg`) must be zero: the wrist camera was
calibrated on raw joint angles, and a strategy cannot read the offsets.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..arm import CollisionChecker, IKStep, Kinematics
from ..arm.guard import ImpactGuard
from ..arm.model import JOINTS, Box
from .camera import HAND_BODY, CameraError, WristCamera, wrist_intrinsics
from .command import GraspCommand
from .contract import (
    ARM_SPEED_DEG_S,
    ARRIVE_DEG,
    ARRIVE_M,
    ARRIVE_STILL_DEG_S,
    ARRIVE_TIMEOUT_S,
    CEILING,
    CHECKPOINT_SHA256,
    CLOSE_BELOW,
    FLOOR_MARGIN,
    FRAME_AGE_MAX_S,
    GAP_CLOSED,
    GAP_PER_UNIT,
    HOVER_J5_MEASURED_MAX_DEG,
    IK_DAMPING,
    IK_ITERS,
    IK_MAX_JUMP_DEG,
    IK_TOL_POS,
    IK_TOL_ROT,
    IMG_H,
    IMG_W,
    JOINT_HI,
    JOINT_LO,
    LAG_HOLD_DEG,
    LIFT,
    MAX_TICKS,
    OBJECT_HEIGHT_DEFAULT,
    PAIR_GAP_MAX_S,
    PREPOSITION_SPEED_DEG_S,
    RAW_H,
    RAW_W,
    RETREAT_BELOW,
    RETREAT_CLEAR,
    RETREAT_NEAR_XY,
    RETREAT_STEP,
    RETREAT_TILT_MAX_DEG,
    SAG_FILTER_S,
    SAG_LIMIT,
    SAG_MAX,
    SHADOW_CLEAR,
    SHADOW_FLOOR_ABOVE_BELIEVED,
    STREAM_UNITS_S,
    TABLE_TOUCH_FAR,
    TABLE_END_BELOW,
    TABLE_END_TICKS,
    TICK_S,
    TIP_AHEAD,
    TIP_CLEAR,
    TRAIN_HOVER_Z,
)
from .dumps import Dumps, TickRecord
from .hover import STAGES, Hover, HoverError, JointBox, joint_box, solve_hover, stage_plan
from .observer import GraspObserver, range_mismatch
from .policy import CheckpointError, GraspStudentPolicy
from .rig import Rig, RigError

if TYPE_CHECKING:
    from lerobot_mp.twin.strategy import Frame, Observation, RunInfo, Strategy

logger = logging.getLogger(__name__)

ARM = JOINTS
GRIPPER = "gripper"
JOINTS_AND_GRIPPER = (*ARM, GRIPPER)

#: Loading the checkpoint sets torch's intra-op threads, which is process-wide: inside the twin
#: every other torch user gets the same 2 threads.
TORCH_THREADS = 2

#: Longer than any frame age that does not end the episode at once (`FRAME_END_S`).
HISTORY_S = 2.0
#: The policy's grid is a sum of 0.05 s steps: a tick on a grid point must not round to just before it.
SCHEDULE_TOL_S = 1e-6
#: The twin measures at 50 Hz, so two 100 Hz ticks in a row can carry the same joints.
SPEED_WINDOW_S = 0.05
FRAME_END_S = 0.5
#: Soft skips in a row that end an episode; lag gets twice as long, as in PR 25: an arm behind its
#: command catches up by itself, a camera or pairing gap does not.
SKIPS_END = 5
SKIPS_END_LAG = 10
RATE_LIMITED_DEG = 0.5
#: Rate-limited policy ticks in a row that end an episode, as long as the lag streak (0.5 s). At
#: 90 deg/s a target (at most IK_MAX_JUMP_DEG away) is reached within 67 ms; only a yaw at the
#: policy's full 5 degrees a tick runs 10 deg/s over the limit, and the largest misalignment
#: training drew (45 deg) takes 9 such ticks. Ten in a row is a speed limit below the policy's.
RATE_LIMITED_END = 10
#: Next to the tick cap, which counts skipped ticks: 120 ticks take 6 s on time, and a run whose
#: ticks keep coming late still ends.
MAX_SECONDS = 12.0
# The measured-side floor starts at the command floor's own margin: at full lateral speed the
# tool dips for 3 to 4 ticks while the raised floor brings the command back.
MEASURED_FLOOR_START = FLOOR_MARGIN
#: The jaw opens with 99 first: the A1X backend ignores a first command within 0.5 of the
#: reported opening, and a jaw reported open would never get its stream.
GRIP_STAGE = 99.0
GRIP_OPEN = 100.0
GRIP_FIRST_S = 0.1
GRIP_WAIT_S = 2.0
SAG_AVERAGE_S = 0.3
RAMP_MIN_S = 0.5
#: PR 25's lag rule: tighter in the episode, whose steps are small, than on the ramps.
GUARD_LAG_S = 0.15
PREPOSITION_LAG_DEG = 4.0
EPISODE_LAG_DEG = 3.0
TIME_MARGIN_S = 5.0
#: On a real arm, an observed object older than this is not believed. The twin's cube perception
#: reports at camera rate, well under 0.1 s; a pose a second old is the last sighting of a cube the
#: cameras lost, and since then a hand or the previous episode may have moved it further than the
#: 1.5 cm hover noise the policy was trained to absorb.
OBJECT_AGE_MAX_S = 1.0
#: Endings after which the object may be in the jaw: the jaw keeps its command.
KEEP_GRIP = ("success", "lifted", "tick cap", "time cap")


class GraspRefused(RuntimeError):
    """`start` refuses the run; the message is Polish, for the operator."""


@dataclass
class GraspSettings:
    #: The published checkpoint, student.pt of hvsr-robotics/a1x-wrist-grasp-sim.
    student_pt: str
    #: The sha256 the checkpoint must have; another checkpoint needs its own.
    student_sha256: str = CHECKPOINT_SHA256
    #: The per-rig file (galaxeo.grasp.rig); empty = `Rig.for_sim()`, allowed only on a simulated arm.
    rig_json: str = ""
    #: The wrist camera's name in the twin.
    camera: str = "wrist"
    #: The object in `Observation.objects` whose pose is the believed grasp.
    object_name: str = "cube"
    #: Shadow: the jaw stays open and the floor stays above the object, never a success.
    shadow: bool = False
    #: Where ticks.jsonl, summary.json and the model-input dumps go; empty = nowhere.
    dump_dir: str = ""
    #: A model-input dump every N policy ticks; 0 = none.
    dump_every: int = 0


@dataclass
class Move:
    """The twin's `Action`, by its attributes."""

    target: dict[str, float] = field(default_factory=dict)
    done: bool = False
    success: bool | None = None
    error: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Ending:
    code: str                          # English code for logs and ticks.jsonl
    note: str                          # Polish, for the panel
    success: bool | None = None


class _Ended(Exception):
    def __init__(self, ending: Ending) -> None:
        super().__init__(ending.code)
        self.ending = ending


class _Skip(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Phase(Enum):
    PLAN = "plan"
    RAMPS = "ramps"
    JAW = "jaw"
    ARRIVE = "arrive"
    SAG = "sag"
    EPISODE = "episode"
    DONE = "done"


@dataclass(frozen=True)
class JointSample:
    t: float                           # obs.t [s]
    joints: dict[str, float]           # measured [deg], gripper 0..100


class JointHistory:
    """The measured joints of the last `keep_s` seconds, stamped with the observation's time."""

    def __init__(self, keep_s: float = HISTORY_S) -> None:
        self.keep_s = float(keep_s)
        self._samples: deque[JointSample] = deque()

    def add(self, t: float, joints: Mapping[str, float]) -> None:
        if self._samples and t <= self._samples[-1].t:
            return
        self._samples.append(JointSample(float(t), {k: float(v) for k, v in joints.items()}))
        while self._samples[0].t < t - self.keep_s:
            self._samples.popleft()

    def newest(self) -> JointSample | None:
        return self._samples[-1] if self._samples else None

    def nearest(self, t: float) -> JointSample | None:
        if not self._samples:
            return None
        return min(self._samples, key=lambda s: abs(s.t - t))

    def speed_deg_s(self) -> dict[str, float]:
        """Per joint [deg/s] between the newest sample and the newest one `SPEED_WINDOW_S` older;
        empty while the history is shorter."""
        if not self._samples:
            return {}
        b = self._samples[-1]
        older = [s for s in self._samples if s.t <= b.t - SPEED_WINDOW_S]
        if not older:
            return {}
        a = older[-1]
        dt = b.t - a.t
        return {k: (v - a.joints[k]) / dt for k, v in b.joints.items() if k in a.joints}


@dataclass
class _Ramp:
    """Smoothstep from `start` to `goal` [rad] over `duration`, from the first tick that asks.

    It advances by the time between ticks, at most two periods a tick: a runner that stalls for a
    second must not find the ramp a second further along and jump there.
    """

    start: np.ndarray
    goal: np.ndarray
    grip: float
    duration: float
    elapsed: float = 0.0
    last_t: float | None = None


@dataclass(frozen=True)
class _RampPoint:
    q: np.ndarray
    finished: bool


@dataclass(frozen=True)
class _Acceptance:
    q: np.ndarray | None               # the accepted joints [rad]; None = refused
    step: IKStep
    why: str                           # "" | "ik" | "jump" | "collision"


@dataclass(frozen=True)
class _Believed:
    point: np.ndarray                  # (3,) base frame: grasp xy, z = touch + TIP_CLEAR
    yaw: float
    source: str                        # Polish: where the belief comes from
    object_z: float | None             # the observed object's centre height at the start; None = not observed


@dataclass
class _VirtualJaw:
    """Shadow only: the opening a jaw would have under the policy's command, slewed at `rate`.

    The real jaw is pinned open in shadow, so its measured gap would tell the policy that its close
    did nothing: a gap training never showed after a close.
    """

    opening: float                     # 0..100
    t: float                           # obs.t of the last update [s]
    rate: float                        # units/s

    def follow(self, t: float, commanded: float) -> float:
        """The opening at `t`; the time since the last update counts whole, skipped ticks included."""
        step = self.rate * max(t - self.t, 0.0)
        self.t = t
        self.opening += min(max(_units(commanded) - self.opening, -step), step)
        return self.opening


@dataclass
class _Episode:
    """The state of the episode between policy ticks."""

    cmd: GraspCommand
    sag_used: float
    sag_est: float
    t0: float
    floor_z0: float
    next_policy_t: float
    target: dict[str, float]           # the last target sent
    policy_ticks: int = 0
    skipped: int = 0
    streak: int = 0
    table_ticks: int = 0
    ik_refused: int = 0
    jump_refused: int = 0
    collision_refused: int = 0
    rate_limited: int = 0
    box_hits: int = 0
    floor_hits: int = 0
    floor_raised: int = 0
    loop_gaps: int = 0
    max_lag: float = 0.0
    min_tcp_z: float = math.inf
    infer_ms: list[float] = field(default_factory=list)
    pending: TickRecord | None = None
    pending_target: dict[str, float] | None = None
    jaw: _VirtualJaw | None = None     # shadow only
    rate_streak: int = 0


def smoothstep(x: float) -> float:
    s = min(max(x, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _arm_rad(joints: Mapping[str, float]) -> np.ndarray:
    return np.radians([float(joints[j]) for j in ARM])


def _arm_deg(q: np.ndarray) -> dict[str, float]:
    return {j: math.degrees(float(v)) for j, v in zip(ARM, q, strict=True)}


def _units(value: float) -> float:
    return min(max(float(value), 0.0), 100.0)


def _ramp_duration(largest_deg: float) -> float:
    """The smoothstep's peak speed is 1.5 times its mean: the peak stays at PREPOSITION_SPEED_DEG_S."""
    return max(RAMP_MIN_S, 1.5 * largest_deg / PREPOSITION_SPEED_DEG_S)


def _time_limit_s() -> float:
    """Pre-positioning at its slowest, the episode and a margin: the twin's backstop, read before `start`."""
    span = dict(zip(ARM, np.degrees(np.asarray(JOINT_HI) - np.asarray(JOINT_LO)), strict=True))
    stages = sum(_ramp_duration(max(span[j] for j in stage)) for stage in STAGES)
    retreat = math.ceil(TRAIN_HOVER_Z[1] / RETREAT_STEP) * RAMP_MIN_S
    hover = GRIP_FIRST_S + GRIP_WAIT_S + ARRIVE_TIMEOUT_S + SAG_AVERAGE_S
    return retreat + stages + hover + MAX_SECONDS + TIME_MARGIN_S


class Grasp:
    """The wrist-camera grasp student: from a hover 6 cm above the object until it is lifted."""

    hz = 100.0
    settings = GraspSettings

    # Set by `start`: what the run knows before anything moves.
    _simulated: bool
    _kin: Kinematics
    _rig: Rig
    _policy: GraspStudentPolicy
    _camera: WristCamera
    _touch: float
    _object_height: float
    _believed: _Believed
    _box: JointBox
    _hover: Hover
    _hover_deg: dict[str, float]
    _obstacles: tuple[Box, ...]
    _checker: CollisionChecker
    _observer: GraspObserver
    _finger_open: float
    _camera_s: float
    _guard: ImpactGuard

    def __init__(self, settings: GraspSettings) -> None:
        if settings.dump_every < 0:
            raise ValueError(f"chwyt: dump_every nie moze byc ujemne, jest {settings.dump_every}")
        if settings.dump_every > 0 and not settings.dump_dir.strip():
            raise ValueError("chwyt: dump_every bez dump_dir - zrzuty nie maja dokad isc")

        self.config = settings
        self.time_limit_s = _time_limit_s()
        self._dumps: Dumps | None = None
        self._reset()

    def _reset(self) -> None:
        """Everything one run changes: a second `start` on the same object begins from nothing."""
        if self._dumps is not None:
            self._dumps.close(self.summary)
        self.phase = Phase.PLAN
        self.ending: Ending | None = None
        #: The episode's numbers once it ended (also in summary.json).
        self.summary: dict[str, Any] | None = None

        self._history = JointHistory()
        self._ramps: deque[_Ramp] = deque()
        self._ramp_total = 0
        self._last_grip: float | None = None
        self._phase_t0 = 0.0
        self._jaw_sent: float | None = None
        self._sag_values: list[float] = []
        self._sag = 0.0
        self._episode: _Episode | None = None
        self._done: Move | None = None
        self._dumps = None

    # ================================================================== start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Refuses (`GraspRefused`, Polish) before anything moves; else sets the run up."""
        self._reset()
        self._check_robot(run)
        self._simulated = bool(run.simulated)

        self._kin = Kinematics()
        self._check_ranges()

        self._rig = self._load_rig()
        self._policy = self._load_policy()
        self._camera = self._check_camera(obs)

        self._touch = self._table_touch()
        self._object_height = self._height()
        self._believed = self._believe(obs)
        self._box = joint_box(self._kin, run.limits)
        self._hover = self._solve_hover()
        self._obstacles = (self._object_box(),)

        self._prepare()
        logger.info("grasp: %s, hover (%.3f, %.3f, %.3f) m, yaw %.1f deg, J6 margin %.1f deg%s",
                    self._believed.source, *self._hover.p, math.degrees(self._hover.yaw),
                    self._hover.j6_margin_deg, " (shadow)" if self.config.shadow else "")

    def _check_robot(self, run: RunInfo) -> None:
        if run.gripper is None:
            raise GraspRefused(f"chwyt: ramie {run.robot} nie ma chwytaka")
        if run.gripper != GRIPPER or set(run.joints) != {*ARM, GRIPPER}:
            raise GraspRefused(f"chwyt: ramie {run.robot} to nie Galaxea A1X - potrzeba stawow "
                               f"{', '.join(ARM)} i chwytaka {GRIPPER}, a sa: {', '.join(run.joints)}")

        speeds = {j: float(run.max_vel.get(j, 0.0)) for j in ARM}
        slow = {j: v for j, v in speeds.items() if v < ARM_SPEED_DEG_S - 1e-6}
        if slow:
            listed = ", ".join(f"{j} {v:.0f} st./s" for j, v in slow.items())
            raise GraspRefused(f"chwyt: nadzorca ogranicza {listed} - polityka uczyla sie przy "
                               f"{ARM_SPEED_DEG_S:.0f} st./s, a wolniejsze ramie po cichu dlawi jej ruchy")

    def _check_ranges(self) -> None:
        mismatch = range_mismatch(self._kin)
        if mismatch is not None:
            raise GraspRefused(f"chwyt: zakresy stawow modelu galaxeo.arm to nie zakresy z treningu ({mismatch})")

    def _load_rig(self) -> Rig:
        path = self.config.rig_json.strip()
        mode = "shadow" if self.config.shadow else "real"

        if not path:
            if not self._simulated:
                raise GraspRefused("chwyt: prawdziwe ramie potrzebuje pliku stanowiska (ustawienie rig_json)")
            return Rig.for_sim()

        try:
            rig = Rig.load(path)
        except RigError as exc:
            raise GraspRefused(f"chwyt: {exc}") from exc

        missing = rig.missing(mode, simulated=self._simulated, now=time.time())
        if missing:
            raise GraspRefused(f"chwyt: plik stanowiska {rig.path} nie pozwala na tryb {mode}: {', '.join(missing)}")
        return rig

    def _load_policy(self) -> GraspStudentPolicy:
        try:
            return GraspStudentPolicy.load(self.config.student_pt, self.config.student_sha256,
                                           threads=TORCH_THREADS)
        except CheckpointError as exc:
            raise GraspRefused(f"chwyt: nie wczytano modelu studenta: {exc}") from exc

    def _check_camera(self, obs: Observation) -> WristCamera:
        name = self.config.camera
        try:
            frame = obs.frames[name]
        except KeyError as exc:
            raise GraspRefused(f"chwyt: brak klatki z kamery {name!r} - wlacz ja dla strategii") from exc

        image = np.asarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise GraspRefused(f"chwyt: kamera {name!r} daje {image.dtype} {image.shape}, potrzeba RGB uint8")

        size = f"{image.shape[1]}x{image.shape[0]}"
        if self._simulated:
            if image.shape != (IMG_H, IMG_W, 3):
                raise GraspRefused(f"chwyt: kamera {name!r} symulowanego ramienia daje {size}, potrzeba "
                                   f"{IMG_W}x{IMG_H} (kamera na nadgarstku z robot.yaml)")
            return WristCamera()

        if image.shape != (RAW_H, RAW_W, 3):
            raise GraspRefused(f"chwyt: kamera {name!r} prawdziwego ramienia daje {size}, potrzeba surowej "
                               f"klatki {RAW_W}x{RAW_H}, ktora strategia przelicza na geometrie treningu")
        try:
            return WristCamera(wrist_intrinsics(self._rig))
        except CameraError as exc:
            raise GraspRefused(f"chwyt: kalibracja kamery na nadgarstku odrzucona: {exc}") from exc

    def _table_touch(self) -> float:
        touch = self._rig.touch_tcp_z()
        if touch is None:
            raise GraspRefused("chwyt: plik stanowiska nie ma dotyku stolu (table.touch_tcp_z)")
        return touch

    def _check_hover_clear(self) -> None:
        """The hover at the opening every episode step is judged at; this also compiles the scene now."""
        collision = self._checker.config_clear(self._hover.q, self._obstacles, opening=self._finger_open)
        if collision is not None:
            raise GraspRefused(f"chwyt: zawis koliduje przy otwartej szczece ({collision})")

    def _height(self) -> float:
        height = self._rig.object_height()
        return OBJECT_HEIGHT_DEFAULT if height is None else height

    def _believe(self, obs: Observation) -> _Believed:
        """The believed grasp: the observed object, else the rig file's taught point."""
        name = self.config.object_name
        seen = obs.objects.get(name)
        if seen is not None:
            position = np.asarray(seen.position, float).reshape(3)
            rotation = np.asarray(seen.rotation, float).reshape(3, 3)
            if not self._simulated:
                self._check_observed(name, float(seen.age_s), position)
            # The closing axis across the object's x faces, as training drew the grasp yaw.
            return _Believed(point=np.array([position[0], position[1], self._touch + TIP_CLEAR]),
                             yaw=math.atan2(rotation[1, 0], rotation[0, 0]),
                             source=f"obiekt {name!r} z obserwacji", object_z=float(position[2]))

        point = self._rig.believed()
        yaw = self._rig.grasp_yaw()
        if point is None or yaw is None:
            raise GraspRefused(f"chwyt: nie wiadomo, gdzie chwytac - w obserwacji nie ma obiektu {name!r}, "
                               "a plik stanowiska nie ma punktu chwytu (grasp.tcp, grasp.yaw)")
        return _Believed(point=point, yaw=float(yaw), source="punkt chwytu z pliku stanowiska", object_z=None)

    def _check_observed(self, name: str, age_s: float, position: np.ndarray) -> None:
        """On a real arm an observed object replaces the taught point only when it is fresh and stands
        where the table's height was measured: the believed grasp sits at the taught touch height."""
        if age_s > OBJECT_AGE_MAX_S:
            raise GraspRefused(f"chwyt: obiekt {name!r} widziany {age_s:.1f} s temu (limit {OBJECT_AGE_MAX_S:.0f} s) "
                               "- mogl sie od tego czasu przesunac")

        table = self._rig.table_tcp()
        far = math.inf if table is None else math.hypot(position[0] - table[0], position[1] - table[1])
        if far > TABLE_TOUCH_FAR:
            raise GraspRefused(f"chwyt: obiekt {name!r} stoi {far * 100:.0f} cm od miejsca dotyku stolu (limit "
                               f"{TABLE_TOUCH_FAR * 100:.0f} cm) - wysokosc stolu zmierzono tam, przy obiekcie moze "
                               "byc inna")

    def _solve_hover(self) -> Hover:
        try:
            hover = solve_hover(self._kin, self._believed.point, self._believed.yaw, self._box)
        except HoverError as exc:
            raise GraspRefused(f"chwyt: brak zawisu nad punktem chwytu ({exc.detail})") from exc

        if hover.outside_training:
            raise GraspRefused(f"chwyt: zawis poza obszarem treningu ({', '.join(hover.outside_training)}) - "
                               "obiekt po lewej stronie ramienia, 0,22-0,44 m od osi")
        return hover

    def _object_box(self) -> Box:
        """The object as a cube on the table at the believed point, turned by the believed yaw: an
        axis-aligned box around a turned cube sticks out 6 mm at 45 deg, enough to refuse every step
        of a jaw closed slightly off-centre."""
        half = self._object_height / 2.0
        point = self._believed.point
        return Box(center=(float(point[0]), float(point[1]), half), half=(half, half, half), name="obiekt",
                   yaw=self._believed.yaw)

    def _prepare(self) -> None:
        self._checker = CollisionChecker()
        self._finger_open = float(self._kin.ix.finger_open)
        self._check_hover_clear()
        self._observer = GraspObserver(self._kin)
        self._camera_s = self._rig.camera_latency_s()
        self._hover_deg = _arm_deg(self._hover.q)
        self._guard = ImpactGuard(ARM, lag_deg=PREPOSITION_LAG_DEG, lag_s=GUARD_LAG_S)

        # The first call of the network is several times slower than the rest.
        self._policy.act(np.zeros((IMG_H, IMG_W, 3), np.uint8), np.zeros(19, np.float32), np.zeros(2, np.float32))

        if self.config.dump_dir.strip():
            self._dumps = Dumps(Path(self.config.dump_dir), every=self.config.dump_every)

    # ================================================================== act

    def act(self, obs: Observation) -> Move:
        if self._done is not None:
            return self._done

        self._history.add(obs.t, obs.joints)
        try:
            self._watch_lag(obs)
            return self._step(obs)
        except _Ended as end:
            return self._finish(obs, end.ending)
        except Exception as exc:
            logger.exception("grasp: error in a tick")
            return self._finish(obs, Ending("error", f"chwyt przerwany bledem: {type(exc).__name__}: {exc}"[:200]))

    def stop(self, reason: str) -> None:
        """The twin ended the run (an answer's `done`, STOP, Home, preemption, the time limit)."""
        if self.ending is None:
            self.ending = Ending("stopped", f"zatrzymany przez blizniaka: {reason}")
            self.summary = self._summary()
            if self._episode is not None:
                self._emit(self._episode)
        if self._dumps is not None:
            self._dumps.close(self.summary)
            self._dumps = None

    def _step(self, obs: Observation) -> Move:
        if self.phase is Phase.PLAN:
            self._plan(obs)
        if self.phase is Phase.RAMPS:
            return self._ramp(obs)
        if self.phase is Phase.JAW:
            return self._open_jaw(obs)
        if self.phase is Phase.ARRIVE:
            return self._arrive(obs)
        if self.phase is Phase.SAG:
            return self._measure_sag(obs)
        return self._run_episode(obs)

    def _watch_lag(self, obs: Observation) -> None:
        """The impact guard's lag rule on the supervisor's command against the measurement; the
        observation has no efforts, so there is no effort rule."""
        sent = {j: float(obs.command[j]) for j in ARM if j in obs.command}
        hit = self._guard.feed(obs.t, sent, obs.joints, True)
        if hit is not None:
            raise _Ended(Ending(f"impact lag {hit.joint}",
                                f"{hit.joint} stoi {hit.value:.1f} st. za poleceniem - uderzenie albo blokada"))

    def _move(self, target: Mapping[str, float] | None = None) -> Move:
        if target is not None and GRIPPER in target:
            self._last_grip = float(target[GRIPPER])
        return Move(target=dict(target or {}), note=self._note())

    def _note(self) -> str:
        shadow = " (cien)" if self.config.shadow else ""
        if self.phase is Phase.RAMPS:
            done = self._ramp_total - len(self._ramps) + 1
            return f"chwyt{shadow}: dojazd do zawisu {done}/{self._ramp_total}"
        if self.phase is Phase.JAW:
            return f"chwyt{shadow}: otwieranie szczeki w zawisie"
        if self.phase is Phase.ARRIVE:
            return f"chwyt{shadow}: czekam, az ramie stanie w zawisie"
        if self.phase is Phase.SAG:
            return f"chwyt{shadow}: pomiar ugiecia w zawisie"
        ep = self._episode
        if ep is None:
            return f"chwyt{shadow}"
        return f"chwyt{shadow}: tik {ep.cmd.state.ticks + ep.skipped}/{MAX_TICKS}, proby {ep.cmd.state.attempts}"

    # ================================================================== pre-positioning

    def _plan(self, obs: Observation) -> None:
        """The retreat and the staged plan from where the arm stands, all checked before it moves."""
        start = _arm_rad(obs.joints)
        keep = _units(obs.command.get(GRIPPER, obs.joints[GRIPPER]))

        retreat = self._retreat_poses(start)
        stages = self._stage_poses(retreat[-1] if retreat else start, obs)

        pose = start
        for goal in [*retreat, *stages]:
            largest = float(np.max(np.abs(np.degrees(goal - pose))))
            self._ramps.append(_Ramp(start=pose, goal=goal, grip=keep, duration=_ramp_duration(largest)))
            pose = goal

        self._ramp_total = len(self._ramps)
        self.phase = Phase.RAMPS
        logger.info("grasp: %d straight-up steps, %d stages to the hover", len(retreat), len(stages))

    def _retreat_poses(self, start: np.ndarray) -> list[np.ndarray]:
        """Straight up to the hover height when the tool stands low near the hover, its tilt kept."""
        T = self._kin.fk(start)
        p = T[:3, 3]
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -float(T[2, 0])))))
        near = math.hypot(p[0] - self._hover.p[0], p[1] - self._hover.p[1]) < RETREAT_NEAR_XY
        top = float(self._hover.p[2])
        if not (tilt <= RETREAT_TILT_MAX_DEG and p[2] < top - RETREAT_BELOW and near):
            return []

        # The tool keeps its measured orientation: squaring a few degrees of tilt in the first 5 mm
        # step needs more than IK_MAX_JUMP_DEG from one joint in 10 to 25 % of low poses, and would
        # swing the fingertips sideways past the object. The staged plan squares it at the hover height.
        R = T[:3, :3].copy()
        q = start
        z = float(p[2])
        poses: list[np.ndarray] = []
        while z < top - 1e-9:
            reached = z
            z = min(z + RETREAT_STEP, top)
            acc = self._accept(np.array([p[0], p[1], z]), R, q, collision=False)
            if acc.q is None:
                detail = (f"krok w gore do z {z:.3f} m odrzucony ({acc.why}, {acc.step.pos_err * 1000:.2f} mm, "
                          f"{acc.step.rot_err * 1000:.2f} mrad, {math.degrees(acc.step.jump):.1f} st.)")
                # At long reach a vertical line leaves the warm solver's branch before the hover
                # height; well above the object the collision-checked staged plan takes over.
                if reached >= self._touch + self._object_height + RETREAT_CLEAR:
                    logger.warning("grasp: %s; the staged plan goes on from TCP z %.3f m", detail, reached)
                    return poses
                raise _Ended(Ending("ik", f"wycofanie w gore: {detail}"))
            poses.append(acc.q)
            q = acc.q
        return poses

    def _stage_poses(self, start: np.ndarray, obs: Observation) -> list[np.ndarray]:
        opening = self._widest_opening(obs)

        def clear(a: np.ndarray, b: np.ndarray) -> str | None:
            return self._checker.path_clear(a, b, self._obstacles, opening=opening)

        plan = stage_plan(start, self._hover.q, clear)
        if plan.collision is not None:
            raise _Ended(Ending("collision", f"kolizja na drodze do zawisu: {plan.collision}"))
        return list(plan.poses) or [self._hover.q.copy()]

    def _ramp(self, obs: Observation) -> Move:
        ramp = self._ramps[0]
        point = self._ramp_point(ramp, obs.t)
        target = {**_arm_deg(point.q), GRIPPER: ramp.grip}
        move = self._move(target)

        if point.finished:
            self._ramps.popleft()
        if not self._ramps:
            self._enter(Phase.JAW, obs)
        return move

    def _ramp_point(self, ramp: _Ramp, t: float) -> _RampPoint:
        if ramp.last_t is not None:
            ramp.elapsed += min(max(t - ramp.last_t, 0.0), 2.0 / self.hz)
        ramp.last_t = t
        s = smoothstep(ramp.elapsed / ramp.duration)
        return _RampPoint(q=ramp.start + s * (ramp.goal - ramp.start), finished=s >= 1.0)

    def _open_jaw(self, obs: Observation) -> Move:
        """99 first unless the last command was 99, so the A1X backend lets the opening through, then
        100; then until the measured jaw is open, at most GRIP_WAIT_S."""
        if self._jaw_sent is None:
            first = GRIP_OPEN if self._last_grip == GRIP_STAGE else GRIP_STAGE
            return self._send_jaw(first, obs)

        if self._jaw_sent == GRIP_STAGE:
            if obs.t - self._phase_t0 < GRIP_FIRST_S:
                return self._move()
            return self._send_jaw(GRIP_OPEN, obs)

        if obs.joints[GRIPPER] >= GRIP_STAGE:
            self._enter(Phase.ARRIVE, obs)
        elif obs.t - self._phase_t0 >= GRIP_WAIT_S:
            logger.warning("grasp: the jaw measures %.1f, not open within %.1f s", obs.joints[GRIPPER], GRIP_WAIT_S)
            self._enter(Phase.ARRIVE, obs)
        return self._move()

    def _send_jaw(self, opening: float, obs: Observation) -> Move:
        self._jaw_sent = opening
        self._phase_t0 = obs.t
        return self._move({**self._hover_deg, GRIPPER: opening})

    def _arrive(self, obs: Observation) -> Move:
        misses = self._misses(obs)
        if misses:
            if obs.t - self._phase_t0 >= ARRIVE_TIMEOUT_S:
                raise _Ended(Ending("not arrived", f"nie stanelo w zawisie w {ARRIVE_TIMEOUT_S:.0f} s: "
                                                   + "; ".join(misses)))
            return self._move()

        j5 = float(obs.joints[ARM[4]])
        if not self._simulated and abs(j5) > HOVER_J5_MEASURED_MAX_DEG:
            raise _Ended(Ending("j5 offset", f"{ARM[4]} {j5:.2f} st. w zawisie - sprawdz jego zero"))

        self._enter(Phase.SAG, obs)
        return self._measure_sag(obs)

    def _misses(self, obs: Observation) -> list[str]:
        m = obs.joints
        misses = [f"{j} {m[j] - self._hover_deg[j]:+.1f} st." for j in ARM
                  if abs(m[j] - self._hover_deg[j]) > ARRIVE_DEG]

        speed = self._history.speed_deg_s()
        misses += [f"{j} w ruchu {speed[j]:.0f} st./s" for j in ARM if abs(speed.get(j, 0.0)) >= ARRIVE_STILL_DEG_S]

        d = self._observer.tcp(_arm_rad(m)) - self._hover.p
        if abs(d[0]) > ARRIVE_M or abs(d[1]) > ARRIVE_M:
            misses.append(f"TCP xy obok o {d[0] * 1000:+.1f} {d[1] * 1000:+.1f} mm")
        if not -ARRIVE_M - SAG_MAX <= d[2] <= ARRIVE_M:
            misses.append(f"TCP z obok o {d[2] * 1000:+.1f} mm")
        return misses

    def _measure_sag(self, obs: Observation) -> Move:
        z = float(self._observer.tcp(_arm_rad(obs.joints))[2])
        self._sag_values.append(min(max(float(self._hover.p[2]) - z, 0.0), SAG_MAX))
        if obs.t - self._phase_t0 < SAG_AVERAGE_S:
            return self._move()

        self._sag = float(np.mean(self._sag_values))
        self._begin_episode(obs)
        return self._run_episode(obs)

    def _enter(self, phase: Phase, obs: Observation) -> None:
        self.phase = phase
        self._phase_t0 = obs.t

    # ================================================================== episode

    def _begin_episode(self, obs: Observation) -> None:
        self.phase = Phase.EPISODE
        floor = self._floor_of(self._sag)
        cmd = GraspCommand(self._hover.p, self._hover.yaw, self._hover.q, floor_z=floor)
        jaw = _VirtualJaw(opening=GRIP_OPEN, t=obs.t, rate=self._jaw_rate()) if self.config.shadow else None
        self._episode = _Episode(cmd=cmd, sag_used=self._sag, sag_est=self._sag, t0=obs.t, floor_z0=floor,
                                 next_policy_t=obs.t, target={**self._hover_deg, GRIPPER: GRIP_OPEN}, jaw=jaw)
        self._guard = ImpactGuard(ARM, lag_deg=EPISODE_LAG_DEG, lag_s=GUARD_LAG_S)
        logger.info("grasp episode: floor %.1f mm, sag %.1f mm", floor * 1000.0, self._sag * 1000.0)

    def _jaw_rate(self) -> float:
        """The virtual jaw's speed [units/s]: the jaw stroke measured on this rig (`gripper` of
        grasp_bench.py), else the contract's stream rate."""
        measured = self._rig.jaw_units_s()
        return STREAM_UNITS_S if measured is None else measured

    def _run_episode(self, obs: Observation) -> Move:
        ep = self._episode
        assert ep is not None
        if not self._due(ep, obs.t):
            return self._move()
        return self._tick(ep, obs)

    @staticmethod
    def _due(ep: _Episode, t: float) -> bool:
        """The policy steps on a fixed 50 ms grid of `obs.t`: a slow tick does not shift the next ones."""
        # The grid is a sum of 0.05 steps: a tick on a grid point must not round to just before it.
        if t < ep.next_policy_t - SCHEDULE_TOL_S:
            return False
        late = t - ep.next_policy_t
        ep.next_policy_t += TICK_S
        if late > TICK_S:
            ep.loop_gaps += 1
            logger.warning("grasp: policy tick %.0f ms late", late * 1000.0)
            ep.next_policy_t = t + TICK_S
        return True

    def _tick(self, ep: _Episode, obs: Observation) -> Move:
        self._judge(ep, obs)
        self._emit(ep)
        if ep.rate_streak >= RATE_LIMITED_END:
            raise _Ended(Ending("rate limited", f"nadzorca nie nadaza za polityka przez {ep.rate_streak} tikow z "
                                                f"rzedu - limit predkosci ramienia ponizej {ARM_SPEED_DEG_S:.0f} st./s "
                                                "(ograniczenie predkosci w panelu, velocity_scale)"))
        if ep.cmd.state.ticks + ep.skipped >= MAX_TICKS:
            raise _Ended(Ending("tick cap", f"limit {MAX_TICKS} tikow"))
        if obs.t - ep.t0 > MAX_SECONDS:
            raise _Ended(Ending("time cap", f"limit {MAX_SECONDS:.0f} s epizodu"))

        rec = self._record(ep, obs)
        try:
            tcp = self._score(ep, obs, rec)
        except _Skip as skip:
            return self._skipped(ep, rec, skip.reason)
        except _Ended as end:
            # The tick that explains a hard ending goes to ticks.jsonl too; it sent no target.
            rec.skip = end.ending.code
            ep.pending = rec
            raise

        ep.streak = 0
        ep.max_lag = max(ep.max_lag, rec.lag_deg)
        ep.min_tcp_z = min(ep.min_tcp_z, float(tcp[2]))
        target = self._target(ep)
        ep.pending = rec
        ep.pending_target = dict(target)

        self._check_success(ep, obs, tcp)
        return self._move(target)

    def _skipped(self, ep: _Episode, rec: TickRecord, reason: str) -> Move:
        """A soft guard: the previous target again; a streak of them ends the episode."""
        rec.kind = "skipped"
        rec.skip = reason
        ep.skipped += 1
        ep.streak += 1
        ep.pending = rec
        ep.pending_target = dict(ep.target)

        limit = SKIPS_END_LAG if reason == "lag" else SKIPS_END
        if ep.streak >= limit:
            raise _Ended(Ending(reason, f"{ep.streak} pominietych tikow z rzedu ({_SKIP_NOTES[reason]})"))
        return self._move(ep.target)

    def _score(self, ep: _Episode, obs: Observation, rec: TickRecord) -> np.ndarray:
        """Floor, frame, pairing, lag, observation, policy, bookkeeping, IK. Returns the measured TCP.

        Raises `_Skip` for a soft guard and `_Ended` for a hard one.
        """
        q_now = _arm_rad(obs.joints)
        tcp = self._observer.tcp(q_now)
        rec.tcp = tcp.tolist()
        rec.lag_deg = max(abs(math.degrees(float(ep.cmd.state.q_cmd[i])) - float(obs.joints[j]))
                          for i, j in enumerate(ARM))

        # Before the frame, pairing and lag guards: a skip streak must not switch off the floor and
        # the table ending while the arm trails a descending command.
        self._measured_floor(ep, q_now, tcp, self._widest_opening(obs))
        rec.floor_z = float(ep.cmd.floor_z)
        rec.sag = float(ep.sag_used)

        frame = self._frame(obs, rec)
        image = self._policy_image(frame.image)
        paired = self._paired(obs, frame, rec)

        if rec.lag_deg > LAG_HOLD_DEG:
            raise _Skip("lag")

        gap = GAP_CLOSED + GAP_PER_UNIT * self._jaw_opening(ep, obs)
        rec.gap = gap
        # The newest joints, as in PR 25 (see `observer`); only the pixel below is paired.
        vector = self._observer.build(q_now, gap, ep.cmd)
        if not np.all(np.isfinite(vector)):
            raise _Ended(Ending("non-finite", f"nieskonczona obserwacja {vector.tolist()}"))

        # The believed pixel from the joints at the exposure time, not the newest ones.
        hand = self._kin.link_pose(_arm_rad(paired.joints), HAND_BODY)
        uv = self._camera.believed_uv(hand, self._hover.believed)
        rec.uv = uv.tolist()

        started = perf_counter()
        action = np.asarray(self._policy.act(image, vector, uv), np.float32)
        rec.infer_ms = (perf_counter() - started) * 1000.0
        ep.infer_ms.append(rec.infer_ms)
        rec.action = action.ravel().tolist()
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise _Ended(Ending("non-finite", f"nieskonczona akcja {action.tolist()}"))

        self._dump(ep, image, uv, hand)
        self._step_command(ep, action, rec)
        rec.kind = "policy"
        ep.policy_ticks += 1
        return tcp

    @staticmethod
    def _jaw_opening(ep: _Episode, obs: Observation) -> float:
        """The opening the policy reads: the measured jaw, or in shadow the virtual one under the
        jaw command before this tick's step."""
        if ep.jaw is None:
            return _units(obs.joints[GRIPPER])
        return ep.jaw.follow(obs.t, 100.0 * float(ep.cmd.state.g_cmd))

    def _frame(self, obs: Observation, rec: TickRecord) -> Frame:
        try:
            frame = obs.frames[self.config.camera]
        except KeyError:
            raise _Skip("camera") from None

        age = float(frame.age_s)
        rec.frame_age_s = age
        if age > FRAME_END_S:
            raise _Ended(Ending("camera", f"klatka sprzed {age:.2f} s"))
        if age > FRAME_AGE_MAX_S:
            raise _Skip("camera")
        return frame

    def _policy_image(self, image: np.ndarray) -> np.ndarray:
        try:
            return self._camera.policy_image(image)
        except CameraError as exc:
            raise _Ended(Ending("camera", f"zla klatka z kamery: {exc}")) from exc

    def _paired(self, obs: Observation, frame: Frame, rec: TickRecord) -> JointSample:
        """The joint sample nearest the frame's exposure: the frame's age and the camera latency back."""
        t_exposure = obs.t - float(frame.age_s) - self._camera_s
        sample = self._history.nearest(t_exposure)
        if sample is None:
            raise _Skip("pairing")
        rec.pair_gap_s = abs(sample.t - t_exposure)
        if rec.pair_gap_s > PAIR_GAP_MAX_S:
            raise _Skip("pairing")
        return sample

    def _step_command(self, ep: _Episode, action: np.ndarray, rec: TickRecord) -> None:
        """One bookkeeping step; a refused IK step keeps p, yaw and q and commits the rest, as in training."""
        prop = ep.cmd.propose(action)
        rec.box_hit = prop.box_hit
        rec.floor_hit = prop.floor_hit
        rec.ceiling_hit = prop.ceiling_hit
        ep.box_hits += int(prop.box_hit)
        ep.floor_hits += int(prop.floor_hit)

        acc = self._accept(prop.p, prop.R, ep.cmd.state.q_cmd, collision=True)
        ep.cmd.commit(prop, acc.q)
        rec.ik_ok = acc.q is not None
        rec.ik_why = acc.why
        rec.ik_pos_err = acc.step.pos_err
        rec.ik_rot_err = acc.step.rot_err
        rec.ik_jump_deg = math.degrees(acc.step.jump)
        if acc.why == "collision":
            ep.collision_refused += 1
        elif acc.why:
            ep.ik_refused += 1
            ep.jump_refused += int(acc.why == "jump")

        s = ep.cmd.state
        rec.p_cmd = s.p_cmd.tolist()
        rec.yaw_cmd = float(s.yaw_cmd)
        rec.g_cmd = float(s.g_cmd)
        rec.attempts = int(s.attempts)

    def _target(self, ep: _Episode) -> dict[str, float]:
        grip = GRIP_OPEN if self.config.shadow else 100.0 * float(ep.cmd.state.g_cmd)
        target = {**_arm_deg(ep.cmd.state.q_cmd), GRIPPER: grip}
        if not all(math.isfinite(v) for v in target.values()):
            raise _Ended(Ending("non-finite", f"nieskonczony cel {target}"))
        ep.target = target
        return target

    def _check_success(self, ep: _Episode, obs: Observation, tcp: np.ndarray) -> None:
        """Training's condition when the object is observed: the jaw closed and the object risen more
        than LIFT. Without it: the closed jaw risen LIFT above its lowest, the object unknown."""
        if self.config.shadow or ep.cmd.state.g_cmd >= CLOSE_BELOW:
            return

        seen = obs.objects.get(self.config.object_name)
        start_z = self._believed.object_z
        if start_z is not None and seen is not None:
            rise = float(np.asarray(seen.position, float)[2]) - start_z
            if rise > LIFT:
                raise _Ended(Ending("success", f"obiekt uniesiony o {rise * 1000:.0f} mm", success=True))
            return

        if float(tcp[2]) >= ep.min_tcp_z + LIFT:
            raise _Ended(Ending("lifted", f"zamknieta szczeka uniesiona o {LIFT * 100:.0f} cm; czy obiekt jest "
                                          "w szczekach, strategia nie wie"))

    def _measured_floor(self, ep: _Episode, q: np.ndarray, tcp: np.ndarray, opening: float) -> None:
        """The command floor from the measured tool: sag while still, raised by a measured dip, and the
        table ending."""
        low = self._tool_low(q, float(tcp[2]), opening)

        # Sag grows with reach; estimated only while the arm is nearly still (tracking lag is not sag).
        speed = self._history.speed_deg_s()
        if speed and all(abs(speed.get(j, math.inf)) < ARRIVE_STILL_DEG_S for j in ARM):
            sag = float(self._kin.fk(ep.cmd.state.q_cmd)[2, 3]) - float(tcp[2])
            ep.sag_est += (sag - ep.sag_est) * min(1.0, TICK_S / SAG_FILTER_S)
            ep.sag_used = min(max(max(ep.sag_used, ep.sag_est), 0.0), SAG_LIMIT)

        floor = self._floor_of(ep.sag_used)

        # A measured tool under the start height raises the floor by the deficit, from the floor
        # itself: after a dive the command is already rising while the tool still overshoots, and a
        # floor lifted to that command would hold every later attempt 5 to 10 mm higher.
        deficit = self._touch + MEASURED_FLOOR_START - low
        if deficit > 0.0:
            floor = max(floor, min(max(float(ep.cmd.floor_z), floor) + deficit,
                                   self._touch + FLOOR_MARGIN + SAG_LIMIT))
            ep.floor_raised += 1

        # The floor only rises within an episode, and never above the ceiling.
        ep.cmd.floor_z = min(max(ep.cmd.floor_z, floor), float(ep.cmd.state.p0[2]) + CEILING)

        if low < self._touch + TABLE_END_BELOW:
            ep.table_ticks += 1
            if ep.table_ticks >= TABLE_END_TICKS:
                raise _Ended(Ending("table", f"narzedzie {low * 1000:.1f} mm (TCP {float(tcp[2]) * 1000:.1f} mm), "
                                             f"dotyk stolu {self._touch * 1000:.1f} mm"))
        else:
            ep.table_ticks = 0

    def _tool_low(self, q: np.ndarray, tcp_z: float, opening: float) -> float:
        """The measured tool height as a TCP height: the lower of the TCP and the lowest fingertip
        corner plus `TIP_AHEAD`. A wrist lagging at full lateral speed tilts the tool, and an open
        finger rolled 5 deg by joint 5 sits 5 mm lower than the TCP says."""
        tips = self._kin.fingertips(q, opening)
        return min(tcp_z, float(np.min(tips[:, 2])) + TIP_AHEAD)

    def _floor_of(self, sag: float) -> float:
        """The command floor on the commanded TCP height."""
        bench = self._touch + FLOOR_MARGIN + sag
        if not self.config.shadow:
            return bench
        # An open finger can stand over the object: its commanded tip stays SHADOW_CLEAR above the top.
        return max(bench, self._touch + self._object_height + SHADOW_CLEAR + sag,
                   float(self._hover.believed[2]) + SHADOW_FLOOR_ABOVE_BELIEVED)

    # ================================================================== shared

    def _accept(self, p: np.ndarray, R: np.ndarray, seed: np.ndarray, *, collision: bool) -> _Acceptance:
        """One warm IK step of training's solver and, on the accepted joints, one collision question.

        The object is checked with the fingers fully open: open plates straddle it, so the palm, the
        finger roots or a plate landing on it count, while a jaw closed on the object - inside its
        box by definition - does not refuse the lift. Fully open is also the widest the jaw can be
        for the table.
        """
        step = self._kin.ik_step(p, R, seed, iters=IK_ITERS, damping=IK_DAMPING, tol_pos=IK_TOL_POS,
                                 tol_rot=IK_TOL_ROT, lo=self._box.lo, hi=self._box.hi)
        if not step.converged:
            return _Acceptance(q=None, step=step, why="ik")
        if math.degrees(step.jump) > IK_MAX_JUMP_DEG:
            return _Acceptance(q=None, step=step, why="jump")
        if collision and self._checker.config_clear(step.q, self._obstacles, opening=self._finger_open) is not None:
            return _Acceptance(q=None, step=step, why="collision")
        return _Acceptance(q=step.q, step=step, why="")

    def _widest_opening(self, obs: Observation) -> float:
        """The widest jaw opening [m per finger] that can be true now: the jaw trails its command."""
        values = [float(obs.joints[GRIPPER])]
        if GRIPPER in obs.command:
            values.append(float(obs.command[GRIPPER]))
        if self._last_grip is not None:
            values.append(self._last_grip)
        return _units(max(values)) / 100.0 * self._finger_open

    def _dump(self, ep: _Episode, image: np.ndarray, uv: np.ndarray, hand: np.ndarray) -> None:
        if self._dumps is None or not self._dumps.due(ep.policy_ticks):
            return
        top_z = self._touch - TIP_AHEAD + self._object_height
        extra = {"cube top": self._camera.cube_top_uv(hand, self._believed.point[:2], top_z)}
        heat = self._policy.inputs(image, uv)[3]
        self._dumps.frame(ep.policy_ticks, image, uv, extra, heat)

    def _record(self, ep: _Episode, obs: Observation) -> TickRecord:
        s = ep.cmd.state
        return TickRecord(
            tick=s.ticks + ep.skipped, t=float(obs.t), kind="skipped", skip="", action=None, uv=None,
            p_cmd=s.p_cmd.tolist(), yaw_cmd=float(s.yaw_cmd), g_cmd=float(s.g_cmd), attempts=int(s.attempts),
            ik_ok=None, ik_why="", ik_pos_err=None, ik_rot_err=None, ik_jump_deg=None, box_hit=False,
            floor_hit=False, ceiling_hit=False, floor_z=float(ep.cmd.floor_z), sag=float(ep.sag_used),
            tcp=[math.nan] * 3, gap=math.nan, lag_deg=math.nan, rate_limited=False, frame_age_s=None,
            pair_gap_s=None, infer_ms=None)

    def _judge(self, ep: _Episode, obs: Observation) -> None:
        """The previous tick's rate-limit verdict: its target against the supervisor's command now."""
        rec = ep.pending
        target = ep.pending_target
        # Judged once, on a later tick: the supervisor has not seen this tick's own target yet.
        if rec is None or not target or not obs.command or rec.t >= obs.t:
            return
        diffs = [abs(float(target[j]) - float(obs.command[j])) for j in ARM if j in target and j in obs.command]
        rec.rate_limited = bool(diffs) and max(diffs) > RATE_LIMITED_DEG
        ep.rate_limited += int(rec.rate_limited)
        ep.rate_streak = ep.rate_streak + 1 if rec.rate_limited else 0

    def _emit(self, ep: _Episode) -> None:
        if ep.pending is not None and self._dumps is not None:
            self._dumps.tick(ep.pending)
        ep.pending = None
        ep.pending_target = None

    # ================================================================== endings

    def _finish(self, obs: Observation, ending: Ending) -> Move:
        """An ending that may hold the object keeps the last episode target: the measurement trails the
        command, and holding it would step back with the object in the jaw. Every other ending holds
        where the arm measures, jaw included: a joint pressed against something stops pushing."""
        ep = self._episode
        if ep is not None and ending.code in KEEP_GRIP:
            target = dict(ep.target)
        else:
            target = {j: float(obs.joints[j]) for j in JOINTS_AND_GRIPPER}

        if ep is not None:
            self._judge(ep, obs)
            self._emit(ep)

        self.phase = Phase.DONE
        self.ending = ending
        self.summary = self._summary()
        level = logging.INFO if ending.code in KEEP_GRIP else logging.WARNING
        logger.log(level, "grasp ended: %s (%s)", ending.code, ending.note)

        self._done = Move(target=target, done=True, success=ending.success, note=f"chwyt: {ending.note}")
        return self._done

    def _summary(self) -> dict[str, Any]:
        ending = self.ending or Ending("", "")
        out: dict[str, Any] = {"ending": ending.code, "note": ending.note, "success": ending.success,
                               "shadow": self.config.shadow}
        ep = self._episode
        if ep is None:
            return out

        infer = np.asarray(ep.infer_ms, float)
        out.update(
            ticks=ep.cmd.state.ticks + ep.skipped, policy_ticks=ep.policy_ticks, skipped=ep.skipped,
            attempts=int(ep.cmd.state.attempts), ik_refused=ep.ik_refused, jump_refused=ep.jump_refused,
            collision_refused=ep.collision_refused, rate_limited=ep.rate_limited, box_hits=ep.box_hits,
            floor_hits=ep.floor_hits, floor_raised=ep.floor_raised, loop_gaps=ep.loop_gaps,
            max_lag_deg=ep.max_lag, min_tcp_z=ep.min_tcp_z, floor_z=ep.floor_z0, floor_z_end=float(ep.cmd.floor_z),
            sag=ep.sag_used,
            infer_ms_median=float(np.median(infer)) if infer.size else None,
            infer_ms_p95=float(np.percentile(infer, 95)) if infer.size else None,
        )
        return out


_SKIP_NOTES = {
    "camera": "brak swiezej klatki z kamery",
    "pairing": "brak probki stawow blisko chwili zdjecia",
    "lag": f"ramie ponad {LAG_HOLD_DEG:.0f} st. za poleceniem",
}


if TYPE_CHECKING:
    # The twin's protocol checks `settings` and `act` for a type checker; nothing runs here.
    _: type[Strategy[GraspSettings]] = Grasp
