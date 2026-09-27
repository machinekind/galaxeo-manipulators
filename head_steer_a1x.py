#!/usr/bin/env python3
"""Steer the A1X with your head: MediaPipe Face Landmarker on a webcam, galaxeo underneath.

    python head_steer_a1x.py --dry-run     # camera, head angles and targets; no arm
    python head_steer_a1x.py               # lift, SPACE follows the head, X lowers the arm and exits

On start the arm goes from wherever it rests to a "look" pose (tool forward, ~0.29 m over
the table) with galaxeo.arm guarded moves: collision-checked, <= 40 deg per step, impact
guard. SPACE then streams p_des from the head:

    head turn  -> J1 (base)          +-45 deg around the look pose
    head nod   -> J4 (wrist pitch)   +-30 deg
    head tilt  -> J6 (tool roll)     +-45 deg

Always relative: on SPACE the head pose and the arm pose are captured, and only the
difference moves the arm, so nothing jumps. Losing the face holds the arm; the face coming
back re-anchors without a jump. X (or Ctrl-C) stops following and lowers the arm through
the look pose to a relaxed pose (folded, wrist straight): the one to cut power in, since
the A1X has no brakes (docs/SAFETY.md).

The stream (`Streamer`) starts from the measured pose, slews at --max-speed (30 deg/s),
clamps to the joint limits, and stops on stale or frozen feedback, on a joint that stays
away from its setpoint (deaf or blocked arm), or when an effort moves more than
--effort-stop from its value at SPACE (a push, a hit). ESC stops too; SPACE resumes.

Needs: pip install -e ".[arm,head]" (numpy, mujoco, mediapipe, opencv). The face model
(~3.8 MB, Apache-2.0, Google) is downloaded on first use to models/face_landmarker.task.
Keys in the window: SPACE follow on/off, ESC stop, C re-anchor, 1/2/3 flip turn/nod/tilt, X quit.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("head_steer")

FACE_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

JOINT_NAMES = tuple(f"J{i}" for i in range(1, 7))
#: J2..J5 of the look pose (tool forward, ~0.29 m over the table). J1 and J6 stay as at start.
LOOK = (44.5, -13.1, -23.8, 0.0)
#: Head axis -> (joint index, default sign). Checked on the rig 2026-09-27 with the camera in
#: front of the operator: J4 + tilts the tool DOWN, so a nod up must drive J4 negative.
AXES = {"yaw": (0, 1.0), "pitch": (3, -1.0), "roll": (5, 1.0)}
SEGMENT_DEG = 40.0          # guarded moves: the primitive refuses jumps over 45 deg
MOVE_SPEED = 8.0            # deg/s, lift and lowering
FACE_LOST_S = 0.3
ESC = 27


# ------------------------------------------------------------------------------ head
class OneEuro:
    """One Euro filter (Casiez et al. 2012) for one scalar: smooth at rest, little lag when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[float] = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, dt: float) -> float:
        if self.x is None or dt <= 0.0:
            self.x = value if self.x is None else self.x
            return self.x
        dx = (value - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt) * (value - self.x)
        return self.x


@dataclass
class HeadPose:
    yaw: float      # deg, + = face turned toward the right edge of the (unmirrored) image
    pitch: float    # deg, + = face normal up
    roll: float     # deg, + = counter-clockwise in the image
    points: np.ndarray


def head_pose(matrix) -> tuple:
    """Head angles from MediaPipe's facial transformation matrix (x right, y up, z to the camera):
    where the face normal (z) points, and the face's x axis in the image plane."""
    R = np.asarray(matrix, float)[:3, :3]
    n, u = R[:, 2], R[:, 0]
    return (math.degrees(math.atan2(n[0], n[2])),
            math.degrees(math.atan2(n[1], math.hypot(n[0], n[2]))),
            math.degrees(math.atan2(u[1], u[0])))


def face_model() -> Path:
    if FACE_MODEL.is_file() and FACE_MODEL.stat().st_size > 0:
        return FACE_MODEL
    log.info("downloading the MediaPipe face model to %s", FACE_MODEL)
    part = FACE_MODEL.with_suffix(".part")
    try:
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=60) as r:  # noqa: S310
            data = r.read()
        FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(data)
        part.replace(FACE_MODEL)
    except Exception as ex:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"cannot download the face model ({ex}); save {FACE_MODEL_URL} as {FACE_MODEL}") from ex
    return FACE_MODEL


class FaceTracker:
    def __init__(self, video: bool = True):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._cv2, self._mp, self._video, self._last_ms = cv2, mp, video, -1
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model())),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1, output_facial_transformation_matrixes=True))

    def process(self, bgr: np.ndarray, t: float) -> Optional[HeadPose]:
        rgb = np.ascontiguousarray(self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            self._last_ms = max(int(t * 1000.0), self._last_ms + 1)
            res = self._lm.detect_for_video(image, self._last_ms)
        else:
            res = self._lm.detect(image)
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return None
        h, w = bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        return HeadPose(*head_pose(res.facial_transformation_matrixes[0]), pts)

    def close(self) -> None:
        self._lm.close()


def open_camera(index: int):
    import cv2
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def find_face_camera(max_index: int = 6) -> Optional[int]:
    """The first camera that shows a face (a few frames each, for the exposure to settle)."""
    tracker = FaceTracker(video=False)
    try:
        for index in range(max_index):
            cap = open_camera(index)
            if cap is None:
                continue
            try:
                for _ in range(15):
                    ok, img = cap.read()
                    if ok and img is not None and tracker.process(img, 0.0) is not None:
                        return index
            finally:
                cap.release()
    finally:
        tracker.close()
    return None


# ---------------------------------------------------------------------------- stream
class Streamer:
    """p_des stream for following: one `step` per 1/rate s, run by a thread (or a test).

    Nothing is sent until the first `set_target`. The stream starts from the MEASURED pose and
    moves toward the target at most `max_speed` deg/s, clamped to [lo, hi]. Stale feedback,
    frozen telemetry (FF 2) or a deaf / blocked joint latch `fault`, drop the target and stop
    sending: the uncommanded arm holds where it is. `clear()` rearms after a fault.
    """

    def __init__(self, transport, lo, hi, max_speed: float = 30.0, rate: float = 200.0,
                 stale_s: float = 0.15, deaf_deg: float = 3.0, deaf_s: float = 1.0):
        self.t = transport
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.max_step = math.radians(max_speed)
        self.period = 1.0 / rate
        self.stale_s, self.deaf_rad, self.deaf_s = stale_s, math.radians(deaf_deg), deaf_s
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None
        self._p: Optional[np.ndarray] = None
        self._last: Optional[float] = None
        self._deaf_since: dict = {}
        self.reading = None
        self.fault = ""
        self._run = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_target(self, q) -> None:
        with self._lock:
            if not self.fault:
                self._target = np.clip(np.asarray(q, float), self.lo, self.hi)

    def hold(self) -> None:
        """Target = the latest measured pose (a STOP that keeps the arm where it is)."""
        with self._lock:
            if self.reading is not None and not self.fault:
                self._target = np.clip(self.reading.q.copy(), self.lo, self.hi)
                self._p = self.reading.q.copy()

    def fresh(self):
        """The latest reading if it is younger than `stale_s`, else None."""
        with self._lock:
            r = self.reading
            return r if r is not None and self.t.now() - r.t <= self.stale_s else None

    def clear(self) -> None:
        with self._lock:
            self.fault, self._target, self._p, self._deaf_since = "", None, None, {}

    def _trip(self, why: str) -> None:
        if not self.fault:
            log.warning("stream stopped: %s", why)
        self.fault, self._target, self._p, self._deaf_since = why, None, None, {}

    def step(self) -> None:
        with self._lock:
            r = self.t.read()
            now = self.t.now()
            dt = self.period if self._last is None else min(max(now - self._last, 0.0), 2 * self.period)
            self._last = now
            if r is None or now - r.t > self.stale_s:
                self._trip("no feedback from the arm" if r is None else
                           f"feedback {1000 * (now - r.t):.0f} ms old")
                return
            if r.frozen:
                self._trip("telemetry frozen (FF 2?) - power-cycle the arm")
                return
            self.reading = r
            if self._target is None or self.fault:
                return
            if self._p is None:
                self._p = r.q.copy()                       # start from the MEASURED pose
            step = self.max_step * dt
            self._p = self._p + np.clip(self._target - self._p, -step, step)
            self.t.write(self._p)
            for j in range(6):
                if abs(self._p[j] - r.q[j]) > self.deaf_rad and abs(r.dq[j]) < math.radians(0.3):
                    t0 = self._deaf_since.setdefault(j, now)
                    if now - t0 >= self.deaf_s:
                        self._trip(f"{JOINT_NAMES[j]} {math.degrees(self._p[j] - r.q[j]):+.1f} deg from its "
                                   f"setpoint and not moving - arm deaf or blocked")
                        return
                else:
                    self._deaf_since.pop(j, None)

    def start(self) -> None:
        self._run.set()
        self._thread = threading.Thread(target=self._loop, name="head-steer-stream", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        nxt = time.monotonic()
        while self._run.is_set():
            try:
                self.step()
            except Exception as ex:                      # adapter pulled, bus error
                with self._lock:
                    self._trip(f"bus error: {ex}")
            nxt += self.period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.05:
                nxt = time.monotonic()

    def stop(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None


# -------------------------------------------------------------------------- mapping
class HeadSteer:
    """Head angles -> joint targets, relative to the head and arm poses captured on engage."""

    def __init__(self, look_deg, windows_deg, gain: float = 1.0, deadzone: float = 2.0,
                 effort_stop: float = 10.0, filter_cutoff: float = 1.0, filter_beta: float = 0.05,
                 invert=(), stream: Optional[Streamer] = None, clock: Callable[[], float] = time.monotonic):
        self.look = np.asarray(look_deg, float)
        self.windows = dict(windows_deg)
        self.gain, self.deadzone, self.effort_stop = gain, deadzone, effort_stop
        self.sign = {k: (-s if k in invert else s) for k, (_, s) in AXES.items()}
        self.filters = {k: OneEuro(filter_cutoff, filter_beta) for k in AXES}
        self.stream, self.clock = stream, clock
        self.engaged = False
        self.stopped = ""
        self.targets = self.look.copy()
        self.anchor_head: Optional[dict] = None
        self.anchor_targets = self.targets.copy()
        self.anchor_effort: Optional[np.ndarray] = None
        self.last_face = -math.inf
        self.message, self.message_t = "", -math.inf

    def note(self, text: str) -> None:
        self.message, self.message_t = text, self.clock()
        log.info(text)

    def engage(self, face_visible: bool) -> None:
        if not face_visible:
            self.note("no face - look at the camera")
            return
        if self.stream is not None:
            self.stream.clear()
            r = self.stream.fresh()
            if r is None:
                # An old reading as the target would drive the arm back to where it was before
                # the link dropped - only a fresh measurement may seed it.
                self.note("no fresh feedback from the arm")
                return
            self.targets = np.degrees(r.q)
            self.anchor_effort = np.asarray(r.effort, float).copy()
            self.stream.set_target(r.q)
        self.anchor_head = None
        self.engaged, self.stopped = True, ""
        self.note("following ON")

    def disengage(self, why: str = "") -> None:
        self.engaged, self.anchor_head = False, None
        if self.stream is not None:
            self.stream.hold()
        self.note(why or "following OFF - the arm holds")

    def stop(self, why: str) -> None:
        self.engaged, self.stopped, self.anchor_head = False, why, None
        if self.stream is not None:
            self.stream.hold()
        log.warning("STOP: %s", why)

    def release(self) -> None:
        """Re-anchor on the next face: the target stays where it is."""
        self.anchor_head = None

    def flip(self, axis: str) -> None:
        self.sign[axis] *= -1.0
        self.release()
        self.note(f"{axis}: sign {'-' if self.sign[axis] < 0 else '+'}")

    def step(self, head: Optional[HeadPose], dt: float) -> None:
        now = self.clock()
        if head is not None:
            self.last_face = now
        if self.engaged and self.stream is not None:
            if self.stream.fault:
                self.stop(self.stream.fault)
                return
            r = self.stream.reading
            if r is not None and self.anchor_effort is not None:
                d = np.abs(np.asarray(r.effort, float) - self.anchor_effort)
                j = int(np.argmax(d))
                if d[j] > self.effort_stop:
                    self.stop(f"{JOINT_NAMES[j]} effort {r.effort[j]:+.1f} (was {self.anchor_effort[j]:+.1f} "
                              f"at SPACE) - a push or a hit?")
                    return
        if not self.engaged:
            return
        if head is None:
            if now - self.last_face > FACE_LOST_S:
                self.release()                               # the arm holds its last target
            return
        raw = {"yaw": head.yaw, "pitch": head.pitch, "roll": head.roll}
        smooth = {k: self.filters[k](v, max(dt, 1e-3)) for k, v in raw.items()}
        if self.anchor_head is None:
            self.anchor_head = smooth
            self.anchor_targets = self.targets.copy()
        for axis, (j, _) in AXES.items():
            d = smooth[axis] - self.anchor_head[axis]
            d = 0.0 if abs(d) < self.deadzone else d - math.copysign(self.deadzone, d)
            want = self.anchor_targets[j] + self.sign[axis] * self.gain * d
            w = self.windows[axis]
            self.targets[j] = min(max(want, self.look[j] - w), self.look[j] + w)
        if self.stream is not None:
            self.stream.set_target(np.radians(self.targets))


# ------------------------------------------------------------------- guarded moves
def guarded_path(arm, transport, poses_deg, speed: float = MOVE_SPEED) -> np.ndarray:
    """Joint poses [deg] through galaxeo.arm guarded moves. Every segment is split into steps
    of <= SEGMENT_DEG and every step is checked (limits, collisions) BEFORE anything moves.
    Returns the measured pose [deg]."""
    q = measured(arm, transport)
    steps, prev = [], q
    for pose in poses_deg:
        goal = np.radians(pose)
        n = max(1, math.ceil(math.degrees(float(np.max(np.abs(goal - prev)))) / SEGMENT_DEG))
        for k in range(1, n + 1):
            s = prev + (goal - prev) * k / n
            bad = np.flatnonzero((s < arm.lo - 1e-6) | (s > arm.hi + 1e-6))
            if bad.size:
                raise RuntimeError(f"{np.round(np.degrees(s), 1).tolist()} outside the limits of "
                                   f"{[JOINT_NAMES[i] for i in bad]}")
            why = arm.collision.path_clear(steps[-1] if steps else q, s, ())
            if why is not None:
                raise RuntimeError(f"collision on the way to {np.round(np.degrees(s), 1).tolist()}: {why}")
            steps.append(s)
        prev = goal
    arm.set_armed(True)
    try:
        for k, s in enumerate(steps, 1):
            log.info("  step %d/%d -> %s", k, len(steps), np.round(np.degrees(s), 1).tolist())
            r = arm.move(s, speed)
            if r.state != "reached":
                raise RuntimeError(f"move ended '{r.state}': {r.reason} {r.detail} - the arm holds")
    finally:
        arm.set_armed(False)
    return np.degrees(measured(arm, transport))


def measured(arm, transport, secs: float = 2.0) -> np.ndarray:
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        transport.read()
        q = arm.measured()
        if q is not None:
            return q
        time.sleep(0.01)
    raise RuntimeError("no fresh 0x052 feedback - arm unpowered or the CAN link is down")


def look_pose(start_deg) -> list:
    return [start_deg[0], *LOOK, start_deg[5]]


def rest_pose(start_deg) -> list:
    """Folded, wrist straight, base and roll as at start. J2 and J3 half a degree off their
    0-deg stops: a joint commanded into its stop pushes there forever."""
    return [start_deg[0], 0.5, -0.5, 0.0, 0.0, start_deg[5]]


def return_to_rest(arm, transport, start_deg, attempts: int = 2) -> np.ndarray:
    """Up to the look pose (if lifted), then down to the relaxed pose; one retry from the new
    measured pose after a transient error."""
    for attempt in range(1, attempts + 1):
        try:
            now = np.degrees(measured(arm, transport))
            look = look_pose(start_deg)
            path = [look] if now[1] > 10.0 and np.max(np.abs(now - look)) > 0.5 else []
            log.info("lowering to the relaxed pose %s (attempt %d)", np.round(rest_pose(start_deg), 1).tolist(),
                     attempt)
            done = guarded_path(arm, transport, path + [rest_pose(start_deg)])
            log.info("arm in the relaxed pose: %s", np.round(done, 1).tolist())
            return done
        except Exception as ex:
            log.error("lowering, attempt %d: %s", attempt, ex)
            if attempt == attempts:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------------------- ui
def draw(img, steer: HeadSteer, head: Optional[HeadPose], meas_deg, fps: float):
    import cv2
    view = cv2.flip(img, 1)                              # mirror: natural for the operator
    w = view.shape[1]
    if head is not None:
        for x, y in head.points[::6]:
            cv2.circle(view, (int(w - x), int(y)), 1, (0, 220, 0), -1)
    if steer.stopped:
        lines, color = [f"STOP: {steer.stopped}"], (0, 0, 255)
    elif steer.engaged:
        lines, color = ["FOLLOWING"], (0, 200, 0)
    else:
        lines, color = ["OFF - SPACE to follow"], (0, 200, 255)
    lines.append("head: no face" if head is None else
                 f"head: turn {head.yaw:+5.1f}  nod {head.pitch:+5.1f}  tilt {head.roll:+5.1f}")
    t = steer.targets
    lines.append(f"target J1 {t[0]:+6.1f}  J4 {t[3]:+6.1f}  J6 {t[5]:+6.1f}")
    if meas_deg is not None:
        lines.append(f"meas.  J1 {meas_deg[0]:+6.1f}  J4 {meas_deg[3]:+6.1f}  J6 {meas_deg[5]:+6.1f}")
    signs = " ".join(f"{k}{'-' if steer.sign[a] < 0 else '+'}" for k, a in zip("123", AXES))
    lines.append(f"{fps:4.1f} fps   signs (1 2 3): {signs}")
    if steer.message and steer.clock() - steer.message_t < 3.0:
        lines.append(steer.message)
    for i, text in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color if i == 0 else (255, 255, 255),
                    1, cv2.LINE_AA)
    cv2.putText(view, "SPACE follow  ESC stop  C re-anchor  1/2/3 flip axis  X quit", (10, view.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def run(cap, steer: HeadSteer) -> None:
    import cv2
    tracker = FaceTracker()
    window = "A1X head steer"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    prev, fps = time.monotonic(), 0.0
    try:
        while True:
            ok, img = cap.read()
            now = time.monotonic()
            dt, prev = min(now - prev, 0.25), now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
            if not ok or img is None:
                steer.step(None, dt)
                if cv2.waitKey(10) & 0xFF in (ord("x"), ord("X")):
                    break
                continue
            head = tracker.process(img, now)
            steer.step(head, dt)
            r = steer.stream.reading if steer.stream is not None else None
            cv2.imshow(window, draw(img, steer, head, None if r is None else np.degrees(r.q), fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("x"), ord("X")):
                break
            if key == ESC:
                steer.stop("ESC")
            elif key == ord(" "):
                steer.disengage() if steer.engaged else steer.engage(head is not None)
            elif key in (ord("c"), ord("C")):
                steer.release()
                steer.note("re-anchored on the current head pose")
            elif key in (ord("1"), ord("2"), ord("3")):
                steer.flip(tuple(AXES)[key - ord("1")])
    finally:
        tracker.close()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=None, help="can0 (Linux), xcan (macOS; Windows with WinUSB), "
                                                  "PCAN_USBBUSn; default: galaxeo.bus.default_iface()")
    ap.add_argument("--camera", type=int, default=None, help="camera index; default: the first one showing a face")
    ap.add_argument("--dry-run", action="store_true", help="no arm: camera, head angles and targets only")
    ap.add_argument("--gain", type=float, default=1.0, help="joint deg per head deg")
    ap.add_argument("--deadzone", type=float, default=2.0, help="head angle dead zone, deg")
    ap.add_argument("--window-yaw", type=float, default=45.0, help="J1 range around the look pose, +-deg")
    ap.add_argument("--window-pitch", type=float, default=30.0, help="J4 range around the look pose, +-deg")
    ap.add_argument("--window-roll", type=float, default=45.0, help="J6 range around the look pose, +-deg")
    ap.add_argument("--max-speed", type=float, default=30.0, help="stream slew limit, deg/s")
    ap.add_argument("--effort-stop", type=float, default=10.0,
                    help="STOP when a joint's effort moves this far from its value at SPACE")
    ap.add_argument("--invert", action="append", default=[], choices=tuple(AXES),
                    help="flip an axis (repeatable); keys 1/2/3 do it live")
    ap.add_argument("--filter-cutoff", type=float, default=1.0, help="One Euro min cutoff, Hz")
    ap.add_argument("--filter-beta", type=float, default=0.05, help="One Euro beta")
    ap.add_argument("--no-lower", action="store_true", help="on exit do NOT lower to the relaxed pose")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index = a.camera if a.camera is not None else find_face_camera()
    if index is None:
        sys.exit("no camera shows a face; pass --camera N")
    cap = open_camera(index)
    if cap is None:
        sys.exit(f"cannot open camera {index}")
    log.info("camera %d", index)
    windows = {"yaw": a.window_yaw, "pitch": a.window_pitch, "roll": a.window_roll}
    opts = dict(gain=a.gain, deadzone=a.deadzone, effort_stop=a.effort_stop, filter_cutoff=a.filter_cutoff,
                filter_beta=a.filter_beta, invert=tuple(a.invert))
    if a.dry_run:
        try:
            run(cap, HeadSteer([0.0, *LOOK, 0.0], windows, **opts))
        finally:
            cap.release()
        return 0

    from galaxeo.arm import Arm, BusTransport
    from galaxeo.bus import default_iface, open_bus

    iface = a.iface or default_iface()
    log.info("CAN interface: %s", iface)
    bus = open_bus(iface)                     # one open for the whole session
    transport = BusTransport(bus, tx=True)
    arm = Arm(transport, rate_hz=200.0)
    stream: Optional[Streamer] = None
    start = None
    code = 0
    try:
        start = np.degrees(measured(arm, transport))
        log.info("start %s -> look pose %s", np.round(start, 1).tolist(), np.round(look_pose(start), 1).tolist())
        guarded_path(arm, transport, [look_pose(start)])
        stream = Streamer(transport, arm.lo, arm.hi, max_speed=a.max_speed)
        stream.start()
        run(cap, HeadSteer(look_pose(start), windows, stream=stream, **opts))
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        log.error("%s", ex)
        code = 1
    finally:
        cap.release()
        if stream is not None:
            stream.stop()                     # the stream ends; the uncommanded arm holds
        try:
            if start is not None and not a.no_lower:
                return_to_rest(arm, transport, start)
        except Exception:
            log.error("lowering failed. The arm holds its pose - do NOT cut power before it is down "
                      "(run this again, or jog_a1x.py / move_to_point_a1x.py).")
            code = 1
        finally:
            arm.stop("exit")
            bus.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
