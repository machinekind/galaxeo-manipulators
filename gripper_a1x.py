#!/usr/bin/env python3
"""Measure the A1X jaw: stroke time and effort, free and on a 3 cm cube, at >= 150 samples/s.

    python3 gripper_a1x.py --i-know                         # the real arm on the default CAN link
    python3 gripper_a1x.py --i-know --out jaw.npz           # also keep the raw traces and the summary
    python3 gripper_a1x.py --i-know --config local.yaml     # the jaw as the twin's a1x section sets it
    python3 gripper_a1x.py --sim                            # a rehearsal on the simulated jaw

The jaw makes six free moves (99, 100, 0, 100, 0, 100), then closes and opens on a 3 cm cube that
the operator puts between the plates. The arm stays still. The `a1x` backend drives the jaw, as in the
twin. The freeze (`a1x.gripper_force`) is off so that it cannot cut a stroke short, and a move in which
it fired anyway does not count.

The tool prints one row per move and a summary: the close and open stroke times, the free and loaded
effort peaks, and whether they separate. When they do, it suggests a freeze threshold halfway between,
for `a1x.gripper_force` in the twin's config.

Type s and Enter when you see the jaw stop; that time is the operator's stroke time. A bare Enter is
STOP, and so is Ctrl-C (docs/SAFETY.md). Nothing is written without --out.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench_a1x import (
    FREEZE_OFF,
    KEYS_POLL_S,
    BackendLink,
    Keys,
    Refusal,
    SimJaw,
    Station,
    a1x_settings,
    arm_options,
    confirm_motion,
    listen_for_other_senders,
    open_station,
    operated,
    require_consent,
    run,
)

TOOL = "gripper_a1x.py"

#: 99 first: a first command within 0.5 of the opening the backend reports (100) sends nothing.
GRIPPER_FREE_MOVES = (99.0, 100.0, 0.0, 100.0, 0.0, 100.0)
GRIPPER_LOADED_MOVES = (0.0, 100.0)
GRIPPER_SETTLE_S = 2.0                # after each command; the A1X stream needs 1.0 to 1.3 s for a stroke
PRE_MOVE_S = 0.5                      # logged before each command: the noise band of its effort
GRIPPER_POLL_S = 0.001
MIN_RATE_HZ = 150.0
NOISE_SIGMAS = 3.0
EFFORT_NOISE_MIN = 0.05               # backend effort units; a flat trace would give a zero band
STROKE_MIN_UNITS = 50.0               # a move shorter than this is not a stroke
STROKE_EDGE_UNITS = 1.0               # start and end of a stroke: this far from its end values
SEPARABLE_RATIO = 1.5
OPERATOR_MIN_S = 0.05                 # an operator line sooner than this after a command was typed before it


@dataclass(frozen=True)
class JawSample:
    t: float                          # station clock at the poll that saw a new backend stamp
    effort: float
    streamed: float
    commanded: float
    frozen: bool


@dataclass
class EffortLog:
    """One jaw move: gripper states, `t` relative to the command (before it < 0)."""

    t: np.ndarray
    effort: np.ndarray
    streamed: np.ndarray
    commanded: np.ndarray
    frozen: np.ndarray
    rate_hz: float
    valid: bool                       # False when the freeze fired during the move
    target: float
    t0: float                         # station clock of the command
    operator_s: float | None = None   # when the operator saw the jaw stop, after the command


@dataclass(frozen=True)
class Stroke:
    """What one jaw move shows."""

    stream_s: float | None            # a full 0..100 stroke of the backend's stream, scaled
    effort_s: float | None            # the last time after the command the effort left its noise band
    peak: float | None
    mean: float | None
    noise: float
    delta: float | None               # units the stream moved


@dataclass(frozen=True)
class JawLogs:
    free: list[EffortLog]
    loaded: list[EffortLog]


@dataclass(frozen=True)
class JawMove:
    part: str                         # "free" | "cube"
    log: EffortLog


def _nan(v: float | None) -> float:
    return math.nan if v is None else float(v)


class JawRecorder:
    """Gripper states polled every GRIPPER_POLL_S, one row per new backend stamp."""

    def __init__(self, jaw: BackendLink | SimJaw, stopped: threading.Event) -> None:
        self.jaw = jaw
        self.stopped = stopped
        self._last_t: float | None = None

    def until(self, t_end: float) -> list[JawSample] | None:
        """The rows up to `t_end`; None when a STOP or a lost link cut them short."""
        rows: list[JawSample] = []
        while True:
            if self.stopped.is_set():
                return None
            gs = self.jaw.gripper_state()
            if gs is None:
                return None
            now = self.jaw.clock()
            if gs.t != self._last_t:
                self._last_t = gs.t
                rows.append(JawSample(t=now, effort=_nan(gs.effort), streamed=_nan(gs.streamed),
                                      commanded=_nan(gs.commanded), frozen=bool(gs.frozen)))
            if now >= t_end:
                return rows
            self.jaw.wait(GRIPPER_POLL_S)


def _froze(t: np.ndarray, frozen: np.ndarray) -> bool:
    """Whether the freeze fired during the move: `frozen` turned true after the command.

    A flag left over from the previous close, which this move's open clears, does not count.
    """
    before = frozen[t < 0.0]
    seq = np.concatenate([[bool(before[-1]) if len(before) else False], frozen[t >= 0.0]])
    return bool(np.any(seq[1:] & ~seq[:-1]))


def effort_log(rows: Sequence[JawSample], target: float, t0: float) -> EffortLog:
    t = np.array([r.t for r in rows], float) - t0
    frozen = np.array([r.frozen for r in rows], bool)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return EffortLog(
        t=t,
        effort=np.array([r.effort for r in rows], float),
        streamed=np.array([r.streamed for r in rows], float),
        commanded=np.array([r.commanded for r in rows], float),
        frozen=frozen,
        rate_hz=(len(t) - 1) / span if span > 0.0 else 0.0,
        valid=not _froze(t, frozen),
        target=float(target),
        t0=t0,
    )


def log_gripper(jaw: BackendLink | SimJaw, moves: Sequence[float], recorder: JawRecorder) -> list[EffortLog]:
    """Commands each opening in turn and samples the gripper state around it; a move cut short is dropped."""
    logs: list[EffortLog] = []
    for target in moves:
        before = recorder.until(jaw.clock() + PRE_MOVE_S)
        if before is None:
            break
        t0 = jaw.clock()
        jaw.command(target)
        after = recorder.until(t0 + GRIPPER_SETTLE_S)
        if after is None:
            break
        logs.append(effort_log([*before, *after], target, t0))
    return logs


def log_with_operator(jaw: BackendLink | SimJaw, keys: Keys, moves: Sequence[float],
                      stopped: threading.Event) -> list[EffortLog]:
    """`log_gripper`, while a reader stamps the operator's lines ("the jaw has stopped")."""
    stamps: list[float] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            text = keys.readline(timeout_s=KEYS_POLL_S)
            if text is None:
                done.wait(0.02)
            elif text:
                stamps.append(jaw.clock())

    reader = threading.Thread(target=read, name="bench-operator", daemon=True)
    reader.start()
    try:
        logs = log_gripper(jaw, moves, JawRecorder(jaw, stopped))
    finally:
        done.set()
        reader.join(1.0)
    for log in logs:
        seen = [t - log.t0 for t in stamps if OPERATOR_MIN_S <= t - log.t0 <= GRIPPER_SETTLE_S]
        if seen:
            log.operator_s = min(seen)
    return logs


def stroke_from_log(log: EffortLog, *, noise: float | None = None) -> Stroke:
    """Stream stroke, effort peak and mean, and the last effort above the noise band of one move.

    `stream_s` scales the part of the move between STROKE_EDGE_UNITS from its start and from its end
    to a full 0..100 stroke (None for a move under STROKE_MIN_UNITS). `effort_s` is the last time
    after the command at which the effort left the band of NOISE_SIGMAS sigma of the samples before
    it (or `noise` given).
    """
    t = np.asarray(log.t, float)
    post = t >= 0.0
    eff = np.asarray(log.effort, float)
    pre_eff = eff[(~post) & np.isfinite(eff)]
    centre = float(np.mean(pre_eff)) if len(pre_eff) else 0.0
    if noise is None:
        noise = NOISE_SIGMAS * float(np.std(pre_eff)) if len(pre_eff) > 1 else 0.0
        noise = max(noise, EFFORT_NOISE_MIN)

    post_eff = eff[post]
    finite = np.isfinite(post_eff)
    peak = float(np.max(np.abs(post_eff[finite]))) if finite.any() else None
    mean = float(np.mean(np.abs(post_eff[finite]))) if finite.any() else None
    tp = t[post]
    above = finite & (np.abs(post_eff - centre) > noise)
    effort_s = float(tp[above][-1]) if above.any() else None

    s = np.asarray(log.streamed, float)[post]
    ok = np.isfinite(s)
    stream_s: float | None = None
    delta: float | None = None
    if ok.sum() >= 2:
        ts = tp[ok]
        ss = s[ok]
        delta = float(ss[-1] - ss[0])
        if abs(delta) >= STROKE_MIN_UNITS:
            moved = np.abs(ss - ss[0]) > STROKE_EDGE_UNITS
            arrived = np.abs(ss - ss[-1]) <= STROKE_EDGE_UNITS
            i0 = int(np.argmax(moved))
            i1 = int(np.argmax(arrived))
            if moved.any() and arrived.any() and i1 > i0:
                stream_s = float(ts[i1] - ts[i0]) * 100.0 / (abs(delta) - 2.0 * STROKE_EDGE_UNITS)
    return Stroke(stream_s=stream_s, effort_s=effort_s, peak=peak, mean=mean, noise=float(noise), delta=delta)


def _is_stroke(stroke: Stroke, closing: bool) -> bool:
    d = stroke.delta
    if d is None:
        return False
    return d <= -STROKE_MIN_UNITS if closing else d >= STROKE_MIN_UNITS


def gripper_summary(free: Sequence[EffortLog], loaded: Sequence[EffortLog]) -> dict[str, Any]:
    """The summary from VALID moves only; a key without a valid move is left out.

    A freeze cuts a close short, and that short time would become the jaw speed, so nothing of a move
    in which the freeze fired counts.
    """
    out: dict[str, Any] = {}
    valid = [log for log in free if log.valid]
    strokes = [stroke_from_log(log) for log in valid]
    for closing in (True, False):
        name = "close" if closing else "open"
        moves = [k for k, s in enumerate(strokes) if _is_stroke(s, closing)]
        effort = [strokes[k].effort_s for k in moves if strokes[k].effort_s is not None]
        operator = [valid[k].operator_s for k in moves if valid[k].operator_s is not None]
        if effort:
            out[f"stroke_{name}_s_effort"] = round(max(effort), 4)
        if operator:
            out[f"stroke_{name}_s_operator"] = round(max(operator), 4)

    free_peaks = [s.peak for s in strokes if _is_stroke(s, True) and s.peak is not None]
    loaded_strokes = [stroke_from_log(log) for log in loaded if log.valid]
    loaded_peaks = [s.peak for s in loaded_strokes if _is_stroke(s, True) and s.peak is not None]
    if free_peaks:
        out["free_effort_peak"] = round(max(free_peaks), 4)
    if loaded_peaks:
        out["loaded_effort_peak"] = round(max(loaded_peaks), 4)
    if free_peaks and loaded_peaks:
        separable = max(loaded_peaks) > SEPARABLE_RATIO * max(free_peaks)
        out["separable"] = bool(separable)
        out["suggested_threshold"] = round(0.5 * (max(free_peaks) + max(loaded_peaks)), 4) if separable else None
    return out


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None or not math.isfinite(v) else format(v, spec)


def print_moves(moves: Sequence[JawMove]) -> None:
    print("\nmove  phase    goal   stream stroke [s]     effort peak     mean    last above noise [s]  "
          "eye [s]  valid", flush=True)
    for i, move in enumerate(moves, 1):
        s = stroke_from_log(move.log)
        valid = "yes" if move.log.valid else "NO (freeze)"
        print(f"{i:4d}  {move.part:<8s} {move.log.target:4.0f}  {_fmt(s.stream_s):>18s}  {_fmt(s.peak):>14s}  "
              f"{_fmt(s.mean):>7s}  {_fmt(s.effort_s):>22s}  {_fmt(move.log.operator_s):>7s}  {valid}", flush=True)


def npz_path(out: str | None) -> Path | None:
    """--out as the file numpy writes: `.npz` added when missing."""
    if out is None:
        return None
    path = Path(out).expanduser()
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def save_logs(path: Path, moves: Sequence[JawMove], summary: Mapping[str, Any]) -> Path:
    """The raw traces of every move and the summary (as JSON) in one .npz, for a jaw model that is not a guess."""
    arrays: dict[str, np.ndarray] = {
        "summary": np.array(json.dumps(dict(summary))),
        "part": np.array([m.part for m in moves]),
        "target": np.array([m.log.target for m in moves], float),
        "t0": np.array([m.log.t0 for m in moves], float),
        "rate_hz": np.array([m.log.rate_hz for m in moves], float),
        "valid": np.array([m.log.valid for m in moves], bool),
        "operator_s": np.array([_nan(m.log.operator_s) for m in moves], float),
    }
    for i, m in enumerate(moves):
        for key in ("t", "effort", "streamed", "commanded", "frozen"):
            arrays[f"{key}_{i}"] = np.asarray(getattr(m.log, key))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def sampling_rate(moves: Sequence[JawMove]) -> float:
    samples = sum(len(m.log.t) for m in moves)
    span = sum(float(m.log.t[-1] - m.log.t[0]) for m in moves if len(m.log.t) > 1)
    return samples / span if span > 0 else 0.0


def advise_freeze(values: Mapping[str, Any]) -> None:
    if values.get("separable") is False:
        print("The effort on the cube does not stand out from the free moves. No threshold fits, so leave "
              f"the freeze off (a1x.gripper_force: {FREEZE_OFF:g}).", flush=True)
    elif values.get("suggested_threshold") is not None:
        print(f"Suggested freeze threshold: {values['suggested_threshold']:g} (free "
              f"{values['free_effort_peak']:g}, on the cube {values['loaded_effort_peak']:g}). Set it as "
              f"a1x.gripper_force in the twin's config.", flush=True)


def report(logs: JawLogs, out: Path | None) -> int:
    """Prints the moves and the summary, saves them to `out` when given; 0 only for a complete, valid run."""
    complete = len(logs.free) == len(GRIPPER_FREE_MOVES) and len(logs.loaded) == len(GRIPPER_LOADED_MOVES)
    moves = [JawMove("free", log) for log in logs.free] + [JawMove("cube", log) for log in logs.loaded]
    if not moves:
        print("No measurement (cut short).", file=sys.stderr)
        return 1
    print_moves(moves)

    rate = sampling_rate(moves)
    print(f"Sampling: {rate:.0f} samples/s (required >= {MIN_RATE_HZ:.0f})", flush=True)
    values: dict[str, Any] = {}
    if rate < MIN_RATE_HZ:
        print(f"Gripper state sampled too slowly ({rate:.0f} < {MIN_RATE_HZ:.0f}/s): no summary.", file=sys.stderr)
    else:
        invalid = [i for i, m in enumerate(moves, 1) if not m.log.valid]
        if invalid:
            print(f"Moves {invalid}: the freeze fired, so they do not count.", flush=True)
        values = gripper_summary(logs.free, logs.loaded)
        advise_freeze(values)
        print(f"Summary: {json.dumps(values)}" if values else "No valid moves.", flush=True)
    if out is not None:
        print(f"Raw traces: {save_logs(out, moves, values)}", flush=True)
    return 0 if complete and rate >= MIN_RATE_HZ else 1


def measure_jaw(station: Station, keys: Keys) -> JawLogs:
    """The free moves, then the loaded ones once the operator has put the cube between the plates."""
    jaw = station.jaw
    loaded: list[EffortLog] = []
    with jaw.phase():
        if jaw.gripper_state() is None:
            raise Refusal("not armed", "the backend reports no gripper state")
        confirm_motion(station, keys, "Enter = measure the jaw (99, 100, 0, 100, 0, 100, then on a 3 cm cube "
                                      "0, 100); the arm stays still; Ctrl-C = quit. ")
        with operated(station, keys):
            print("Nothing between the jaws. When the jaws stop, type s and Enter (timed by eye); "
                  "a bare Enter = STOP.", flush=True)
            free = log_with_operator(jaw, keys, GRIPPER_FREE_MOVES, station.stopped)
            if len(free) == len(GRIPPER_FREE_MOVES):
                keys.note_end()
                if keys.proceed("Put a 3 cm cube between the jaws, then d and Enter; a bare Enter = STOP. "):
                    loaded = log_with_operator(jaw, keys, GRIPPER_LOADED_MOVES, station.stopped)
    return JawLogs(free=free, loaded=loaded)


@dataclass(frozen=True)
class Deps:
    """What a run opens outside this file; tests replace each with a fake."""

    open_station: Callable[[argparse.Namespace, Mapping[str, Any]], Station] = open_station
    keys: Callable[[], Keys] = Keys


def measure(a: argparse.Namespace, deps: Deps) -> int:
    require_consent(a, TOOL)
    out = npz_path(a.out)
    # The freeze would cut a free stroke short, and its time would become the jaw speed.
    settings = {**a1x_settings(a.config), "gripper_force": FREEZE_OFF}

    keys = deps.keys()
    keys.start()
    station = deps.open_station(a, settings)
    try:
        listen_for_other_senders(station)
        logs = measure_jaw(station, keys)
    finally:
        station.close()
    return report(logs, out)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[arm_options()])
    ap.add_argument("--out", default=None, help="the .npz file for the raw traces and the summary; "
                                                "without it nothing is written")
    return ap


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(lambda: measure(a, deps or Deps()))


if __name__ == "__main__":
    sys.exit(main())
