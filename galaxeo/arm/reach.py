"""The safe reach box: an axis-aligned box in the base frame where every point is reachable and collision-free.

    box = ReachBox.load()             # cached next to the model, recomputed if the model or tool changed
    box.contains(p); box.clamp(p)

`compute` starts from a candidate box and verifies a 4x4x3 grid (all 8 corners
included): IK must land within 3 mm and the pose must be clear of the table and of
itself. A failing point pulls in the face it lies on by 1 cm, and the grid is checked
again. A face that had to move gets one more step, because only grid points were
checked and the pose between them was not. For the A1X gripper the candidate x from 0.18 m fails because the fingers
hit the table at the front-bottom corners (0.18, +-0.30, 0.05), so the box starts
further out.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .model import ASSETS, GRIPPER, MJCF, Tool

CACHE = os.path.join(ASSETS, "reach.json")
#: Candidate box (lo, hi) [m] the computation shrinks from.
CANDIDATE = ((0.18, -0.30, 0.05), (0.50, 0.30, 0.30))
GRID = (4, 4, 3)
IK_TOL = 0.003
STEP = 0.01
#: Lowest floor above the table [m]: table height is calibrated too, so the fingers must stay this high.
MIN_Z = 0.05


@dataclass
class ReachBox:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, float).reshape(3)
        self.hi = np.asarray(self.hi, float).reshape(3)
        if not (np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi))):
            raise ValueError("reach box: coordinates must be finite")
        if not np.all(self.lo < self.hi):
            raise ValueError(f"reach box: lo {self.lo.tolist()} must be below hi {self.hi.tolist()}")

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    def contains(self, p: Sequence[float], margin: float = 0.0) -> bool:
        """`margin` > 0 shrinks the box, < 0 grows it. Boundary counts as inside."""
        q = np.asarray(p, float).reshape(3)
        if not np.all(np.isfinite(q)):
            return False
        return bool(np.all(q >= self.lo + margin) and np.all(q <= self.hi - margin))

    def clamp(self, p: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(p, float).reshape(3), self.lo, self.hi)

    def grid(self, n: Sequence[int] = GRID) -> List[np.ndarray]:
        axes = [np.linspace(self.lo[k], self.hi[k], n[k]) for k in range(3)]
        return [np.array(p) for p in itertools.product(*axes)]

    def to_dict(self) -> dict:
        return {"lo": [round(float(v), 6) for v in self.lo], "hi": [round(float(v), 6) for v in self.hi]}

    @classmethod
    def from_dict(cls, data: dict) -> "ReachBox":
        try:
            return cls(np.asarray(data["lo"], float), np.asarray(data["hi"], float))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"reach box: expected {{'lo': [x, y, z], 'hi': [x, y, z]}}, got {data!r}") from exc

    @classmethod
    def load(cls, tool: Tool = GRIPPER, path: str = CACHE, kin=None, checker=None) -> "ReachBox":
        """The cached box for `tool`; computed and written back when missing or out of date."""
        key = cache_key(tool)
        try:
            with open(path) as f:
                entries = json.load(f)
        except (OSError, ValueError):
            entries = {}
        hit = entries.get(key)
        if hit:
            return cls.from_dict(hit)
        box, _ = compute(tool, kin=kin, checker=checker)
        entries[key] = {**box.to_dict(), "tool": {"pos": list(tool.pos), "quat": list(tool.quat)}}
        try:
            with open(path, "w") as f:
                json.dump(entries, f, indent=2, sort_keys=True)
                f.write("\n")
        except OSError:
            pass                                   # read-only install: keep it in memory
        return box


def cache_key(tool: Tool) -> str:
    """Changes whenever the model, the tool or the rules the box was computed with change."""
    h = hashlib.sha256()
    with open(MJCF, "rb") as f:
        h.update(f.read())
    h.update(json.dumps([list(tool.pos), list(tool.quat), CANDIDATE, GRID, IK_TOL, STEP, MIN_Z]).encode())
    return h.hexdigest()[:16]


def check_point(p: np.ndarray, kin, checker) -> Optional[str]:
    """None if IK reaches `p` within IK_TOL from home and the pose is collision-free, else why not."""
    sol = kin.ik(p, seed=kin.home, tol=IK_TOL)
    if not sol.ok:
        return f"IK {sol.pos_err * 1000:.1f} mm"
    return checker.config_clear(sol.q)


def compute(tool: Tool = GRIPPER, candidate=CANDIDATE, kin=None, checker=None,
            max_rounds: int = 40) -> Tuple[ReachBox, List[str]]:
    """Shrink `candidate` until every grid point passes. Returns the box and a log of what moved."""
    from .collision import CollisionChecker
    from .kinematics import Kinematics

    kin = kin or Kinematics(tool)
    checker = checker or CollisionChecker(tool)
    lo, hi = np.array(candidate[0], float), np.array(candidate[1], float)
    lo[2] = max(lo[2], MIN_Z)
    log: List[str] = []
    moved = set()
    for _ in range(max_rounds):
        box = ReachBox(lo, hi)
        bad = [(p, why) for p in box.grid() if (why := check_point(p, kin, checker))]
        if not bad:
            if not moved:
                return box, log
            for k, s in sorted(moved):
                if s == 0:
                    lo[k] += STEP
                else:
                    hi[k] -= STEP
                log.append(f"{'lo' if s == 0 else 'hi'}[{'xyz'[k]}] -> {lo[k] if s == 0 else hi[k]:.2f}: margin")
            moved = set()
            continue
        votes = {}
        for p, _why in bad:
            faces = [(k, s) for k in range(3) for s, edge in ((0, lo[k]), (1, hi[k])) if abs(p[k] - edge) < 1e-9]
            if not faces:                          # inside: pull the face it is nearest to, relative to size
                k = int(np.argmin(np.minimum(p - lo, hi - p) / (hi - lo)))
                faces = [(k, 0 if p[k] - lo[k] < hi[k] - p[k] else 1)]
            for f in faces:
                votes[f] = votes.get(f, 0) + 1
        (k, s), _ = max(votes.items(), key=lambda kv: kv[1])
        moved.add((k, s))
        if s == 0:
            lo[k] += STEP
        else:
            hi[k] -= STEP
        log.append(f"{'lo' if s == 0 else 'hi'}[{'xyz'[k]}] -> {lo[k] if s == 0 else hi[k]:.2f}: "
                   f"{bad[0][0].round(3).tolist()} {bad[0][1]}")
        if not np.all(lo < hi):
            break
    raise RuntimeError("reach box: no safe box found from the candidate: " + "; ".join(log[-3:]))
