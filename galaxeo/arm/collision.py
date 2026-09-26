"""Is the arm clear of the table, of caller-supplied obstacles and of itself, in a pose and on the way to it.

MuJoCo checks the installation model on its own MjData. Two rules:

  * Check the path, not only the goal. A clear goal can still be reached through
    the table. The joint-space segment is sampled every 0.04 rad of the largest
    joint move; ten samples per two-second move let a finger pass through a bottle.
  * Contacts present in the model's home pose are mounting contacts and allowed;
    every other pair touching the arm is a collision. No hand-kept exception list.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional, Sequence, Set, Tuple

import mujoco
import numpy as np

from .model import BASE, GRIPPER, Obstacle, Tool, _as_obstacles, build, indices

PATH_STEP_RAD = 0.04


class _Scene:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ix = indices(model)
        base = model.body(BASE).id
        in_arm = np.zeros(model.nbody, bool)
        for b in range(model.nbody):
            p = b
            while p > 0 and p != base:
                p = model.body_parentid[p]
            in_arm[b] = p == base
        self.arm_geom = in_arm[model.geom_bodyid]
        # Obstacles are world geoms and shift every geom id after them, so pairs are
        # compared by (body, n-th geom of that body), which is the same in every scene.
        self.key = [(model.body(b).name, g - int(model.body_geomadr[b]))
                    for g, b in enumerate(model.geom_bodyid)]

    def contacts(self, q: np.ndarray) -> Set[Tuple[tuple, tuple]]:
        d = self.data
        d.qpos[self.ix.qadr] = q
        d.qpos[self.ix.finger_qadr] = self.ix.finger_home
        mujoco.mj_fwdPosition(self.model, d)
        out = set()
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.arm_geom[g1] or self.arm_geom[g2]:
                out.add(tuple(sorted((self.key[g1], self.key[g2]))))
        return out

    def describe(self, pair: Tuple[tuple, tuple]) -> str:
        m = self.model
        names = []
        for body, k in pair:
            g = int(m.body_geomadr[m.body(body).id]) + k
            names.append(m.geom(g).name or body)
        return " with ".join(names)


class CollisionChecker:
    """Collision checks for one tool; obstacle sets compile their own model, cached."""

    def __init__(self, tool: Tool = GRIPPER, cache: int = 8):
        self.tool = tool
        self._cache: "OrderedDict[tuple, _Scene]" = OrderedDict()
        self._size = cache
        base = self._scene(())
        self.allowed = base.contacts(base.ix.home)

    def _scene(self, obstacles: Tuple[Obstacle, ...]) -> _Scene:
        s = self._cache.get(obstacles)
        if s is None:
            s = _Scene(build(self.tool, obstacles))
            self._cache[obstacles] = s
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(obstacles)
        return s

    def config_clear(self, q: Sequence[float], obstacles: Iterable[Obstacle] = ()) -> Optional[str]:
        """None when the pose is clear, else a description of the first collision."""
        s = self._scene(_as_obstacles(obstacles))
        bad = s.contacts(np.asarray(q, float)) - self.allowed
        return f"{s.describe(min(bad))}" if bad else None

    def path_clear(self, start: Sequence[float], goal: Sequence[float], obstacles: Iterable[Obstacle] = (),
                   step_rad: float = PATH_STEP_RAD) -> Optional[str]:
        """None when the straight joint-space move is clear all the way, else where it collides."""
        obstacles = _as_obstacles(obstacles)
        q0, q1 = np.asarray(start, float), np.asarray(goal, float)
        n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / step_rad)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            why = self.config_clear(q0 + s * (q1 - q0), obstacles)
            if why:
                return f"{why} at {s:.0%} of the way"
        return None
