"""Top-down pinch grasps for the A1X parallel jaw.

    from planner.grasp import propose_grasps
    for g in propose_grasps(obj):        # best first
        ...                              # g.pos, g.R feed straight into Arm.ik

Geometry of the jaw, measured off `a1x.xml` in the TCP frame (x = approach,
y = closing):

    plate     x in [-0.015, +0.033], inner faces 0.0209 m apart when closed
    curled tip x in [+0.026, +0.034], inner faces 0.003 m apart when closed

So the fingertips reach 33 mm past the TCP and the plates straddle the object
around the TCP. Picking off a table therefore needs the TCP at least
33 mm + clearance above the table, which is why the handover ball (r = 30 mm,
centre 8 mm below the TCP) works out the way it does.
"""
from dataclasses import dataclass

import numpy as np

from a1x_control import rot

TIP_AHEAD = 0.033        # fingertip reach past the TCP along the approach axis
PLATE_MID = 0.0089       # plate centre, same axis
PLATE_GAP0 = 0.0209      # plate face separation with the fingers fully closed
TIP_CLEAR = 0.006        # keep the tips this far off the supporting surface


@dataclass(frozen=True)
class Grasp:
    pos: np.ndarray      # TCP position [m]
    R: np.ndarray        # TCP rotation; columns are approach, closing, third axis
    width: float         # object extent along the closing axis [m]
    score: float
    label: str

    @property
    def close_dir(self):
        return self.R[:, 1]


def _yaw_dirs(obj):
    """Candidate closing directions in the world xy plane, narrowest first."""
    R = obj.R
    if obj.round:                      # circular cross-section: yaw is free
        w = 2 * obj.half[0]
        return [(w, np.array([0.0, 1.0, 0.0])), (w, np.array([1.0, 0.0, 0.0])),
                (w, np.array([0.7071, 0.7071, 0.0])), (w, np.array([0.7071, -0.7071, 0.0]))]
    out = []
    for axis in (0, 1):                # close across local x or local y
        d = R[:, axis].copy(); d[2] = 0.0
        n = np.linalg.norm(d)
        if n < 1e-6:                   # that axis points up: not a horizontal grasp
            continue
        out.append((2 * obj.half[axis], d / n))
    return sorted(out, key=lambda t: t[0])


def propose_grasps(obj, max_opening=0.09, min_opening=0.018):
    """Rank top-down pinch grasps for one object. Returns a list, best first.

    `max_opening` is the widest object the jaw should be asked to hold; the
    plates physically reach 0.121 m but a margin keeps the closing servo in
    its useful range.
    """
    support = obj.pos[2] - obj.support_offset()          # the surface it rests on
    top = obj.pos[2] + obj.support_offset()
    z_min = support + TIP_AHEAD + TIP_CLEAR              # tips must clear the surface
    z_max = top + PLATE_MID - 0.004                      # plate centre stays under the top
    out = []
    for width, d in _yaw_dirs(obj):
        if not (min_opening <= width <= max_opening):
            continue
        for frac, tag in ((0.0, "mid"), (0.35, "high")):
            plate_c = obj.pos[2] + frac * obj.half[2]
            z = float(np.clip(plate_c + PLATE_MID, z_min, max(z_min, z_max)))
            for sign, par in ((1.0, "a"), (-1.0, "b")):
                pos = np.array([obj.pos[0], obj.pos[1], z])
                score = (max_opening - width) - 1.5 * abs(z - PLATE_MID - obj.pos[2])
                out.append(Grasp(pos, rot([0, 0, -1], sign * d), float(width),
                                 float(score), f"{obj.name}:{tag}{par}"))
    return sorted(out, key=lambda g: -g.score)


def gap_ok(gap, width, margin=0.012, closed=0.003):
    """Grasp check on the measured fingertip gap.

    An empty gripper closes to `closed`; a jaw wedged on something wider than
    the object means it caught a neighbour or the table. A round object is
    pinched by the plates rather than the tips, so the gap can come out up to
    PLATE_GAP0 - closed smaller than the object -- only the upper bound is a
    real constraint."""
    return closed + 0.0005 < gap < width + margin
