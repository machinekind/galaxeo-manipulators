"""The pick-and-pour state machine.

    from planner.pour import Pour
    res = Pour(model, data, perception, info).run(on_step=recorder)

Stages, each with a named outcome:

    perceive -> plan_grasp -> pick -> verify_grasp -> lift -> verify_lift
             -> plan_pour -> pour -> verify_pour -> upright -> place -> verify_place

The bottle is grasped across its neck (or body) with the approach pitched
down -- a level approach is out of the wrist's reach at bottle heights -- and
pouring tips it about a horizontal axis while steering the *mouth* to a point
above the glass rim, so the TCP swings around the mouth rather than the other
way round. Every waypoint goes through `Motion.solve` with the bottle rigidly
attached, so a swing that drags the bottle's base through the table or the
glass is rejected before it is tried.

`verify_pour` is a geometric proxy for liquid: the mouth held inside the rim
and past the tilt threshold for long enough, with the glass still standing.
A bead fluid can replace it later without touching the rest of the machine.
Placing the bottle back somewhere new is the episode's reset.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from a1x_control import OPEN, TIP_AHEAD, Arm, Script, rot
from pour_scene import GRIP_CMD, POUR_ABOVE, POUR_TILT, REGION, grasp_R

from .grasp import gap_ok
from .motion import Held, Motion

PRE = 0.10                       # pre-grasp standoff along the approach [m]
DEPTH = 0.012                    # TCP past the bottle axis along the approach [m]
LIFT = (0.06, 0.09, 0.04)        # how far to raise the bottle off the table [m]
BASE_CLEAR = 0.05                # bottle base above the table during the swing [m]
POUR_STEPS = 8                   # waypoints from upright to the pour angle
POUR_HOLD = 2.0                  # seconds at the pour angle
POUR_MIN_TILT = 1.55             # rad from upright the mouth must pass for liquid to run
POUR_MIN_SECS = 1.2              # ... and stay there this long
RETREAT_Z = (0.03, 0.07, 0.12)


def _rot_axis(axis, angle):
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    q = np.zeros(4); R = np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, ax, angle)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


@dataclass
class Result:
    success: bool
    stage: str
    detail: str = ""
    stages: list = field(default_factory=list)
    duration: float = 0.0


@dataclass
class GraspPlan:
    label: str
    pos: np.ndarray
    R: np.ndarray
    width: float
    q_pre: np.ndarray
    q_grasp: np.ndarray


class Pour:
    def __init__(self, model, data, perception, info, arm_prefix="arm/", verbose=False):
        self.m, self.d, self.per, self.info = model, data, perception, info
        self.arm = Arm(model, arm_prefix)
        self.motion = Motion(model, data, self.arm)
        self.script = Script(model, data, {"arm": arm_prefix})
        self.home = info["q_home"]
        self.top = info["table_top"]
        self.verbose = verbose
        self.stages, self.label = [], "idle"
        self.held = self.on_step = None
        self.pour_log = []                       # (t, tilt, mouth-in-rim) while pouring

    # ---------------------------------------------------------------- helpers
    def _advance(self, label):
        self.label = label
        while self.d.time < self.script.t - 1e-9:
            self.script.apply(self.d.time)
            mujoco.mj_step(self.m, self.d)
            if label == "pour":
                self._log_pour()
            if self.on_step:
                self.on_step(self)

    def _grip(self):
        return abs(float(self.d.qpos[self.arm.fadr[0]]))

    def _tcp(self):
        return self.arm.tcp_pose(self.d)

    def _body(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy(), self.d.cvel[b][3:].copy()

    def _grab(self):
        """Freeze the bottle's pose in the TCP frame; the grasp is rigid from here."""
        pos, R_o, _ = self._body(self.bottle.name)
        tcp, R = self._tcp()
        self.p_local, self.R_local = R.T @ (pos - tcp), R.T @ R_o
        self.mouth_local = R.T @ (pos + R_o[:, 2] * self.bottle.height - tcp)
        self.held = Held(self.m, self.bottle.name, self.p_local, self.R_local)
        if not hasattr(self, "pinch_local"):          # remembered from the first grab
            self.pinch_local = R_o.T @ (tcp - pos)     # TCP in the bottle frame
            self.pinch_tcp0, self.R_local0 = np.zeros(3), self.R_local.copy()

    def _mouth(self):
        pos, R_o, _ = self._body(self.bottle.name)
        return pos + R_o[:, 2] * self.bottle.height

    def _tilt(self):
        _, R_o, _ = self._body(self.bottle.name)
        return float(np.arccos(np.clip(R_o[2, 2], -1, 1)))

    def _in_hand(self):
        """(translation [m], rotation [rad]) of the bottle's pinch point relative
        to where the grasp put it, i.e. how much it has moved in the fingers."""
        pos, R_o, _ = self._body(self.bottle.name)
        tcp, R = self._tcp()
        pinch_now = R.T @ (pos + R_o @ self.pinch_local - tcp)
        dR = (R @ self.R_local0).T @ R_o
        return (float(np.linalg.norm(pinch_now - self.pinch_tcp0)),
                float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))

    def _log_pour(self):
        mouth = self._mouth()
        d_xy = float(np.linalg.norm(mouth[:2] - self.glass.rim[:2]))
        dz = float(mouth[2] - self.glass.rim[2])
        inside = d_xy < self.glass.r - 0.004 and 0.0 < dz < 0.08
        self.pour_log.append((self.d.time, self._tilt(), inside))

    # ----------------------------------------------------------------- stages
    def _perceive(self):
        self.bottle, self.glass = self.per.bottle(), self.per.glass()
        if self.bottle.R[2, 2] < 0.95:
            return False, "bottle not upright"
        return True, (f"bottle r={self.bottle.body_r * 1000:.0f}mm h={self.bottle.height * 1000:.0f}mm, "
                      f"glass r={self.glass.r * 1000:.0f}mm")

    def _grasp_candidates(self):
        """Grasps across an upright bottle, approach from the base side.

        High on the body first, TCP behind the axis so the plates take the
        widest line and the housing's top corner stays out of the shoulder;
        then the neck under the lip, TCP past the axis so the tips cage it.
        A shallow pitch is what lets a wrist roll pour past horizontal, and
        the look-ahead in `_plan_grasp` is what settles the choice."""
        b, base = self.bottle, self.info["arm_base"]
        toward = b.base[:2] - base[:2]
        ang0 = np.arctan2(toward[1], toward[0])
        places = []
        for pitch in (0.3, 0.45, 0.6):
            places.append(("body-high", b.body_h - 0.03, 2 * b.body_r, -0.02, pitch))
        for pitch in (0.55, 0.4, 0.7):
            if b.neck_h - 0.010 > TIP_AHEAD * np.sin(pitch) + 0.008:      # tips land on the neck
                places.append(("neck", b.height - 0.006 - 0.010, 2 * b.neck_r + 0.005, 0.012, pitch))
        for pitch in (0.45, 0.7):
            places.append(("body", 0.5 * b.body_h, 2 * b.body_r, -0.02, pitch))
        out = []
        for where, z_rel, width, depth, pitch in places:
            if width > 0.09:
                continue
            for dyaw in (0.0, 0.35, -0.35, 0.7, -0.7):
                a = ang0 + dyaw
                R = grasp_R((np.cos(a), np.sin(a)), pitch)
                pos = b.base + [0, 0, z_rel] + depth * np.array([np.cos(a), np.sin(a), 0.0])
                out.append((f"{where} yaw{dyaw:+.2f} pitch{pitch:.2f}", pos, R, width))
        return out

    def _housing_touches(self, q, grip):
        """True if the gripper housing (not the fingers) meets the bottle at q:
        the collision check allows contact with the target, but only the
        plates and tips are supposed to make it."""
        s = self.motion._pose(q, grip, None)
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = self.motion.body_of_geom[c.geom1], self.motion.body_of_geom[c.geom2]
            if {b1, b2} == {"arm/gripper_link", self.bottle.name}:
                return True
        return False

    def _plan_grasp(self):
        """Take the first grasp that is reachable, collision free on the way in,
        and from which a pour path exists once the bottle is lifted. The
        look-ahead uses the bottle's nominal pose in the hand; `plan_pour`
        re-solves with the measured one after the grasp."""
        q0, tried, why = self.script.q["arm"], 0, "none"
        allow = {self.bottle.name}
        b = self.bottle
        for label, pos, R, width in self._grasp_candidates():
            tried += 1
            # nothing may brush the bottle on the way to the pre-grasp: a finger
            # through the shoulder topples it before the grasp has begun
            q_pre, why = self.motion.solve(q0, pos - PRE * R[:, 0], R, OPEN, (), home=self.home, q_from=q0)
            if why:
                continue
            q_g, why = self.motion.solve(q_pre, pos, R, OPEN, allow, home=self.home, q_from=q_pre)
            if why:
                continue
            if self._housing_touches(q_g, OPEN):
                why = "housing on bottle"
                continue
            # nominal grasp: bottle upright at its perceived pose, rigid in the hand
            p_local, R_local = R.T @ (b.base - pos), R.T
            mouth_local = R.T @ (b.mouth - pos)
            held = Held(self.m, b.name, p_local, R_local)
            q_up, why = self.motion.solve(q_g, pos + [0, 0, LIFT[0]], R, width / 2, allow,
                                          home=self.home, held=held, q_from=q_g)
            if why:
                continue
            plan = self._search_pour(q_up, R, R_local, p_local, mouth_local, held, width / 2,
                                     b.mouth + [0, 0, LIFT[0]])
            if plan is None:
                why = "no pour from here"
                continue
            self.grasp = GraspPlan(label, pos, R, width, q_pre, q_g)
            self.pour_hint = plan[1]
            return True, f"{label} width={width * 1000:.0f}mm, {tried} candidate(s), pour {plan[1]}"
        return False, f"no feasible grasp in {tried} candidate(s), last: {why}"

    def _pick(self):
        self.script.move_q("arm", self.grasp.q_pre, 2.2, "pre-grasp", grip=OPEN)
        self.script.move_q("arm", self.grasp.q_grasp, 1.4, "approach")
        self.script.gripper("arm", GRIP_CMD, 1.0, "close")
        self.script.wait(0.5, "settle")
        self._advance("pick")
        return True, ""

    def _verify_grasp(self):
        gap = self.arm.gap(self.d)
        ok = gap_ok(gap, self.grasp.width)
        if ok:
            self._grab()
        return ok, f"gap={gap * 1000:.1f}mm width={self.grasp.width * 1000:.1f}mm"

    def _lift(self):
        tcp, R = self._tcp()
        q0, why = self.script.q["arm"], "none"
        for dz in LIFT:
            q, why = self.motion.solve(q0, tcp + [0, 0, dz], R, self._grip(), {self.bottle.name},
                                       home=self.home, held=self.held, q_from=q0)
            if why:
                continue
            self.lifted = dz
            self.script.move_q("arm", q, 1.5, "lift")
            self.script.wait(0.4, "lift settle")
            self._advance("lift")
            return True, ""
        return False, f"blocked by {why}"

    def _verify_lift(self):
        pos, _, _ = self._body(self.bottle.name)
        tcp, _ = self._tcp()
        risen = pos[2] - self.bottle.base[2]
        held = float(np.linalg.norm(pos + self.bottle.R[:, 2] * 0 - tcp))
        ok = bool(risen > 0.5 * self.lifted and self._tilt() < 0.15)
        if ok:
            self._grab()
        return ok, f"rose {risen * 1000:.0f}mm, tilt {np.degrees(self._tilt()):.1f}deg, {held * 1000:.0f}mm from tcp"

    # ------------------------------------------------------------------- pour
    def _pour_path(self, R_b0, R_local, p_local, mouth_local, mouth_now, axis, tilt_max):
        """Waypoints (pos, R) for the TCP that tip the *bottle* `tilt_max` about
        `axis` (world frame, through the mouth) from its orientation `R_b0`,
        while carrying the mouth from where it is to just above the glass rim.
        `R_local`, `p_local`, `mouth_local` say how the bottle currently sits
        in the hand, so re-measuring them mid-pour corrects both where the
        mouth goes and how far the bottle really tips. The mouth height is the
        larger of the target height and what keeps the base `BASE_CLEAR` above
        the table."""
        target = self.glass.rim + [0, 0, POUR_ABOVE]
        out = []
        for k in range(POUR_STEPS + 1):
            s = k / POUR_STEPS
            R = _rot_axis(axis, tilt_max * s) @ R_b0 @ R_local.T           # TCP rotation
            base_below_mouth = (R @ (p_local - mouth_local))[2]
            mouth_xy = mouth_now[:2] + (target[:2] - mouth_now[:2]) * s
            z = max(target[2], self.top + BASE_CLEAR - base_below_mouth)
            out.append((np.array([mouth_xy[0], mouth_xy[1], z]) - R @ mouth_local, R))
        return out

    def _pour_axes(self, R_now, mouth_now):
        """Candidate (label, axis, angles) to tip the bottle.

        A roll about the approach axis first: the pinch resists it with both
        contacts a neck radius apart. A lean about a horizontal axis is the
        pendulum motion a two-finger pinch barely resists, so it comes last.
        For the roll, the angle that yields each wanted tilt follows from the
        approach pitch: cos(tilt) = cos(a) + sin^2(p) (1 - cos(a))."""
        a = R_now[:, 0]
        sp2 = float(a[2] ** 2)                       # sin^2 of the approach pitch
        rolls = []
        for tilt in (POUR_TILT, POUR_TILT - 0.15, POUR_TILT + 0.15):
            c = (np.cos(tilt) - sp2) / (1 - sp2)
            if -1 < c < 1:
                rolls.append(float(np.arccos(c)))
        out = []
        if rolls:
            out += [("roll+", a, rolls), ("roll-", -a, rolls)]
        to_glass = self.glass.rim[:2] - mouth_now[:2]
        ang0 = np.arctan2(to_glass[1], to_glass[0])
        leans = (POUR_TILT, POUR_TILT - 0.2, POUR_TILT + 0.15)
        for dang in (0.0, np.pi, 0.5, -0.5, np.pi + 0.5, np.pi - 0.5):
            d = np.array([np.cos(ang0 + dang), np.sin(ang0 + dang), 0.0])
            out.append((f"lean{np.degrees(dang):+.0f}", np.cross([0, 0, 1.0], d), leans))
        return out

    def _search_pour(self, q_from, R_now, R_local, p_local, mouth_local, held, grip, mouth_now, first=None):
        """(joint waypoints, label) of the first axis/angle whose whole path solves."""
        allow = {self.bottle.name}
        axes = self._pour_axes(R_now, mouth_now)
        if first is not None:
            axes.sort(key=lambda t: t[0] != first)
        R_b0 = R_now @ R_local
        for name, axis, angles in axes:
            for ang in angles:
                # the bottle must actually pass the pour angle about this axis
                tilt = np.arccos(np.clip((_rot_axis(axis, ang) @ R_b0)[2, 2], -1, 1))
                if tilt < POUR_MIN_TILT + 0.05:
                    continue
                qs, q_prev, ok = [], q_from, True
                for pos, R in self._pour_path(R_b0, R_local, p_local, mouth_local, mouth_now, axis, ang):
                    q, why = self.motion.solve(q_prev, pos, R, grip, allow, home=self.home, held=held, q_from=q_prev)
                    if why:
                        ok = False
                        break
                    qs.append(q)
                    q_prev = q
                if ok:
                    self.pour_plan = (axis, ang, R_b0, mouth_now)
                    return qs, f"{name} {np.degrees(ang):.0f}deg -> tilt {np.degrees(tilt):.0f}deg"
        return None

    def _plan_pour(self):
        _, R_now = self._tcp()
        plan = self._search_pour(self.script.q["arm"], R_now, self.R_local, self.p_local, self.mouth_local,
                                 self.held, self._grip(), self._mouth(),
                                 first=getattr(self, "pour_hint", "").split()[0] or None)
        if plan is None:
            return False, "no reachable pour path"
        self.pour_qs, self.pour_label = plan
        return True, plan[1]

    def _resolve_rest(self, k):
        """Re-measure the bottle in the hand and re-solve waypoints k.. of the
        planned path; keep the old ones where the new solve fails."""
        self._grab()
        axis, ang, R_b0, mouth0 = self.pour_plan
        path = self._pour_path(R_b0, self.R_local, self.p_local, self.mouth_local, mouth0, axis, ang)
        q_prev = self.script.q["arm"]
        for j in range(k, len(path)):
            pos, R = path[j]
            q, why = self.motion.solve(q_prev, pos, R, self._grip(), {self.bottle.name}, home=self.home,
                                       held=self.held, q_from=q_prev)
            if why:
                break
            self.pour_qs[j] = q
            q_prev = q

    def _pour(self):
        """Tip the bottle along the planned waypoints, re-measuring how it sits
        in the hand every couple of steps: a full bottle pivots a little as it
        tilts and the mouth would otherwise drift off the glass."""
        self.script.move_q("arm", self.pour_qs[0], 2.5, "to glass")
        self._advance("pour")
        self.pour_log = []
        n = len(self.pour_qs)
        for k in range(1, n):
            if k % 2 == 1 and k > 1:
                self._resolve_rest(k)
            self.script.move_q("arm", self.pour_qs[k], 0.9, f"tip {k}")
            self._advance("pour")
        # hold, correcting every half second: a heavy bottle keeps settling in
        # the fingers, and each re-solve restores both the tilt and the mouth
        for i in range(int(POUR_HOLD / 0.5) + 1):
            self._resolve_rest(n - 1)
            self.script.move_q("arm", self.pour_qs[n - 1], 0.5, "pouring" if i else "centre mouth")
            self._advance("pour")
        return True, ""

    def _verify_pour(self):
        """Mouth inside the rim, past the tilt threshold, for long enough, and
        nothing knocked over."""
        good = [(t, tilt, inside) for t, tilt, inside in self.pour_log if inside and tilt > POUR_MIN_TILT]
        secs = (good[-1][0] - good[0][0]) if len(good) > 1 else 0.0
        max_tilt = max((tilt for _, tilt, _ in self.pour_log), default=0.0)
        g_pos, g_R, _ = self._body(self.glass.name)
        g_up = bool(g_R[2, 2] > 0.95)
        g_moved = float(np.linalg.norm(g_pos[:2] - self.info["glass_pos"][:2]))
        slip, rot = self._in_hand()
        ok = secs >= POUR_MIN_SECS and g_up and g_moved < 0.02 and slip < 0.03 and rot < 0.8
        bad = " ".join(w for w, c in (("short-pour", secs >= POUR_MIN_SECS), ("glass-down", g_up),
                                      ("glass-moved", g_moved < 0.02), ("slipped", slip < 0.03),
                                      ("pivoted", rot < 0.8)) if not c)
        return ok, (f"{secs:.1f}s in rim past {np.degrees(POUR_MIN_TILT):.0f}deg, max tilt "
                    f"{np.degrees(max_tilt):.0f}deg, glass moved {g_moved * 1000:.0f}mm, in hand "
                    f"{slip * 1000:.0f}mm/{np.degrees(rot):.0f}deg" + (f" -> {bad}" if bad else ""))

    def _upright(self):
        """Untip along the reverse path, then take out whatever pivot the bottle
        picked up in the fingers so it goes down vertical."""
        for k, q in enumerate(reversed(self.pour_qs[:-1]), 1):
            self.script.move_q("arm", q, 0.8, f"untip {k}")
        self.script.wait(0.3, "upright settle")
        self._advance("upright")
        self._grab()
        _, R_o, _ = self._body(self.bottle.name)
        tilt = self._tilt()
        if tilt > 0.04:
            axis = np.cross(R_o[:, 2], [0, 0, 1.0])
            tcp, R = self._tcp()
            q, why = self.motion.solve(self.script.q["arm"], tcp, _rot_axis(axis, tilt) @ R, self._grip(),
                                       {self.bottle.name}, home=self.home, held=self.held,
                                       q_from=self.script.q["arm"])
            if not why:
                self.script.move_q("arm", q, 1.0, "level")
                self.script.wait(0.3, "level settle")
                self._advance("upright")
                self._grab()
        return bool(self._tilt() < 0.15), f"tilt {np.degrees(tilt):.1f}deg -> {np.degrees(self._tilt()):.1f}deg"

    # ------------------------------------------------------------------ place
    def _place(self):
        """Set the bottle down somewhere new on the table: this is the reset."""
        rng = np.random.default_rng(int(self.d.time * 1000) % 2**32)
        _, R = self._tcp()
        grip = self._grip()
        allow = {self.bottle.name}
        q0, why = self.script.q["arm"], "none"
        for _ in range(40):
            xy = np.array([rng.uniform(*REGION["x"]), rng.uniform(*REGION["y"])])
            if np.linalg.norm(xy - self.glass.rim[:2]) < 0.12:
                continue
            base = np.array([xy[0], xy[1], self.top + 0.004])
            # bottle base -> TCP: base = tcp + R p_local (p_local is the body origin = base)
            tcp_goal = base - R @ self.p_local
            q_over, why = self.motion.solve(q0, tcp_goal + [0, 0, 0.06], R, grip, allow,
                                            home=self.home, held=self.held, q_from=q0)
            if why:
                continue
            q_dn, why = self.motion.solve(q_over, tcp_goal, R, grip, allow, home=self.home,
                                          held=self.held, q_from=q_over)
            if why:
                continue
            break
        else:
            return False, f"no place pose ({why})"
        self.script.move_q("arm", q_over, 2.2, "carry")
        self.script.move_q("arm", q_dn, 1.2, "set down")
        self.script.wait(0.3, "steady")
        self.script.gripper("arm", OPEN, 0.9, "release")
        self.script.wait(0.3, "let go")
        self._advance("place")
        self.held = None
        # Back out in short hops: up and a little back along the approach, or
        # straight up, whichever solves; then home if the ramp there is clear.
        clear = {self.bottle.name}
        q_prev, tcp_pos = q_dn, tcp_goal
        for i, dz in enumerate(RETREAT_Z):
            for back in (0.6, 0.0, 1.2):
                q_n, why = self.motion.solve(q_prev, tcp_pos - R[:, 0] * dz * back + [0, 0, dz], R, OPEN,
                                             clear, home=self.home, q_from=q_prev)
                if not why:
                    break
            if why:
                break
            self.script.move_q("arm", q_n, 0.8, f"back off {i}")
            q_prev = q_n
        if self.motion.path_clear(q_prev, self.home, OPEN, clear) is None:
            self.script.move_q("arm", self.home, 2.0, "home")
        else:
            up = self.motion.solve(q_prev, self.arm.tcp_pose(self.motion._pose(q_prev, OPEN, None))[0] + [0, 0, 0.1],
                                   R, OPEN, clear, home=self.home, q_from=q_prev)[0]
            self.script.move_q("arm", up, 1.2, "up")
            self.script.move_q("arm", self.home, 2.0, "home")
        self.script.wait(1.0, "settle")
        self._advance("retreat")
        return True, ""

    def _verify_place(self):
        pos, R_o, vel = self._body(self.bottle.name)
        g_pos, g_R, _ = self._body(self.glass.name)
        upright = bool(R_o[2, 2] > 0.95)
        on_table = bool(abs(pos[2] - self.top) < 0.01)
        still = bool(np.linalg.norm(vel) < 0.05)
        g_ok = bool(g_R[2, 2] > 0.95)
        bad = " ".join(w for w, c in (("toppled", upright), ("not-on-table", on_table),
                                      ("moving", still), ("glass-down", g_ok)) if not c)
        return upright and on_table and still and g_ok, (
            f"bottle at {np.round(pos[:2], 3)} dz={pos[2] - self.top:+.3f} v={np.linalg.norm(vel):.3f}"
            + (f" -> {bad}" if bad else ""))

    # -------------------------------------------------------------------- run
    def run(self, on_step=None):
        self.on_step = on_step
        steps = [self._perceive, self._plan_grasp, self._pick, self._verify_grasp, self._lift,
                 self._verify_lift, self._plan_pour, self._pour, self._verify_pour, self._upright,
                 self._place, self._verify_place]
        for fn in steps:
            name = fn.__name__[1:]
            ok, detail = fn()
            self.stages.append((name, bool(ok), detail))
            if self.verbose:
                print(f"    {'ok  ' if ok else 'FAIL'} {name:13s} {detail}", flush=True)
            if not ok:
                return Result(False, name, detail, self.stages, self.d.time)
        return Result(True, "done", "", self.stages, self.d.time)
