"""The pick-and-place state machine.

    from planner.pickplace import PickPlace
    res = PickPlace(model, data, perception, info).run(on_step=recorder)
    print(res.success, res.stage, res.detail)

Stages, each with a named outcome so a failure is attributable:

    perceive -> plan_pick -> pick -> verify_grasp -> lift -> verify_lift
             -> travel -> place -> verify_place

Nothing is teleported or welded: the object is held by friction between the
finger plates, and every waypoint is a TCP pose put through IK, a collision
check and a sweep of the joint-space ramp that leads to it. The dog tag is
re-read immediately before the final descent because the dog sways.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from a1x_control import CLOSED, OPEN, Arm, Script, rot
from pickplace_scene import apply_sway

from .grasp import gap_ok, propose_grasps
from .motion import Held, Motion

LIFT = (0.16, 0.12, 0.09)   # how far to raise the object off the table [m], best first
PRE = 0.11               # pre-grasp standoff above the grasp [m]
PLACE_CLEAR = 0.006      # gap left under the object when releasing [m]
# Carry heights to try, above the table top. The arm cannot hold a top-down
# wrist much above base + 0.20, so the ceiling here is a kinematic limit.
CARRY_Z = (0.16, 0.12, 0.20, 0.08, 0.04)
RETREAT_Z = (0.03, 0.07, 0.12)   # short vertical hops out of the release pose


@dataclass
class Result:
    success: bool
    stage: str
    detail: str = ""
    stages: list = field(default_factory=list)
    target: str = ""
    duration: float = 0.0


class PickPlace:
    def __init__(self, model, data, perception, info, arm_prefix="arm/", verbose=False):
        self.m, self.d, self.per, self.info = model, data, perception, info
        self.arm = Arm(model, arm_prefix)
        self.motion = Motion(model, data, self.arm)
        self.script = Script(model, data, {"arm": arm_prefix})
        self.home = info["q_home"]
        self.verbose = verbose
        self.stages, self.label = [], "idle"
        self.target = self.held = None
        self.on_step = None

    # ---------------------------------------------------------------- helpers
    def _advance(self, label):
        """Step physics until the script runs out, calling back each step."""
        self.label = label
        while self.d.time < self.script.t - 1e-9:
            self.script.apply(self.d.time)
            apply_sway(self.d, self.info)
            mujoco.mj_step(self.m, self.d)
            if self.on_step:
                self.on_step(self)

    def _grip(self):
        return abs(float(self.d.qpos[self.arm.fadr[0]]))

    def _obj_state(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xquat[b].copy(), self.d.cvel[b][3:].copy()

    def _tcp(self):
        return self.arm.tcp_pose(self.d)

    def _grab(self):
        """Freeze how the object sits in the TCP frame; the grasp is rigid from here."""
        pos, quat, _ = self._obj_state(self.target.name)
        tcp, R = self._tcp()
        o_R = np.zeros(9); mujoco.mju_quat2Mat(o_R, quat)
        self.p_local = R.T @ (pos - tcp)
        self.R_local = R.T @ o_R.reshape(3, 3)
        self.held = Held(self.m, self.target.name, self.p_local, self.R_local)

    # ----------------------------------------------------------------- stages
    def _perceive(self):
        self.objs = self.per.objects()
        self.dog0 = self.per.dog()
        if not self.objs:
            return False, "no objects"
        return True, f"{len(self.objs)} object(s)"

    def _plan_pick(self):
        """Try objects nearest-first; take the first grasp that is reachable and
        collision-free at the pre-grasp, at the grasp, and along the way in."""
        base = self.info["arm_base"]
        q0, tried, why = self.script.q["arm"], 0, "none"
        for obj in sorted(self.objs, key=lambda o: np.linalg.norm(o.pos[:2] - base[:2])):
            allow = {obj.name}
            for g in propose_grasps(obj):
                tried += 1
                q_pre, why = self.motion.solve(q0, g.pos + [0, 0, PRE], g.R, OPEN, allow,
                                               home=self.home, q_from=q0)
                if why:
                    continue
                q_g, why = self.motion.solve(q_pre, g.pos, g.R, OPEN, allow,
                                             home=self.home, q_from=q_pre)
                if why:
                    continue
                self.target, self.grasp, self.q_pre, self.q_grasp = obj, g, q_pre, q_g
                return True, f"{g.label} width={g.width * 1000:.0f}mm, {tried} candidate(s)"
        return False, f"no feasible grasp in {tried} candidate(s), last: {why}"

    def _pick(self):
        self.script.move_q("arm", self.q_pre, 2.2, "pre-grasp", grip=OPEN)
        self.script.move_q("arm", self.q_grasp, 1.4, "descend")
        self.script.gripper("arm", CLOSED, 1.0, "close")
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
        tcp, _ = self._tcp()
        q0, why = self.script.q["arm"], "none"
        for dz in LIFT:
            q, why = self.motion.solve(q0, tcp + [0, 0, dz], self.grasp.R, self._grip(),
                                       {self.target.name}, home=self.home, held=self.held, q_from=q0)
            if why:
                continue
            self.lifted = dz
            self.script.move_q("arm", q, 1.5, "lift")
            self.script.wait(0.4, "lift settle")
            self._advance("lift")
            return True, ""
        return False, f"blocked by {why}"

    def _verify_lift(self):
        """The object must have come up with the gripper, not stayed on the table."""
        pos, _, _ = self._obj_state(self.target.name)
        tcp, _ = self._tcp()
        risen, held = pos[2] - self.target.pos[2], float(np.linalg.norm(pos - tcp))
        ok = bool(risen > 0.5 * self.lifted and held < 0.09)
        if ok:
            self._grab()                       # re-measure after the object settles
        return ok, f"rose {risen * 1000:.0f}mm, {held * 1000:.0f}mm from tcp"

    # ------------------------------------------------------------ place poses
    def _place_pose(self, dog, R):
        """TCP pose that sets the held object down on the platform under the tag."""
        support = self.target.support_offset(R @ self.R_local)
        goal = np.array([dog.tag_pos[0], dog.tag_pos[1],
                         dog.tag_pos[2] + support + PLACE_CLEAR])
        return goal - R @ self.p_local

    def _place_R(self, dog):
        """Closing directions to try over the dog: the one we grasped with first,
        then across and along the dog's own axes."""
        yaw = float(np.arctan2(dog.R[1, 0], dog.R[0, 0]))
        out = [rot([0, 0, -1], self.grasp.close_dir)]
        for a in np.concatenate([[yaw + np.pi / 2, yaw], yaw + np.linspace(0, np.pi, 7)[1:-1]]):
            out.append(rot([0, 0, -1], [np.cos(a), np.sin(a), 0.0]))
        return out

    def _travel(self):
        """Rise to a carry height the wrist can hold top-down, swing across at
        that height, and hold above the tag."""
        dog, top, grip = self.per.dog(), self.info["table_top"], self._grip()
        allow = {self.target.name}
        tcp, _ = self._tcp()
        q0, why = self.script.q["arm"], "none"
        for R in self._place_R(dog):
            place = self._place_pose(dog, R)
            for dz in CARRY_Z:
                z = top + dz
                if z < place[2] + 0.10:
                    continue
                q_up, why = self.motion.solve(q0, [tcp[0], tcp[1], z], R, grip, allow,
                                              home=self.home, held=self.held, q_from=q0)
                if why:
                    continue
                q_ov, why = self.motion.solve(q_up, [place[0], place[1], z], R, grip, allow,
                                              home=self.home, held=self.held, q_from=q_up)
                if why:
                    continue
                # do not commit to a wrist yaw or height we cannot land from
                _, why = self.motion.solve(q_ov, place, R, grip, allow | {"dog/torso"},
                                           home=self.home, held=self.held, q_from=q_ov)
                if why:
                    continue
                self.place_R, self.carry_z = R, z
                self.script.move_q("arm", q_up, 1.8, "carry up")
                self.script.move_q("arm", q_ov, 2.8, "over dog")
                self.script.wait(0.3, "hold")
                self._advance("travel")
                return True, ""
        return False, f"no reachable pose over the dog (last: {why})"

    def _place(self):
        dog = self.per.dog()                       # re-read the tag: the dog sways
        place = self._place_pose(dog, self.place_R)
        allow = {self.target.name, "dog/torso"}
        q0 = self.script.q["arm"]
        q, why = self.motion.solve(q0, place, self.place_R, self._grip(), allow,
                                   home=self.home, held=self.held, q_from=q0)
        if why:
            return False, f"descent blocked by {why}"
        self.script.move_q("arm", q, 1.6, "descend to platform")
        self.script.wait(0.3, "steady")
        self.script.gripper("arm", OPEN, 0.9, "release")
        self.script.wait(0.4, "let go")
        self._advance("place")
        # Back straight out in short vertical hops. A tall object often topples
        # against a jaw as it is let go, so brushing it is allowed -- but one
        # long joint-space ramp bows sideways and drags it off the platform,
        # which is exactly how the first version lost its placements.
        clear = {self.target.name, "dog/torso"}
        q_prev = q
        for i, dz in enumerate(RETREAT_Z):
            q_n, why = self.motion.solve(q_prev, place + [0, 0, dz], self.place_R, OPEN,
                                         clear, home=self.home, q_from=q_prev)
            if why:
                return False, f"retreat blocked by {why}"
            self.script.move_q("arm", q_n, 0.8, f"back off {i}")
            q_prev = q_n
        q_up, why = self.motion.solve(q_prev, [place[0], place[1], self.carry_z], self.place_R,
                                      OPEN, clear, home=self.home, q_from=q_prev)
        if why:
            return False, f"retreat blocked by {why}"
        self.script.move_q("arm", q_up, 1.2, "retreat")
        self.script.wait(1.4, "settle")
        self._advance("retreat")
        return True, ""

    def _verify_place(self):
        dog = self.per.dog()
        pos, quat, vel = self._obj_state(self.target.name)
        local = dog.R.T @ (pos - dog.tag_pos)
        o_R = np.zeros(9); mujoco.mju_quat2Mat(o_R, quat)
        support = self.target.support_offset(o_R.reshape(3, 3))
        on_pad = bool(np.all(np.abs(local[:2]) < dog.platform_half[:2]))
        resting = bool(-0.012 < local[2] - support < 0.03)
        still = bool(float(np.linalg.norm(vel)) < 0.05)
        bad = " ".join(w for w, c in (("off-pad", on_pad), ("wrong-height", resting),
                                      ("moving", still)) if not c)
        return on_pad and resting and still, (
            f"xy={np.round(local[:2], 3)} pad={np.round(dog.platform_half[:2], 3)} "
            f"dz={local[2] - support:+.3f} v={np.linalg.norm(vel):.3f}"
            + (f" -> {bad}" if bad else ""))

    # -------------------------------------------------------------------- run
    def run(self, on_step=None):
        self.on_step = on_step
        steps = [self._perceive, self._plan_pick, self._pick, self._verify_grasp,
                 self._lift, self._verify_lift, self._travel, self._place, self._verify_place]
        for fn in steps:
            name = fn.__name__[1:]
            ok, detail = fn()
            self.stages.append((name, bool(ok), detail))
            if self.verbose:
                print(f"    {'ok  ' if ok else 'FAIL'} {name:13s} {detail}")
            if not ok:
                return Result(False, name, detail, self.stages,
                              self.target.name if self.target else "", self.d.time)
        return Result(True, "done", "", self.stages,
                      self.target.name if self.target else "", self.d.time)
