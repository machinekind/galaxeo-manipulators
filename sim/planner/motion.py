"""Collision-checked IK waypoints.

    from planner.motion import Motion
    mo = Motion(model, data, arm)
    q, why = mo.solve(q_now, pos, R, grip, allow={"obj1"}, q_from=q_now)

`solve` runs the damped-least-squares IK from `a1x_control.Arm` and then asks
MuJoCo whether that configuration puts any arm geom through the table, the dog
or an object we are not trying to touch. Contacts with the target object are
allowed -- that is what a grasp is. With `q_from` it also samples the joint-space
ramp between the two configurations, which is what the position servos actually
follow, so a waypoint whose straight-line path sweeps the arm through the table
is rejected too.

An object being carried is not in the scratch state at its carried pose, so
pass `held` and the checker rigidly re-attaches it to the TCP before colliding.

DLS on this arm falls into local minima easily (joint 2 is limited to [0, pi]
and joint 3 to [-3.32, 0], so there is only an elbow-down branch, and x = 0 is
a shoulder singularity). `solve` therefore restarts from a fan of seeds: the
previous waypoint first, so consecutive poses stay on the same branch, then the
home pose swept around joint 1 with three different elbow bends.
"""
import mujoco
import numpy as np

from a1x_control import smoothstep

ARM = "arm/"
ELBOWS = ((1.0, -1.6, 0.6), (1.9, -2.4, 0.5), (0.7, -1.0, 0.3))


class Held:
    """A rigidly grasped object: where it sits in the TCP frame."""

    def __init__(self, model, name, p_local, R_local):
        self.name = name
        self.qadr = model.jnt_qposadr[model.body(name).jntadr[0]]
        self.p_local, self.R_local = np.asarray(p_local), np.asarray(R_local)


class Motion:
    def __init__(self, model, data, arm):
        self.m, self.d, self.arm = model, data, arm
        self.scratch = mujoco.MjData(model)
        self.body_of_geom = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
                             for g in range(model.ngeom)]
        self.geom_name = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                          for g in range(model.ngeom)]

    def _pose(self, q, grip, held):
        s = self.scratch
        s.qpos[:] = self.d.qpos
        s.qvel[:] = 0.0
        s.qpos[self.arm.qadr] = q
        s.qpos[self.arm.fadr] = (grip, -grip)
        if self.m.nmocap:
            s.mocap_pos[:] = self.d.mocap_pos
            s.mocap_quat[:] = self.d.mocap_quat
        mujoco.mj_kinematics(self.m, s)
        if held is not None:
            tcp = s.site_xpos[self.arm.site]
            R = s.site_xmat[self.arm.site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (R @ held.R_local).ravel())
            s.qpos[held.qadr:held.qadr + 3] = tcp + R @ held.p_local
            s.qpos[held.qadr + 3:held.qadr + 7] = quat
            mujoco.mj_kinematics(self.m, s)
        mujoco.mj_collision(self.m, s)
        return s

    def collides(self, q, grip, allow=(), held=None):
        """Name of the first forbidden thing the arm or its payload touches."""
        s = self._pose(q, grip, held)
        carried = () if held is None else (held.name,)
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = self.body_of_geom[c.geom1], self.body_of_geom[c.geom2]
            mine = [b.startswith(ARM) or b in carried for b in (b1, b2)]
            if mine[0] == mine[1]:            # self-contact, or two things I do not control
                continue
            g = c.geom2 if mine[0] else c.geom1
            other = b2 if mine[0] else b1
            label = self.geom_name[g] or other or "world"
            if label in allow or other in allow:
                continue
            return label
        return None

    def path_clear(self, q0, q1, grip, allow=(), held=None, n=10):
        """Sample the smoothstep ramp the servos will follow between two poses."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        for s in np.linspace(0, 1, n + 1)[1:-1]:
            hit = self.collides(q0 + smoothstep(s) * (q1 - q0), grip, allow, held)
            if hit:
                return hit
        return None

    def seeds(self, q_seed, home):
        out = [np.asarray(q_seed, float)]
        for elbow in ELBOWS:
            for a in np.linspace(-np.pi, np.pi, 8, endpoint=False):
                q = np.asarray(home, float).copy()
                q[0], q[1], q[2], q[3] = a, *elbow
                out.append(np.clip(q, self.arm.lo, self.arm.hi))
        return out

    def solve(self, q_seed, pos, R, grip, allow=(), home=None, held=None, q_from=None):
        """(q, reason). `reason` is None on success, else 'ik' or the blocker's name.

        Returns the first seed that converges, lands collision free and (when
        `q_from` is given) gets there without sweeping through anything."""
        pos = np.asarray(pos, float)
        home = q_seed if home is None else home
        blocked, q = None, np.asarray(q_seed, float)
        for s in self.seeds(q_seed, home):
            q, ok = self.arm.ik(s, pos, R)
            if not ok:
                continue
            hit = self.collides(q, grip, allow, held)
            if hit is None and q_from is not None:
                hit = self.path_clear(q_from, q, grip, allow, held)
            if hit is None:
                return q, None
            blocked = hit
        return q, blocked or "ik"
