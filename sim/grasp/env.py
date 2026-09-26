"""The last centimetres of a grasp, as a Gymnasium environment.

    from grasp.env import GraspEnv
    env = GraspEnv(seed=0)
    obs, info = env.reset()
    obs, r, terminated, truncated, info = env.step(action)   # action in [-1, 1]^5

The planner (or the VLA) has done its part: the arm hovers a few centimetres
above a proposed top-down pinch grasp, with the hover error real perception
leaves behind. From here the policy owns the motion until the object is
lifted: small TCP steps in x, y, z, a yaw step about the vertical and a
gripper target. A failed close is not the end of the episode -- the policy
has to open, re-aim and try again, and every attempt costs something.

Reset:
  * a scene from a small pool compiled from `pickplace_scene.build` (each with
    its own object shapes), objects re-placed at random, friction and mass
    re-drawn;
  * the target is one object, the intended grasp the best feasible one from
    `planner.grasp.propose_grasps` -- what the hover thinks it is grasping;
  * the arm is put at the hover pose: the grasp raised by `hover`, shifted
    and yawed by the noise, solved and collision-checked by the planner's
    motion solver.

Observation (privileged; the wrist-image student gets the images instead):
  joints (6), fingertip gap (1), intended grasp minus TCP (3), closing-axis
  yaw error (2), object minus TCP (3, live), object half extents (3), grasp
  width (1), gripper target (1), previous action (5), attempts (1),
  holding flag (1) -> 27.

Reward: -0.5 * |TCP - grasp| - 0.2 * |yaw error| - 0.01 per tick,
  -0.3 per close attempt, +2 the first time the gap says the object is held,
  +10 and done when it is held and lifted `lift` above where it rested,
  -2 and done when it leaves the table or is pushed more than `pushed` away.
"""
from __future__ import annotations

import math
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)

from a1x_control import CLOSED, OPEN, Arm, rot                      # noqa: E402
from pickplace_scene import OBJ_REGION, Q_HOME, TABLE_TOP, _place_objects, build, yaw_quat  # noqa: E402
from planner.grasp import gap_ok, propose_grasps                     # noqa: E402
from planner.motion import Motion                                    # noqa: E402
from planner.perception import SimPerception                         # noqa: E402

CONTROL_HZ = 20
SUBSTEPS = 25                       # 25 x 2 ms = one 50 ms tick
STEP_XYZ = 0.008                    # largest TCP step per tick [m]
STEP_YAW = math.radians(5.0)
HOVER = 0.06                        # hand-over height above the grasp [m]
NOISE = dict(xy=0.015, z=0.010, yaw=math.radians(10.0))
BOX_XY = 0.08                       # the TCP may wander this far from the hover [m]
LIFT = 0.04                         # held this high above rest = success [m]
PUSHED = 0.15                       # object this far from where it started = failure [m]
OBS_DIM = 27
ACT_DIM = 5


def _yaw_of(R):
    """Yaw of the closing axis (column 1) about the vertical."""
    d = R[:, 1]
    return math.atan2(d[1], d[0])


def _grasp_R(yaw):
    """Top-down TCP rotation whose closing axis has the given yaw."""
    return rot([0.0, 0.0, -1.0], [math.cos(yaw), math.sin(yaw), 0.0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=0, pool=8, max_steps=120, hover=HOVER, noise=NOISE,
                 step_xyz=STEP_XYZ, step_yaw=STEP_YAW, wrist=None, wrist_size=(320, 240),
                 table_size=(480, 320), rand_physics=True, composites=0.0, disturb_p=0.0,
                 force_p=0.0):
        """`composites`: share of objects that are tools / markers / L, T shapes / pucks /
        bars instead of single primitives. `disturb_p`: chance per episode that the object
        is shoved 1-2 cm while the gripper descends. `force_p`: chance per episode that the
        first close is forced early, before alignment, so the policy has to recover."""
        super().__init__()
        self.composites = float(composites)
        self.disturb_p, self.force_p = float(disturb_p), float(force_p)
        self.base_seed = int(seed)
        self.pool_size = int(pool)
        self.max_steps = int(max_steps)
        self.hover = float(hover)
        self.noise = dict(noise)
        self.step_xyz, self.step_yaw = float(step_xyz), float(step_yaw)
        self.wrist = wrist                       # None | path to a camera_spec json (adds the wrist camera)
        self.wrist_size, self.table_size = wrist_size, table_size
        self.rand_physics = rand_physics
        self.rng = np.random.default_rng(self.base_seed)
        self.pool = {}                           # index -> (model, info)
        self.renderers = {}
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.m = self.d = self.arm = self.motion = None
        self.steps = 0

    # ------------------------------------------------------------------ scenes
    def _arm_hook(self):
        if self.wrist is None:
            return None
        sys.path.insert(0, os.path.join(os.path.dirname(SIM), "hardware", "g1_camera_mounts", "sim"))
        import wrist_camera as wc
        design = wc.load_spec(camera_spec=self.wrist)
        rng = self.rng

        def hook(spec):
            wc.attach_wrist_camera(spec, hand="right", design=design, collision=False, visual=False,
                                   jitter=dict(pos_mm=10.0, rot_deg=2.0, fovy_deg=1.0), rng=rng)
        return hook

    def _scene(self, k):
        if k not in self.pool:
            model, _, info = build(self.base_seed * 1000 + k, sway=False, arm_hook=self._arm_hook(),
                                   composites=self.composites)
            self.pool[k] = (model, info)
        return self.pool[k]

    def _solve_hover(self, pos, R, allow_name):
        """Joints for the hover pose: warm-started IK from the nearest earlier
        solution, collision-checked; the planner's full seed fan only as a
        fallback. Resets were 80 % IK before this."""
        cache = getattr(self, "_q_cache", None)
        if cache is None:
            cache = self._q_cache = []
        seeds = []
        if cache:
            d = [np.linalg.norm(p - pos) for p, _ in cache]
            seeds.append(cache[int(np.argmin(d))][1])
        seeds.append(Q_HOME)
        for s0 in seeds:
            q, ok = self.arm.ik(s0, pos, R, iters=120)
            if ok and self.motion.collides(q, OPEN, {allow_name}) is None:
                break
        else:
            q, why = self.motion.solve(Q_HOME, pos, R, OPEN, {allow_name}, home=Q_HOME)
            if why:
                return q, why
        cache.append((np.asarray(pos, float).copy(), q.copy()))
        if len(cache) > 64:
            del cache[0]
        return q, None

    # ------------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(50):
            if self._try_reset():
                return self._obs(), self._info()
        raise RuntimeError("could not find a hover pose in 50 tries")

    def _try_reset(self):
        k = int(self.rng.integers(self.pool_size))
        self.m, info = self._scene(k)
        self.d = mujoco.MjData(self.m)
        self.info_scene = info
        self.arm = Arm(self.m, "arm/")
        self.motion = Motion(self.m, self.d, self.arm)
        objs = info["objects"]
        if not _place_objects(self.rng, objs):
            return False
        for o in objs:
            j = self.m.joint(o["name"]).id
            adr = self.m.jnt_qposadr[j]
            self.d.qpos[adr:adr + 3] = o["pos"]
            self.d.qpos[adr + 3:adr + 7] = yaw_quat(o["yaw"])
            self.d.qvel[self.m.jnt_dofadr[j]:self.m.jnt_dofadr[j] + 6] = 0.0
            if self.rand_physics:
                b = self.m.body(o["name"]).id
                mu = self.rng.uniform(0.6, 1.6)
                for g in range(self.m.body_geomadr[b], self.m.body_geomadr[b] + self.m.body_geomnum[b]):
                    self.m.geom_friction[g, 0] = mu
        self.d.mocap_pos[0] = info["dog"]["nominal_pos"]
        self.d.mocap_quat[0] = info["dog"]["nominal_quat"]
        self.d.qpos[self.arm.qadr] = Q_HOME
        self.d.ctrl[self.arm.acts] = Q_HOME
        self.d.ctrl[self.arm.grip] = OPEN
        mujoco.mj_forward(self.m, self.d)
        for _ in range(100):                       # let the objects settle
            mujoco.mj_step(self.m, self.d)

        per = SimPerception(self.m, self.d, parts=True)
        bodies = {}
        for o in per.objects():
            bodies.setdefault(o.name, []).append(o)
        names = list(bodies)
        self.rng.shuffle(names)
        for name in names:
            cands = [(g, part) for part in bodies[name] for g in propose_grasps(part, min_opening=0.008)]
            cands.sort(key=lambda t: -t[0].score)
            for g, obj in cands:
                dxy = self.rng.normal(0.0, self.noise["xy"], 2)
                dz = self.rng.normal(0.0, self.noise["z"])
                dyaw = self.rng.normal(0.0, self.noise["yaw"])
                yaw = _yaw_of(g.R) + dyaw
                p_hover = g.pos + np.array([dxy[0], dxy[1], self.hover + dz])
                R_hover = _grasp_R(yaw)
                q, why = self._solve_hover(p_hover, R_hover, obj.name)
                if why:
                    continue
                self.target, self.grasp = obj, g
                self.p_cmd, self.yaw_cmd, self.q_cmd = p_hover.copy(), yaw, q.copy()
                self.p0 = p_hover.copy()
                break
            else:
                continue
            break
        else:
            return False
        # put the arm there and let the servos settle
        self.d.qpos[self.arm.qadr] = self.q_cmd
        self.d.qvel[self.arm.dadr] = 0.0
        self.d.ctrl[self.arm.acts] = self.q_cmd
        self.d.ctrl[self.arm.grip] = OPEN
        mujoco.mj_forward(self.m, self.d)
        for _ in range(50):
            mujoco.mj_step(self.m, self.d)
        b = self.m.body(self.target.name).id
        self.shape = next(o["kind"] for o in objs if o["name"] == self.target.name)
        self.obj_body = b
        self.obj_geom = self.m.body_geomadr[b] + self.target.part
        self.obj_rest = self.d.geom_xpos[self.obj_geom].copy()
        self.g_cmd = 1.0                            # gripper target in [0 closed, 1 open]
        self.prev_action = np.zeros(ACT_DIM, np.float32)
        self.attempts = 0
        self.closing = False
        self.held_once = False
        self.steps = 0
        self.outcome = ""
        self._held = False
        self.disturb = bool(self.rng.uniform() < self.disturb_p)
        self.force = bool(self.rng.uniform() < self.force_p)
        self.disturbed = self.forced = False
        self.force_left = 0
        self.first_close_tick = None
        self.first_close_failed = False
        return True

    # -------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        # TCP target: small steps, boxed around the hover
        p = self.p_cmd + a[:3] * self.step_xyz
        p[:2] = np.clip(p[:2], self.p0[:2] - BOX_XY, self.p0[:2] + BOX_XY)
        p[2] = np.clip(p[2], TABLE_TOP + 0.02, self.p0[2] + 0.08)
        yaw = _wrap(self.yaw_cmd + a[3] * self.step_yaw)
        q, ok = self.arm.ik(self.q_cmd, p, _grasp_R(yaw), iters=60)
        if ok:
            self.p_cmd, self.yaw_cmd, self.q_cmd = p, yaw, q
        # gripper: continuous target, an attempt is counted when it crosses to closing
        g = float(np.clip(0.5 * (a[4] + 1.0), 0.0, 1.0))        # 1 open .. 0 closed
        tcp_now = self.d.site_xpos[self.arm.site]
        if self.force and not self.forced and self.attempts == 0 and self.steps > 2 \
                and np.linalg.norm(tcp_now[:2] - self.grasp.pos[:2]) < 0.03 \
                and tcp_now[2] > self.grasp.pos[2] + 0.035:                 # tips still above the object
            self.forced, self.force_left = True, 6                # slam the jaws shut early
        if self.force_left > 0:
            self.force_left -= 1
            g = 0.0
        if self.disturb and not self.disturbed and tcp_now[2] < self.grasp.pos[2] + 0.03:
            self.disturbed = True                                 # a shove: 1-2 cm across the table
            j = self.m.joint(self.target.name).id
            v = self.m.jnt_dofadr[j]
            ang = self.rng.uniform(0, 2 * math.pi)
            self.d.qvel[v:v + 2] = self.rng.uniform(0.25, 0.45) * np.array([math.cos(ang), math.sin(ang)])
        closing = g < 0.4
        attempt = closing and not self.closing
        self.closing = closing
        if attempt:
            self.attempts += 1
            if self.first_close_tick is None:
                self.first_close_tick = self.steps
            elif not self.held_once:
                self.first_close_failed = True                   # retrying without ever holding
        self.g_cmd = g
        self.d.ctrl[self.arm.acts] = self.q_cmd
        self.d.ctrl[self.arm.grip] = CLOSED + g * (OPEN - CLOSED)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self.steps += 1
        self.prev_action = a

        # ---- reward
        tcp, R = self.arm.tcp_pose(self.d)
        e_pos = float(np.linalg.norm(tcp - self.grasp.pos))
        e_yaw = abs(_wrap(self.yaw_cmd - _yaw_of(self.grasp.R)))
        e_yaw = min(e_yaw, abs(math.pi - e_yaw))               # a pinch is symmetric under 180 deg
        r = -0.5 * e_pos - 0.2 * e_yaw - 0.01 - (0.3 if attempt else 0.0)
        obj = self.d.geom_xpos[self.obj_geom]
        gap = self.arm.gap(self.d)
        held = closing and gap_ok(gap, self.grasp.width) and np.linalg.norm(obj - tcp) < 0.09
        if (self.first_close_tick is not None and not self.held_once
                and self.steps == self.first_close_tick + 8 and not held):
            self.first_close_failed = True
        if held and not self.held_once:
            self.held_once = True
            r += 2.0
        lifted = float(obj[2] - self.obj_rest[2])
        if held:
            r += 5.0 * max(0.0, lifted)                         # shaping toward the lift
        terminated = False
        if held and lifted > LIFT:
            r += 10.0; terminated = True; self.outcome = "success"
        elif obj[2] < TABLE_TOP - 0.05:
            r -= 2.0; terminated = True; self.outcome = "fell"
        elif np.linalg.norm(obj[:2] - self.obj_rest[:2]) > PUSHED:
            r -= 2.0; terminated = True; self.outcome = "pushed"
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            self.outcome = "timeout"
        self._held = held
        return self._obs(), float(r), terminated, truncated, self._info()

    # ------------------------------------------------------------- observation
    def _obs(self):
        tcp, R = self.arm.tcp_pose(self.d)
        q = self.d.qpos[self.arm.qadr]
        qn = 2.0 * (q - self.arm.lo) / (self.arm.hi - self.arm.lo) - 1.0
        gap = self.arm.gap(self.d)
        obj = self.d.geom_xpos[self.obj_geom]
        e_yaw = _wrap(self.yaw_cmd - _yaw_of(self.grasp.R))
        held = getattr(self, "_held", False)
        parts = [qn, [gap * 10.0], (self.grasp.pos - tcp) * 10.0,
                 [math.sin(2 * e_yaw), math.cos(2 * e_yaw)],      # 180-deg symmetric
                 (obj - tcp) * 10.0, self.target.half * 10.0, [self.grasp.width * 10.0],
                 [self.g_cmd], self.prev_action, [self.attempts / 3.0], [1.0 if held else 0.0]]
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)

    def _info(self):
        return dict(outcome=self.outcome, attempts=self.attempts, success=self.outcome == "success",
                    target=self.target.name, kind=self.shape, part=self.target.part,
                    disturbed=self.disturbed, forced=self.forced, first_close_failed=self.first_close_failed)

    # ----------------------------------------------------------------- render
    def render_camera(self, camera, size):
        key = (camera, size)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.m, size[1], size[0])
        r = self.renderers[key]
        r.update_scene(self.d, camera=camera)
        return r.render().copy()

    def render(self):
        return self.render_camera("table_cam", self.table_size)

    def render_wrist(self):
        return self.render_camera("arm/wrist", self.wrist_size)

    def close(self):
        for r in self.renderers.values():
            r.close()
        self.renderers = {}
