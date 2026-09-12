"""What the planner is allowed to know about the world.

On the real robot `objects()` comes from a detector + segmenter + pose fit and
`dog()` from an AprilTag on the dog's back. In sim both read ground truth out
of `MjData`, optionally corrupted with Gaussian noise so the planner can be
stress-tested against the error budget the real pipeline will have.

    from planner.perception import SimPerception
    per = SimPerception(model, data, pos_noise=0.003, rot_noise=0.02)
"""
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

KINDS = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
         mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}


@dataclass(frozen=True)
class ObjectObs:
    """One graspable object. `half` is the axis-aligned half-extent in the
    object's own frame, so the horizontal width seen by the gripper depends on
    the yaw in `quat`."""
    name: str
    pos: np.ndarray          # world position of the centre [m]
    quat: np.ndarray         # world orientation, wxyz
    half: np.ndarray         # half-extents in the object frame [m]
    kind: str                # box | cylinder | sphere | capsule

    @property
    def R(self):
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, np.asarray(self.quat, float))
        return m.reshape(3, 3)

    @property
    def round(self):
        """True if the horizontal cross-section is circular, so yaw is free."""
        return self.kind in ("sphere", "cylinder", "capsule")

    def support_offset(self, R=None):
        """How far the lowest point sits below the centre, for a world rotation R.

        Shape-specific: the box formula sum|R[2,i]|*half[i] overestimates a
        sphere by up to sqrt(3), which is enough to fail a place check that
        actually succeeded."""
        z = (self.R if R is None else np.asarray(R, float))[2]
        hx, _, hz = self.half
        if self.kind == "sphere":
            return float(hx)
        if self.kind == "box":
            return float(np.abs(z) @ self.half)
        horiz = float(np.hypot(z[0], z[1]))
        if self.kind == "cylinder":
            return float(hx * horiz + hz * abs(z[2]))
        return float(hx + (hz - hx) * abs(z[2]))          # capsule


@dataclass(frozen=True)
class DogObs:
    """The quadruped, as seen through the tag on its back."""
    tag_pos: np.ndarray      # world position of the tag site
    tag_quat: np.ndarray     # world orientation, wxyz
    platform_half: np.ndarray   # half-extents of the back platform, tag frame

    @property
    def R(self):
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, np.asarray(self.tag_quat, float))
        return m.reshape(3, 3)


class Perception(Protocol):
    def objects(self) -> list[ObjectObs]: ...
    def dog(self) -> DogObs: ...


class SimPerception:
    """Ground truth from MjData, with optional isotropic noise."""

    def __init__(self, model, data, prefix="obj", tag_site="dog/tag",
                 platform_geom="dog/torso", pos_noise=0.0, rot_noise=0.0, rng=None):
        self.m, self.d = model, data
        self.tag = model.site(tag_site).id
        self.platform_half = model.geom(platform_geom).size.copy()
        self.pos_noise, self.rot_noise = pos_noise, rot_noise
        self.rng = rng or np.random.default_rng(0)
        self.bodies = []
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith(prefix) and model.body_geomnum[b] == 1:
                self.bodies.append((name, b, model.body_geomadr[b]))

    def _noisy(self, pos, quat):
        if self.pos_noise:
            pos = pos + self.rng.normal(0, self.pos_noise, 3)
        if self.rot_noise:
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, self.rng.normal(0, 1, 3), self.rng.normal(0, self.rot_noise))
            out = np.zeros(4); mujoco.mju_mulQuat(out, dq, quat); quat = out
        return pos, quat

    def objects(self) -> list[ObjectObs]:
        out = []
        for name, b, g in self.bodies:
            size, kind = self.m.geom_size[g], KINDS[int(self.m.geom_type[g])]
            if kind == "box":
                half = size[:3].copy()
            elif kind == "sphere":
                half = np.full(3, size[0])
            elif kind == "cylinder":
                half = np.array([size[0], size[0], size[1]])
            else:                                   # capsule, long axis local z
                half = np.array([size[0], size[0], size[1] + size[0]])
            pos, quat = self._noisy(self.d.xpos[b].copy(), self.d.xquat[b].copy())
            out.append(ObjectObs(name, pos, quat, half, kind))
        return out

    def dog(self) -> DogObs:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.tag])
        pos, quat = self._noisy(self.d.site_xpos[self.tag].copy(), quat)
        return DogObs(pos, quat, self.platform_half.copy())
