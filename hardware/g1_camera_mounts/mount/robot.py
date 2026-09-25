"""A1X kinematics and meshes, in gripper_link millimetres.

Wraps ``ros2_ws/src/galaxea_a1xy_description`` so the gate checks can talk to
the real vendor geometry rather than to a sketch of it.  The forward
kinematics, the python-fcl collision objects and the continuous wrist
separating-plane bound are carried over from ``v4/arm_check.py``; the gripper
measurement helper is new, and is what ``params.py``'s G1 block is asserted
against.

The vendor meshes are in metres and are scaled by 1000 on load.  Some of them
(both finger meshes) are not watertight, so every geometric test here is a
triangle-surface test, never a solid-inside test.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import params as P

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
URDF = REPO / "ros2_ws/src/galaxea_a1xy_description/urdf/a1x.urdf"
MESH_DIR = URDF.parent.parent / "meshes"

# gripper_finger_joint1 origin from the URDF; joint 2 is its mirror.
FINGER_ORIGIN = np.array([36.89, 13.453, 0.12059])

LINK_NAMES = ["base_link"] + [f"arm_link{i}" for i in range(1, 7)] + ["gripper_link"]


def load_mesh(name):
    """One vendor STL, scaled to millimetres."""
    mesh = trimesh.load(MESH_DIR / f"{name}.STL")
    mesh.apply_scale(1000.0)
    return mesh


def gripper_meshes(jaw_half_opening):
    """gripper_link plus both fingers at the given half-opening, in mm."""
    out = {"gripper_link": load_mesh("gripper_link")}
    first = load_mesh("gripper_finger_link1")
    first.apply_translation(FINGER_ORIGIN + [0.0, jaw_half_opening, 0.0])
    second = load_mesh("gripper_finger_link2")
    second.apply_translation(FINGER_ORIGIN * [1, -1, -1] + [0.0, -jaw_half_opening, 0.0])
    out["gripper_finger_link1"] = first
    out["gripper_finger_link2"] = second
    return out


def triangles(meshes):
    """(N, 3, 3) triangle soup from an iterable of meshes."""
    return np.concatenate([m.triangles for m in meshes], axis=0)


def collision_object(mesh):
    """python-fcl BVH over a triangle mesh."""
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(mesh.vertices, mesh.faces.astype(np.int32))
    model.endModel()
    return fcl.CollisionObject(model)


def intersects(a, b):
    return bool(fcl.collide(a, b, fcl.CollisionRequest(num_max_contacts=1),
                            fcl.CollisionResult()))


def distance(a, b):
    result = fcl.DistanceResult()
    fcl.distance(a, b, fcl.DistanceRequest(), result)
    return float(result.min_distance)


def measure_gripper():
    """Re-derive the G1 numbers in params.py straight from the meshes.

    Returned in the same units and meaning as the ``params`` constants so
    verify.py can assert that the design was built on what is actually there.
    """
    gripper = load_mesh("gripper_link")
    vertices = gripper.vertices
    # The rail back face is the largest -X facing planar face near the step:
    # group the backward-facing triangles by X and take the one with most area.
    normals, centres = gripper.face_normals, gripper.triangles_center
    back = (normals[:, 0] < -0.99) & (centres[:, 0] > -30.0)
    planes = {}
    for x, area in zip(centres[back, 0].round(2), gripper.area_faces[back]):
        planes[x] = planes.get(x, 0.0) + float(area)
    rail_back_x = float(max(planes, key=planes.get))
    behind = vertices[vertices[:, 0] < rail_back_x - 0.5]
    housing_radius = float(np.hypot(behind[:, 1], behind[:, 2]).max())
    rail = vertices[vertices[:, 0] > rail_back_x + 0.1]   # skip the housing rim ring
    carriage_back = carriage_z = np.inf
    carriage_y = finger_front = fingertip = -np.inf
    for jaw in (0.0, 20.0, 50.0):
        fingers = [m for name, m in gripper_meshes(jaw).items() if "finger" in name]
        for mesh in fingers:
            inside = mesh.vertices[mesh.vertices[:, 0] <= P.RAIL_FRONT_X]
            carriage_back = min(carriage_back, float(inside[:, 0].min()))
            carriage_z = min(carriage_z, float(np.abs(inside[:, 2]).min()))
            carriage_y = max(carriage_y, float(np.abs(inside[:, 1]).max()))
            finger_front = max(finger_front, float(inside[:, 0].max()))
            fingertip = max(fingertip, float(mesh.vertices[:, 0].max()))
    return dict(housing_dia=round(2 * housing_radius, 3),
                housing_x0=round(float(vertices[:, 0].min()), 3),
                rail_back_x=round(rail_back_x, 3),
                rail_front_x=round(float(vertices[:, 0].max()), 3),
                rail_half_y=round(float(np.abs(rail[:, 1]).max()), 3),
                rail_half_z=round(float(np.abs(rail[:, 2]).max()), 3),
                carriage_back_x=round(carriage_back, 3),
                carriage_min_abs_z=round(carriage_z, 3),
                carriage_max_abs_y=round(carriage_y, 3),
                finger_front_x=round(finger_front, 3),
                fingertip_x=round(fingertip, 3))


class Robot:
    """URDF forward kinematics expressed in the gripper_link frame."""

    def __init__(self):
        self.xml = ET.parse(URDF)
        self.joints = []
        for joint in self.xml.getroot().findall("joint"):
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ") * 1000.0
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
            limit = joint.find("limit")
            self.joints.append(dict(
                name=joint.get("name"), kind=joint.get("type"),
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                origin=transform,
                axis=np.fromstring(joint.find("axis").get("xyz"), sep=" "),
                limit=None if limit is None
                else [float(limit.get("lower")), float(limit.get("upper"))]))
        self.limits = np.array([j["limit"] for j in self.joints[:6]])
        self.meshes = {link.get("name"): load_mesh(link.get("name"))
                       for link in self.xml.getroot().findall("link")}
        self.objects = {name: collision_object(mesh)
                        for name, mesh in self.meshes.items()}
        self.names = list(LINK_NAMES)
        # Adjacent links share an intended mechanical interface; test the rest.
        self.bare_pairs = [(a, b) for i, a in enumerate(self.names)
                           for b in self.names[i + 2:]]
        self.bare_pairs += [(finger, arm)
                            for finger in ("gripper_finger_link1", "gripper_finger_link2")
                            for arm in self.names[:-1]]

    def frames(self, q, jaw=20.0):
        """Link poses relative to gripper_link, in mm."""
        frames = {"base_link": np.eye(4)}
        for i, joint in enumerate(self.joints):
            motion = np.eye(4)
            if joint["kind"] == "revolute":
                motion[:3, :3] = Rotation.from_rotvec(joint["axis"] * q[i]).as_matrix()
            elif joint["kind"] == "prismatic":
                sign = 1.0 if joint["name"].endswith("1") else -1.0
                motion[:3, 3] = joint["axis"] * jaw * sign
            frames[joint["child"]] = (frames[joint["parent"]]
                                      @ joint["origin"] @ motion)
        to_gripper = np.linalg.inv(frames["gripper_link"])
        return {name: to_gripper @ value for name, value in frames.items()}

    def set_pose(self, q, jaw=20.0):
        frames = self.frames(q, jaw)
        for name, obj in self.objects.items():
            transform = frames[name]
            obj.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))
        return frames

    def bare_collisions(self):
        """Self-collisions of the arm alone, used to reject impossible samples."""
        return [(a, b) for a, b in self.bare_pairs
                if intersects(self.objects[a], self.objects[b])]


def _sinusoid_max(a, b, lo, hi):
    """Exact per-vertex maximum of a*cos(q) + b*sin(q) on an interval < 2*pi."""
    best = np.maximum(a * np.cos(lo) + b * np.sin(lo),
                      a * np.cos(hi) + b * np.sin(hi))
    theta = np.arctan2(b, a)
    for k in (-1, 0, 1):
        angle = theta + k * 2 * np.pi
        best = np.maximum(best, np.where((angle >= lo) & (angle <= hi),
                                         np.hypot(a, b), -np.inf))
    return best


def wrist_arm_bounds(robot):
    """Upper bound on each wrist link's X in the gripper frame, over all rolls.

    Continuous and conservative, independent of the wrist roll: link3's pitch is
    sampled at 0.1 deg and its yaw maximised analytically, with a Lipschitz term
    covering the angles between samples; link4's yaw is analytic; links 5 and 6
    are exact vertex bounds.  Carried over from v4/arm_check.py unchanged in
    method so the two revisions' numbers are comparable.
    """
    j4, j5, j6, grip = robot.joints[3:7]
    tail = j6["origin"][0, 3] + grip["origin"][0, 3]
    points = robot.meshes["arm_link3"].vertices - j4["origin"][:3, 3]
    lo, hi = j4["limit"]
    samples = np.linspace(lo, hi, int(np.ceil((hi - lo) / np.radians(0.1))) + 1)
    b = points[:, 1] - j5["origin"][1, 3]
    maximum = -np.inf
    for angle in samples:
        a = (points[:, 0] * np.cos(angle) - points[:, 2] * np.sin(angle)
             - j5["origin"][0, 3])
        maximum = max(maximum, float(_sinusoid_max(a, b, *j5["limit"]).max() - tail))
    error = float(np.hypot(points[:, 0], points[:, 2]).max()
                  * (samples[1] - samples[0]) / 2)
    bounds = {"arm_link3": maximum + error}
    p = robot.meshes["arm_link4"].vertices - j5["origin"][:3, 3]
    bounds["arm_link4"] = float(_sinusoid_max(p[:, 0], p[:, 1], *j5["limit"]).max() - tail)
    bounds["arm_link5"] = float(robot.meshes["arm_link5"].vertices[:, 0].max() - tail)
    bounds["arm_link6"] = float(robot.meshes["arm_link6"].vertices[:, 0].max()
                                - grip["origin"][0, 3])
    return bounds, error
