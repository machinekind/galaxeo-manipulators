"""Actual URDF transforms, mesh collision audit, and mounted-camera URDF export.

Distances and meshes are in mm inside this script; URDF mesh scale converts to m.
This is an offline geometric audit, not a motion controller or collision monitor.
"""
import argparse
import os
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
import fcl
from scipy.spatial.transform import Rotation
from design import *

URDF = REPO / 'ros2_ws/src/galaxea_a1xy_description/urdf/a1x.urdf'
MESHES = URDF.parent.parent / 'meshes'


def collision_object(mesh):
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(mesh.vertices, mesh.faces.astype(np.int32))
    model.endModel()
    return fcl.CollisionObject(model)


class Robot:
    def __init__(self):
        self.xml = ET.parse(URDF)
        self.joints = []
        for joint in self.xml.getroot().findall('joint'):
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ') * 1000
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
            limit = joint.find('limit')
            self.joints.append(dict(name=joint.get('name'), kind=joint.get('type'),
                parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
                origin=transform, axis=np.fromstring(joint.find('axis').get('xyz'), sep=' '),
                limit=None if limit is None else [float(limit.get('lower')), float(limit.get('upper'))]))
        self.limits = np.array([j['limit'] for j in self.joints[:6]])
        self.meshes = {}
        for link in self.xml.getroot().findall('link'):
            mesh = trimesh.load(MESHES / f"{link.get('name')}.STL")
            mesh.apply_scale(1000)
            self.meshes[link.get('name')] = mesh
        self.objects = {name: collision_object(m) for name, m in self.meshes.items()}
        self.names = ['base_link'] + [f'arm_link{i}' for i in range(1, 7)] + ['gripper_link']
        # Adjacent links share intended mechanical interfaces. Test every other pair.
        self.bare_pairs = [(a, b) for i, a in enumerate(self.names) for b in self.names[i+2:]]
        self.bare_pairs += [(finger, arm) for finger in ('gripper_finger_link1', 'gripper_finger_link2')
                            for arm in self.names[:-1]]

    def frames(self, q, jaw=20):
        frames = {'base_link': np.eye(4)}
        for i, joint in enumerate(self.joints):
            motion = np.eye(4)
            if joint['kind'] == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(joint['axis'] * q[i]).as_matrix()
            elif joint['kind'] == 'prismatic':
                motion[:3, 3] = joint['axis'] * (jaw if joint['name'].endswith('1') else -jaw)
            frames[joint['child']] = frames[joint['parent']] @ joint['origin'] @ motion
        to_gripper = np.linalg.inv(frames['gripper_link'])
        return {name: to_gripper @ value for name, value in frames.items()}

    def set_pose(self, q, jaw=20):
        frames = self.frames(q, jaw)
        for name, obj in self.objects.items():
            t = frames[name]
            obj.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
        return frames

    def bare_collisions(self):
        return [(a, b) for a, b in self.bare_pairs if intersects(self.objects[a], self.objects[b])]


def intersects(a, b):
    return bool(fcl.collide(a, b, fcl.CollisionRequest(num_max_contacts=1), fcl.CollisionResult()))


def camera_shape(kind):
    # Envelopes, not a claim about an unspecified camera's exact shape/connectors.
    body = box(7, 17, -16, 16, -16, 16) if kind == 'board' else box(-22.5, 22.5, -37, 37, 0, 45)
    return pose(body, kind, 60)


def accessory_shapes(kind):
    shapes = {'upper': upper(kind, 60), 'lower': clamp(60, False), 'camera': camera_shape(kind)}
    for side in (-1, 1):
        y = side * 40.3
        bolt = cylinder_z(CLAMP_X, y, -10.2, 9.8, 2).union(cylinder_z(CLAMP_X, y, 9, 9.8, 4.5)).union(cylinder_z(CLAMP_X, y, 9.8, 13.8, 3.5))
        nut = cq.Workplane('XY', origin=(CLAMP_X, y, -8.9)).polygon(6, 7/math.cos(math.pi/6)).extrude(3.2)
        shapes[f'clamp_fastener_{side}'] = bolt.union(nut)
    if kind == 'webcam':
        # Representative head/washer protrusion; use actual hardware if larger.
        shapes['tripod_head'] = pose(cylinder_z(0, 0, -10, -6, 6), kind, 60)
    else:
        for sy, sz in itertools.product((-1, 1), repeat=2):
            # Enclose the complete 26..30 mm pitch range and M2 nuts/heads.
            shapes[f'pcb_fastener_{sy}_{sz}'] = pose(box(-1.6, 10.2, sy*14-3.5, sy*14+3.5, sz*14-3.5, sz*14+3.5), kind, 60)
    return shapes


def accessory_objects(kind):
    return {name: collision_object(mesh_shape(shape)) for name, shape in accessory_shapes(kind).items()}


def mount_collisions(robot, mount):
    # Intended clamping contact with G1 is covered by verify.py, not a self-collision.
    return [(part, name) for name in robot.names[:-1] for part, obj in mount.items()
            if intersects(obj, robot.objects[name])]


def sinusoid_max(a, b, lo, hi):
    """Exact per-vertex maximum of a*cos(q)+b*sin(q) on an interval < 2*pi."""
    best = np.maximum(a*np.cos(lo)+b*np.sin(lo), a*np.cos(hi)+b*np.sin(hi))
    theta = np.arctan2(b, a)
    for k in (-1, 0, 1):
        angle = theta + k*2*np.pi
        best = np.maximum(best, np.where((angle >= lo) & (angle <= hi), np.hypot(a, b), -np.inf))
    return best


def wrist_separation_bound(robot, mounts):
    """Continuous conservative separating-plane bound, independent of wrist roll.

    Link3: sample pitch at <=0.1 deg, maximize yaw analytically for every vertex,
    then add a Lipschitz bound covering all angles between pitch samples.
    Link4: maximize yaw analytically. Link5/6: exact axial vertex bound.
    A triangle's maximum along a linear axis occurs at a vertex.
    """
    j4, j5, j6, grip = robot.joints[3:7]
    tail = j6['origin'][0, 3] + grip['origin'][0, 3]
    points = robot.meshes['arm_link3'].vertices - j4['origin'][:3, 3]
    lo, hi = j4['limit']
    samples = np.linspace(lo, hi, int(np.ceil((hi-lo)/np.radians(.1)))+1)
    b = points[:, 1] - j5['origin'][1, 3]
    maximum = -np.inf
    for angle in samples:
        a = points[:, 0]*np.cos(angle)-points[:, 2]*np.sin(angle)-j5['origin'][0, 3]
        maximum = max(maximum, float(sinusoid_max(a, b, *j5['limit']).max()-tail))
    error = np.hypot(points[:, 0], points[:, 2]).max() * (samples[1]-samples[0])/2
    bounds = {'arm_link3': float(maximum + error)}
    p = robot.meshes['arm_link4'].vertices - j5['origin'][:3, 3]
    bounds['arm_link4'] = float(sinusoid_max(p[:, 0], p[:, 1], *j5['limit']).max()-tail)
    bounds['arm_link5'] = float(robot.meshes['arm_link5'].vertices[:, 0].max()-tail)
    bounds['arm_link6'] = float(robot.meshes['arm_link6'].vertices[:, 0].max()-grip['origin'][0, 3])
    xmin = min(mesh_shape(shape).bounds[0, 0] for shapes in mounts.values() for shape in shapes.values())
    clearances = {name: round(float(xmin-x), 3) for name, x in bounds.items()}
    assert min(clearances.values()) > 3, clearances
    return dict(method='Conservative continuous axial separating-plane bound over full joints 4/5/6 limits.',
        accessory_min_x_mm=round(float(xmin), 3), arm_max_x_upper_bounds_mm=bounds,
        clearance_lower_bounds_mm=clearances, pitch_interpolation_bound_mm=float(error),
        excludes='Arm links 0/1/2 can fold into the tool; cables and unspecified camera details are not represented.')


def export_urdf(robot, absolute_paths=False):
    folder = ROOT/'collision'
    folder.mkdir(exist_ok=True)
    for kind in ('board', 'webcam'):
        root = ET.fromstring(ET.tostring(robot.xml.getroot()))
        # Store paths relative to the URDF directory; optionally resolve them for loaders
        # that require absolute filenames.
        for mesh in root.findall('.//mesh'):
            mesh_path = MESHES/Path(mesh.get('filename')).name
            mesh.set('filename', str(mesh_path) if absolute_paths else os.path.relpath(mesh_path, folder))
        # A single rigid link avoids treating intentional hardware contacts as
        # collisions between separate payload links in a planning scene.
        linkname = f'camera_{kind}_payload'
        link = ET.SubElement(root, 'link', name=linkname)
        for name, shape in accessory_shapes(kind).items():
            path = folder/f'{kind}_{name}.stl'
            cq.exporters.export(shape, str(path), tolerance=.04, angularTolerance=.08)
            for tag in ('visual', 'collision'):
                item = ET.SubElement(link, tag)
                geometry = ET.SubElement(item, 'geometry')
                ET.SubElement(geometry, 'mesh', filename=str(path) if absolute_paths else path.name, scale='0.001 0.001 0.001')
        joint = ET.SubElement(root, 'joint', name=linkname+'_fixed', type='fixed')
        ET.SubElement(joint, 'parent', link='gripper_link')
        ET.SubElement(joint, 'child', link=linkname)
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
        tree = ET.ElementTree(root)
        ET.indent(tree)
        tree.write(folder/f'a1x_{kind}_camera.urdf', encoding='utf-8', xml_declaration=True)


def audit(samples=10000, absolute_paths=False):
    robot = Robot()
    shapes = {kind: accessory_shapes(kind) for kind in ('board', 'webcam')}
    mounts = {kind: {name: collision_object(mesh_shape(shape)) for name, shape in parts.items()} for kind, parts in shapes.items()}
    bound = wrist_separation_bound(robot, shapes)
    print('Continuous wrist clearance lower bounds:', bound['clearance_lower_bounds_mm'], flush=True)
    rng = np.random.default_rng(20260919)
    counts = {kind: 0 for kind in mounts}
    examples = {kind: [] for kind in mounts}
    clear = 0
    for i, q in enumerate(rng.uniform(robot.limits[:, 0], robot.limits[:, 1], (samples, 6))):
        jaw = (0, 20, 50)[i % 3]
        robot.set_pose(q, jaw)
        if robot.bare_collisions():
            continue
        clear += 1
        for kind, mount in mounts.items():
            hits = mount_collisions(robot, mount)
            if hits:
                counts[kind] += 1
                if len(examples[kind]) < 20:
                    examples[kind].append(dict(q_degrees=np.rad2deg(q).round(5).tolist(), jaw_half_gap_mm=jaw, pairs=hits))
    report = dict(revision=4, mount_pitch_degrees=ANGLE, model=str(URDF.relative_to(REPO)), verified_housing_mm=60,
        continuous_wrist_bound=bound, sampled_full_arm_configurations=samples,
        sampled_bare_arm_clear=clear, sampled_payload_collisions_on_bare_arm_clear_poses=counts,
        collision_examples=examples,
        result='LOCAL WRIST CLEARANCE PASSES. UNRESTRICTED FULL-ARM MOTION FAILS: payload self-collision checking is required.',
        limitations='Triangle-surface tests on vendor meshes; some meshes are open. Random samples are not a complete workspace proof. Nominal camera envelopes, no cables, no physical prototype test. URDF export alone does not enable collision avoidance.')
    (ROOT/'arm_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    export_urdf(robot, absolute_paths=absolute_paths)
    print(json.dumps({k:v for k,v in report.items() if k not in ('collision_examples','continuous_wrist_bound')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-urdf', action='store_true', help='Export the two payload URDFs without rerunning the audit.')
    parser.add_argument('--absolute-mesh-paths', action='store_true', help='Resolve exported mesh filenames to this checkout.')
    parser.add_argument('--pose', type=float, nargs=6, metavar='DEG')
    parser.add_argument('--kind', choices=('board', 'webcam'), default='webcam')
    parser.add_argument('--jaw', type=float, default=20, help='Half-opening in mm, 0..50.')
    args = parser.parse_args()
    if args.export_urdf:
        export_urdf(Robot(), absolute_paths=args.absolute_mesh_paths)
    elif args.pose is None:
        audit(absolute_paths=args.absolute_mesh_paths)
    else:
        robot = Robot(); q = np.radians(args.pose)
        if np.any(q < robot.limits[:,0]) or np.any(q > robot.limits[:,1]) or not 0 <= args.jaw <= 50:
            parser.error('Pose or jaw opening lies outside the URDF limits.')
        robot.set_pose(q, args.jaw)
        bare = robot.bare_collisions(); payload = mount_collisions(robot, accessory_objects(args.kind))
        print(json.dumps(dict(bare_arm_collisions=bare, payload_collisions=payload), indent=2))
        raise SystemExit(1 if bare or payload else 0)
