"""Every gate check for the wrist-camera mount.

Each function returns a plain dict of measured numbers; nothing in here decides
pass or fail.  ``verify.py`` owns the thresholds, asserts them and writes
``validation.json``.

Units are millimetres in the gripper_link frame throughout.
"""
import math

import numpy as np
import trimesh

from . import design, params as P, robot as rb


# --------------------------------------------------------------------------
# tessellation
# --------------------------------------------------------------------------

def to_mesh(solid, tolerance=0.05, angular=0.1):
    """CadQuery solid -> a cleaned trimesh."""
    vertices, faces = solid.val().tessellate(tolerance, angular)
    mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in vertices],
                           faces=faces, process=True)
    mesh.merge_vertices(digits_vertex=4)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def payload_meshes(cam=None):
    """Every solid that rides on the wrist, as trimeshes."""
    return {name: to_mesh(solid)
            for name, solid in design.payload_parts(cam).items()}


def payload_points(meshes, step=1.5, seed=P.SAMPLE_SEED):
    """Dense surface samples of the payload, for the bottle distance field.

    Seeded.  ``sample_surface_even`` draws from the global numpy RNG unless it is
    told otherwise, so this and the two other samplers below used to make every run
    of verify.py produce slightly different numbers -- which is how the README came
    to quote a corridor census (47.2 mm, 61.1 mm) that validation.json did not
    contain (46.78, 60.25).  One seed per sampler, so the file is reproducible and
    a README rendered from it can be asserted against it.
    """
    clouds = []
    for index, mesh in enumerate(meshes.values()):
        area = float(mesh.area)
        count = max(400, int(area / (step * step)))
        clouds.append(trimesh.sample.sample_surface_even(
            mesh, count, seed=seed + index)[0])
        clouds.append(mesh.vertices)
    return np.concatenate(clouds)


# --------------------------------------------------------------------------
# bottles (bottle.py ranges and pour.py grasp candidates, in mm)
# --------------------------------------------------------------------------

TIP_AHEAD = 33.0        # fingertips ahead of the TCP

# (label, z_rel(bottle), depth, pitch) exactly as planner/pour.py builds them
GRASP_CANDIDATES = [
    ("body-high", lambda b: b["body_h"] - 30.0, -20.0, 0.30),
    ("body-high", lambda b: b["body_h"] - 30.0, -20.0, 0.45),
    ("body-high", lambda b: b["body_h"] - 30.0, -20.0, 0.60),
    ("neck", lambda b: b["height"] - 16.0, 12.0, 0.55),
    ("neck", lambda b: b["height"] - 16.0, 12.0, 0.40),
    ("neck", lambda b: b["height"] - 16.0, 12.0, 0.70),
    ("body-mid", lambda b: b["body_h"] / 2.0, -20.0, 0.45),
    ("body-mid", lambda b: b["body_h"] / 2.0, -20.0, 0.70),
]


def sample_bottle(rng):
    """One bottle from bottle.sample_bottle's ranges, in millimetres."""
    bottle = dict(body_r=rng.uniform(30.0, 42.0), body_h=rng.uniform(120.0, 200.0),
                  shoulder_h=rng.uniform(20.0, 50.0), neck_r=rng.uniform(13.0, 18.0),
                  neck_h=rng.uniform(40.0, 80.0), lip=2.5)
    bottle["height"] = (bottle["body_h"] + bottle["shoulder_h"]
                        + bottle["neck_h"] + 6.0)
    return bottle


def bottle_profile(b):
    """(radius, height) polyline of the wall, bottom to top."""
    points = [(0.0, 0.0), (b["body_r"] - 4.0, 0.0), (b["body_r"], 5.0),
              (b["body_r"], b["body_h"])]
    for t in np.linspace(0.0, 1.0, 6)[1:]:
        radius = b["neck_r"] + (b["body_r"] - b["neck_r"]) * np.cos(t * np.pi / 2)
        points.append((radius, b["body_h"] + b["shoulder_h"] * np.sin(t * np.pi / 2)))
    top = b["body_h"] + b["shoulder_h"] + b["neck_h"]
    points += [(b["neck_r"], top), (b["neck_r"] + b["lip"], top + 1.0),
               (b["neck_r"] + b["lip"], top + 6.0), (0.0, top + 6.0)]
    return np.array(points)


def neck_graspable(b, pitch):
    """pour.py's gate: the fingertips have to land on the neck, not the lip."""
    return b["neck_h"] - 10.0 > TIP_AHEAD * math.sin(pitch) + 8.0


def bottle_frame(b, z_rel, depth, pitch):
    """(base point, up axis) of a held bottle in the gripper frame.

    pour_scene.grasp_R gives x = approach pitched `pitch` down, y horizontal,
    z up-and-forward, so world up in the gripper frame is (-sin p, 0, cos p)
    and horizontal forward is (cos p, 0, sin p).  The planner puts the TCP at
    axis + depth * forward, so the axis is TCP - depth * forward.
    """
    forward = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
    up = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
    axis_point = np.array([P.TCP_X, 0.0, 0.0]) - depth * forward
    return axis_point - z_rel * up, up


def _meridian(b):
    """Closed (axial, radial) polygon of the bottle wall plus its axis."""
    profile = bottle_profile(b)[:, ::-1]
    return np.vstack([profile, [0.0, 0.0]])


def _signed_distance(sd, polygon):
    """Distance from (axial, radial) points to the meridian; negative inside."""
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a
    rel = sd[:, None, :] - a[None, :, :]
    t = np.clip((rel * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-12), 0, 1)
    closest = a[None] + t[..., None] * edge[None]
    dist = np.linalg.norm(sd[:, None, :] - closest, axis=-1).min(1)
    s, d = sd[:, 0], sd[:, 1]
    inside = np.zeros(len(sd), bool)
    for i in range(len(a)):
        crosses = (a[i, 1] > d) != (b[i, 1] > d)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = a[i, 0] + (d - a[i, 1]) * (b[i, 0] - a[i, 0]) / (b[i, 1] - a[i, 1])
        inside ^= crosses & (s < x_at)
    return np.where(inside, -dist, dist)


def bottle_clearance(points, b, z_rel, depth, pitch, keep=600):
    """Smallest clearance from a payload point cloud to one held bottle.

    Exact against the surface of revolution: the points are mapped into the
    bottle's (axial, radial) plane and measured against its meridian.  A cheap
    lower bound prunes all but the nearest `keep` points, and the result is
    only trusted if the best pruned bound cannot beat it.
    """
    base, up = bottle_frame(b, z_rel, depth, pitch)
    rel = points - base
    s = rel @ up
    d = np.linalg.norm(rel - np.outer(s, up), axis=1)
    top = b["height"] + 6.0
    bound = np.maximum(d - b["body_r"], np.maximum(0.0, np.maximum(-s, s - top)))
    polygon = _meridian(b)
    if keep >= len(bound):
        return float(_signed_distance(np.stack([s, d], 1), polygon).min())
    order = np.argsort(bound)
    near = order[:keep]
    best = float(_signed_distance(np.stack([s[near], d[near]], 1), polygon).min())
    if bound[order[keep]] >= best:
        return best
    return float(_signed_distance(np.stack([s, d], 1), polygon).min())


BOTTLE_RANGES = dict(body_r=(30.0, 42.0), body_h=(120.0, 200.0),
                     shoulder_h=(20.0, 50.0), neck_r=(13.0, 18.0),
                     neck_h=(40.0, 80.0))


def corner_bottles():
    """Every corner of the bottle parameter box.

    Random sampling never lands exactly on the extremes, and the binding case
    here is an all-maximum bottle, so the 32 corners are checked as well.
    """
    keys = list(BOTTLE_RANGES)
    out = []
    for mask in range(1 << len(keys)):
        bottle = {key: BOTTLE_RANGES[key][(mask >> i) & 1]
                  for i, key in enumerate(keys)}
        bottle["lip"] = 2.5
        bottle["height"] = (bottle["body_h"] + bottle["shoulder_h"]
                            + bottle["neck_h"] + 6.0)
        out.append(bottle)
    return out


def g1_bottle_in_hand(meshes, samples=P.G1_BOTTLE_SAMPLES, seed=P.BOTTLE_SEED):
    """G1: the payload against every planner grasp of every sampled bottle."""
    points = payload_points(meshes)
    rng = np.random.default_rng(seed)
    bottles = [sample_bottle(rng) for _ in range(samples)] + corner_bottles()
    per_candidate = {}
    worst, collisions, checks = np.inf, 0, 0
    for label, z_rel, depth, pitch in GRASP_CANDIDATES:
        values = []
        for b in bottles:
            if label == "neck" and not neck_graspable(b, pitch):
                continue
            values.append(bottle_clearance(points, b, z_rel(b), depth, pitch))
        values = np.array(values)
        per_candidate[f"{label} pitch{pitch:.2f}"] = dict(
            n=len(values), min_mm=round(float(values.min()), 2),
            mean_mm=round(float(values.mean()), 2),
            collisions=int((values < 0).sum()),
            under_gate=int((values < P.G1_BOTTLE_CLEARANCE).sum()))
        worst = min(worst, float(values.min()))
        collisions += int((values < 0).sum())
        checks += len(values)
    return dict(bottles=samples, parameter_box_corners=len(corner_bottles()),
                checks=checks, payload_points=len(points),
                min_clearance_mm=round(worst, 2), collisions=collisions,
                per_candidate=per_candidate)


# --------------------------------------------------------------------------
# G1: real gripper / finger meshes, and the wrist bound on arm links 3-6
# --------------------------------------------------------------------------

def _non_locating_points(meshes, step=1.0, seed=P.SAMPLE_SEED + 100):
    """Dense samples of the two brackets, minus the surfaces meant to touch the G1.

    The exemption G1 grants is "clamp bore contact excepted".  Keying that off the
    part *name* -- which is what the first cut of ``g1_gripper_meshes`` did -- lets
    the strut, the camera plate, the tie lug and the bolt ears off as well, and
    those are not supposed to be anywhere near the housing.  So the exemption is a
    region test instead: see ``design.locating_region``.
    """
    clouds = []
    for index, name in enumerate(design.PRINTED_PARTS):
        mesh = meshes[name]
        count = max(4000, int(mesh.area / (step * step)))
        points = np.vstack([trimesh.sample.sample_surface_even(
            mesh, count, seed=seed + index)[0], mesh.vertices])
        clouds.append(points[~design.locating_region(points)])
    return np.concatenate(clouds)


def g1_gripper_meshes(meshes, jaws=(0.0, 20.0, 50.0)):
    """Payload against the real gripper and finger meshes at three openings.

    The clamp bore, the stop pads and the key ears are *supposed* to run close to
    the gripper, so those surfaces are measured separately -- and by region, not by
    part name, so the strut and the plate are held to the full 4 mm against
    ``gripper_link`` like everything else.
    """
    payload = {name: rb.collision_object(mesh) for name, mesh in meshes.items()}
    free_points = _non_locating_points(meshes)
    rows = {}
    worst_free, worst_locating = np.inf, np.inf
    free_at, bracket_gap = None, np.inf
    for jaw in jaws:
        parts = {name: rb.collision_object(mesh)
                 for name, mesh in rb.gripper_meshes(jaw).items()}
        for link, link_obj in parts.items():
            for name, obj in payload.items():
                locating = (name in design.PRINTED_PARTS and link == "gripper_link")
                gap = rb.distance(obj, link_obj)
                rows[f"jaw{jaw:.0f}:{name}-{link}"] = round(gap, 2)
                if locating:
                    worst_locating = min(worst_locating, gap)
                else:
                    worst_free = min(worst_free, gap)
        # The bracket-vs-gripper_link pairs the loop above had to exempt, redone on
        # the non-locating sample points only.  Point-to-surface, so it is a lower
        # bound on the surface-to-surface distance, which is the safe direction.
        # Unsigned point-to-surface, not signed_distance: gripper_link.STL is not
        # watertight (the vendor meshes are open surfaces), and a signed query on an
        # open surface is only accidentally right.  The two agree here to 1e-3, but
        # only one of them is defined.
        query = trimesh.proximity.ProximityQuery(
            rb.gripper_meshes(jaw)["gripper_link"])
        gaps = query.on_surface(free_points)[1]
        gap = float(gaps.min())
        rows[f"jaw{jaw:.0f}:non-locating bracket-gripper_link"] = round(gap, 2)
        if gap < bracket_gap:
            bracket_gap = gap
            free_at = [round(v, 1) for v in free_points[int(np.argmin(gaps))]]
        worst_free = min(worst_free, gap)
    tightest = dict(sorted(rows.items(), key=lambda kv: kv[1])[:10])
    return dict(min_free_clearance_mm=round(worst_free, 2),
                min_locating_clearance_mm=round(worst_locating, 2),
                non_locating_sample_points=len(free_points),
                non_locating_bracket_clearance_mm=round(bracket_gap, 2),
                tightest_non_locating_point_mm=free_at,
                tightest=tightest,
                note="'locating' = the clamp bore annulus, the stop-pad plane and "
                     "the key ears' inner faces, by region (design.locating_region), "
                     "against gripper_link: contact there is the point of the part. "
                     "Everything else on the brackets is held to the full gate, "
                     "including against gripper_link.")


def bottle_corridor_census(meshes, tolerance=0.1, seed=P.SAMPLE_SEED + 200):
    """What the payload actually has forward of KEEPOUT_X, outside KEEPOUT_R.

    Reported, not gated -- and the report replaces a README claim rather than
    implementing one.  "This bracket never puts anything above the housing forward
    of X = -24 mm" was not true of any revision of it and cannot be: the locating
    yoke's whole job is to reach the rail's back face at X = -15.65, and the lens
    barrel points forward and inboard.  What is true is that nothing sits in the
    corridor a bottle *sweeps*, and the number for that is G1's measured clearance
    against 352 bottles x 8 grasps, not a plane in X.

    So: the census says which parts are forward of the plane, how far, and how close
    to the bottle slab, and G1 is the gate.
    """
    rows, worst_y, count, worst_z = {}, np.inf, 0, -np.inf
    for index, (name, mesh) in enumerate(meshes.items()):
        points = np.vstack([trimesh.sample.sample_surface_even(
            mesh, max(2000, int(mesh.area)), seed=seed + index)[0], mesh.vertices])
        radius = np.hypot(points[:, 1], points[:, 2])
        forward = ((points[:, 0] > P.KEEPOUT_X + tolerance)
                   & (radius > P.KEEPOUT_R))
        if not forward.any():
            continue
        ys = np.abs(points[forward, 1])
        zs = points[forward, 2]
        count += int(forward.sum())
        rows[name] = dict(points=int(forward.sum()),
                          max_x_mm=round(float(points[forward, 0].max()), 2),
                          min_abs_y_mm=round(float(ys.min()), 2),
                          abs_z_range_mm=[round(float(np.abs(zs).min()), 2),
                                          round(float(np.abs(zs).max()), 2)])
        worst_y = min(worst_y, float(ys.min()))
        worst_z = max(worst_z, float(np.abs(zs).max()))
    return dict(keepout_x_mm=P.KEEPOUT_X, keepout_radius_mm=P.KEEPOUT_R,
                bottle_slab_y_mm=P.BOTTLE_SLAB_Y,
                parts_forward_of_keepout=rows, points=count,
                min_abs_y_forward_mm=None if count == 0 else round(worst_y, 2),
                max_abs_z_forward_mm=None if count == 0 else round(worst_z, 2),
                gated_by="G1_bottle_in_hand.min_clearance_mm")


def g1_locating_features(meshes):
    """The locating features against the finger meshes only, spelled out.

    These are the numbers the design is really betting on: the stop pads and
    the key ears live inside the rail's neighbourhood, where the carriages run.
    """
    locating = {name: rb.collision_object(meshes[name])
                for name in design.PRINTED_PARTS}
    rows = {}
    worst = np.inf
    for jaw in (0.0, 10.0, 20.0, 35.0, 50.0):
        fingers = {name: rb.collision_object(mesh)
                   for name, mesh in rb.gripper_meshes(jaw).items()
                   if "finger" in name}
        for finger, finger_obj in fingers.items():
            for name, obj in locating.items():
                gap = rb.distance(obj, finger_obj)
                rows[f"jaw{jaw:.0f}:{name}-{finger}"] = round(gap, 2)
                worst = min(worst, gap)
    return dict(min_clearance_mm=round(worst, 2), per_pair=rows)


def g1_arm_links(meshes, robot=None):
    """G1: the payload behind every reachable position of arm links 3-6."""
    robot = robot or rb.Robot()
    bounds, lipschitz = rb.wrist_arm_bounds(robot)
    payload_min_x = float(min(m.vertices[:, 0].min() for m in meshes.values()))
    clearance = {name: round(payload_min_x - value, 2)
                 for name, value in bounds.items()}
    return dict(payload_min_x_mm=round(payload_min_x, 2),
                arm_max_x_upper_bound_mm={k: round(v, 3) for k, v in bounds.items()},
                clearance_mm=clearance, min_mm=round(min(clearance.values()), 2),
                pitch_interpolation_bound_mm=round(lipschitz, 4),
                method="continuous separating-plane bound over the full joint "
                       "4/5/6 ranges, independent of wrist roll (v4 method)",
                excludes="arm links 0-2 can fold into the tool; they are only "
                         "covered by the random full-arm audit")


# --------------------------------------------------------------------------
# G2': sightlines
# --------------------------------------------------------------------------

def blocked(origin, targets, tris):
    """Moller-Trumbore segment/triangle tests, two-sided (v4 method)."""
    a = tris[:, 0]
    e1, e2 = tris[:, 1] - a, tris[:, 2] - a
    tvec = origin - a
    qvec = np.cross(tvec, e1)
    dot2 = np.einsum("ij,ij->i", e2, qvec)
    out = []
    for target in np.atleast_2d(targets):
        ray = target - origin
        pvec = np.cross(np.broadcast_to(ray, e2.shape), e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=abs(det) > 1e-10)
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        v = np.einsum("j,ij->i", ray, qvec) * inv
        t = dot2 * inv
        hit = ((abs(det) > 1e-10) & (u >= -1e-8) & (v >= -1e-8)
               & (u + v <= 1 + 1e-8) & (t > 1e-5) & (t < 1 - 1e-5))
        out.append(bool(hit.any()))
    return np.array(out)


def blade_tip_points(jaw, side):
    """The camera-side top-edge tip point of each finger blade.

    The blade nose is rounded, so "the tip" is taken as the forwardmost-highest
    corner on the camera side: the vertex maximising x + side*z.  It is then
    lifted 0.3 mm along that same diagonal, off the surface, so the test asks
    "is the tip visible" rather than "does a ray land exactly on a triangle".
    """
    diagonal = np.array([1.0, 0.0, side]) / math.sqrt(2.0)
    points = {}
    for name, mesh in rb.gripper_meshes(jaw).items():
        if "finger" not in name:
            continue
        vertices = mesh.vertices
        tip = vertices[int(np.argmax(vertices @ diagonal))]
        points[name] = tip + 0.3 * diagonal
    return points


def _grasp_grid(jaw, z, count=5):
    xs = np.linspace(60.0, 76.0, count)
    return np.array([[x, sign * (jaw - 2.0), z] for x in xs for sign in (1, -1)])


def _far_jaw_points(jaw, side_y, count=5):
    """Samples on the inner pad face of the jaw away from the camera."""
    xs = np.linspace(60.0, 76.0, count)
    zs = np.linspace(-6.0, 6.0, 5)
    y = -side_y * jaw + side_y * 0.25
    return np.array([[x, y, z] for x in xs for z in zs])


def g2_sightlines(cam, meshes):
    """G2' (a)-(d): what the lens can actually see, on the real meshes."""
    eye = cam.pupil
    camera_side_z = 1.0 if cam.pos[2] > 0 else -1.0
    mount = rb.triangles([meshes[name] for name in design.PRINTED_PARTS])
    result = {"pupil_mm": [round(v, 2) for v in eye],
              "object_zone_z_mm": camera_side_z * P.G2_OBJECT_ZONE_Z}
    tips, zone, far, grasp = {}, {}, {}, {}
    for opening in P.JAW_OPENINGS:
        jaw = opening / 2.0
        scene = np.vstack([rb.triangles(rb.gripper_meshes(jaw).values()), mount])
        # (a) both blade tips
        tip_points = blade_tip_points(jaw, camera_side_z)
        hit = blocked(eye, np.array(list(tip_points.values())), scene)
        tips[f"opening{opening:.0f}"] = {
            name: dict(visible=bool(not flag),
                       point_mm=[round(v, 2) for v in tip_points[name]])
            for name, flag in zip(tip_points, hit)}
        # (b) object zone just above the blades
        targets = _grasp_grid(jaw, camera_side_z * P.G2_OBJECT_ZONE_Z)
        hit = blocked(eye, targets, scene)
        zone[f"opening{opening:.0f}"] = dict(n=len(targets), blocked=int(hit.sum()))
        # (c) inner pad face of the far jaw
        targets = _far_jaw_points(jaw, np.sign(cam.pos[1]))
        hit = blocked(eye, targets, scene)
        far[f"opening{opening:.0f}"] = dict(
            n=len(targets), visible_fraction=round(1.0 - float(hit.mean()), 3))
        # (d) the old G2 target set, on Z = 0 between the jaws
        xs = np.linspace(60.0, 76.0, 5)
        ys = np.linspace(-(jaw - 2.0), jaw - 2.0, 9)
        targets = np.array([[x, y, 0.0] for x in xs for y in ys])
        hit = blocked(eye, targets, scene)
        grasp[f"opening{opening:.0f}"] = dict(
            n=len(targets), blocked=int(hit.sum()),
            visible_fraction=round(1.0 - float(hit.mean()), 3))
    result.update(a_blade_tips=tips, b_object_zone=zone, c_far_jaw=far,
                  d_grasp_zone_z0=grasp)
    result["blocked_by_mount_alone"] = int(sum(
        blocked(eye, _grasp_grid(o / 2.0, camera_side_z * P.G2_OBJECT_ZONE_Z),
                mount).sum() for o in P.JAW_OPENINGS))
    result["note"] = ("the near blade hides a sliver of the Z=0 gap directly "
                      "behind itself; that is the (d) shortfall and it is "
                      "geometric, not a mount shadow")
    return result


# --------------------------------------------------------------------------
# G3: framing
# --------------------------------------------------------------------------

def view_angles(cam, point):
    """(horizontal, vertical) angle of a point off the optical axis, degrees."""
    rel = np.asarray(point, float) - cam.pupil
    forward = rel @ cam.axis
    if forward <= 1e-6:
        return 180.0, 180.0
    return (math.degrees(math.atan2(rel @ cam.right, forward)),
            math.degrees(math.atan2(rel @ cam.up, forward)))


def in_view(cam, points, fov):
    """Boolean per point: inside the rectangular frustum."""
    rel = np.atleast_2d(points) - cam.pupil
    forward = rel @ cam.axis
    tan_h = math.tan(math.radians(fov[0] / 2.0))
    tan_v = math.tan(math.radians(fov[1] / 2.0))
    return ((forward > 1e-6)
            & (np.abs(rel @ cam.right) <= tan_h * forward)
            & (np.abs(rel @ cam.up) <= tan_v * forward))


def g3_forward_view(cam, fov):
    """G3: fingertips and a point 250 mm past them, both in frame."""
    targets = {
        "tool axis at the fingertips": [P.FINGERTIP_X, 0.0, 0.0],
        "near fingertip (jaw 40)": [P.FINGERTIP_X, 20.0, 0.0],
        "far fingertip (jaw 40)": [P.FINGERTIP_X, -20.0, 0.0],
        "near fingertip (jaw 100)": [P.FINGERTIP_X, 50.0, 0.0],
        "far fingertip (jaw 100)": [P.FINGERTIP_X, -50.0, 0.0],
        f"{P.G3_FORWARD_MM:.0f} mm past the tips":
            [P.FINGERTIP_X + P.G3_FORWARD_MM, 0.0, 0.0],
        "grasp zone centre": [68.0, 0.0, 0.0],
    }
    # What G3 actually asks for: the fingertips at a working opening, and the point
    # 250 mm ahead.  The jaw-100 fingertips are 100 mm apart and are reported, not
    # gated -- quoting their margin as "the G3 margin" understates it by 6 deg.
    gated = ("tool axis at the fingertips", "near fingertip (jaw 40)",
             "far fingertip (jaw 40)", f"{P.G3_FORWARD_MM:.0f} mm past the tips")
    rows, margin, gated_margin = {}, np.inf, np.inf
    for name, point in targets.items():
        h, v = view_angles(cam, point)
        inside = abs(h) <= fov[0] / 2 and abs(v) <= fov[1] / 2
        rows[name] = dict(h_deg=round(h, 1), v_deg=round(v, 1), in_frame=bool(inside))
        margin = min(margin, fov[0] / 2 - abs(h), fov[1] / 2 - abs(v))
        if name in gated:
            gated_margin = min(gated_margin, fov[0] / 2 - abs(h),
                               fov[1] / 2 - abs(v))
    axis_pitch = math.degrees(math.asin(max(-1.0, min(1.0, -cam.axis[2]))))
    off_axis = math.degrees(math.acos(max(-1.0, min(1.0, cam.axis[0]))))
    return dict(fov_deg=list(fov), pitch_deg=round(axis_pitch, 2),
                total_off_axis_deg=round(off_axis, 2), yaw_deg=cam.yaw,
                worst_margin_deg=round(float(margin), 2),
                gated_margin_deg=round(float(gated_margin), 2),
                gated_targets=list(gated), targets=rows,
                forward_point_in_frame=rows[f"{P.G3_FORWARD_MM:.0f} mm past the tips"]["in_frame"],
                fingertips_in_frame=all(
                    rows[k]["in_frame"] for k in
                    ("tool axis at the fingertips", "near fingertip (jaw 40)",
                     "far fingertip (jaw 40)")))


def g3_occlusion(cam, fov, bottles=16, grid=40, seed=7):
    """Report: how much of the image a held bottle covers, and is its mouth in.

    Rays are cast on an image grid and intersected with the bottle's surface of
    revolution.  Not a gate -- it is what tells us whether this view can close
    a pour loop.
    """
    rng = np.random.default_rng(seed)
    shapes = [sample_bottle(rng) for _ in range(bottles)]
    tan_h = math.tan(math.radians(fov[0] / 2.0))
    tan_v = math.tan(math.radians(fov[1] / 2.0))
    u = tan_h * np.linspace(-1, 1, int(grid * fov[0] / fov[1]))
    v = tan_v * np.linspace(-1, 1, grid)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    directions = (cam.axis + uu.ravel()[:, None] * cam.right
                  + vv.ravel()[:, None] * cam.up)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    depths = np.arange(25.0, 620.0, 4.0)
    samples = (cam.pupil + directions[:, None, :] * depths[None, :, None]).reshape(-1, 3)
    out = {}
    for label, z_rel, depth, pitch in GRASP_CANDIDATES:
        fractions, mouth_seen, neck_seen = [], [], []
        for b in shapes:
            if label == "neck" and not neck_graspable(b, pitch):
                continue
            base, up = bottle_frame(b, z_rel(b), depth, pitch)
            rel = samples - base
            s = rel @ up
            d = np.linalg.norm(rel - np.outer(s, up), axis=1)
            profile = bottle_profile(b)
            top = profile[:, 1].max()
            inside = (s >= 0) & (s <= top) & (d <= np.interp(s, profile[:, 1],
                                                             profile[:, 0]))
            fractions.append(inside.reshape(len(directions), -1).any(1).mean())
            mouth = base + top * up
            neck = base + (top - b["neck_h"] / 2) * up
            mouth_seen.append(bool(in_view(cam, mouth[None], fov)[0]))
            neck_seen.append(bool(in_view(cam, neck[None], fov)[0]))
        if fractions:
            out[f"{label} pitch{pitch:.2f}"] = dict(
                mean_image_fraction=round(float(np.mean(fractions)), 3),
                max_image_fraction=round(float(np.max(fractions)), 3),
                mouth_in_frame=f"{sum(mouth_seen)}/{len(mouth_seen)}",
                mid_neck_in_frame=f"{sum(neck_seen)}/{len(neck_seen)}")
    return out


# --------------------------------------------------------------------------
# G5: swept radius and the full-arm audit
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# G4: the datum, measured rather than asserted
# --------------------------------------------------------------------------

def _rail_back_face(cell=0.25):
    """Rasterise gripper_link's rail back face: (occupied cell centres, cell area).

    The -X-facing triangles at the rail step, projected onto Y-Z and sampled on a
    grid, so "how much of the pad actually lands on metal" is a measurement and not
    a product of three parameters.
    """
    mesh = rb.load_mesh("gripper_link")
    normals, centres = mesh.face_normals, mesh.triangles_center
    at_step = ((normals[:, 0] < -0.99)
               & (np.abs(centres[:, 0] - P.RAIL_BACK_X) < 0.3))
    tris = mesh.triangles[at_step][:, :, 1:]
    if not len(tris):
        return np.zeros((0, 2)), cell * cell
    lo, hi = tris.reshape(-1, 2).min(0), tris.reshape(-1, 2).max(0)
    ys = np.arange(lo[0], hi[0], cell) + cell / 2
    zs = np.arange(lo[1], hi[1], cell) + cell / 2
    grid = np.stack(np.meshgrid(ys, zs, indexing="ij"), -1).reshape(-1, 2)
    inside = np.zeros(len(grid), bool)
    for a, b, c in tris:
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        rel = grid - a
        u = (rel[:, 0] * v1[1] - v1[0] * rel[:, 1]) / den
        v = (v0[0] * rel[:, 1] - rel[:, 0] * v0[1]) / den
        inside |= (u >= 0) & (v >= 0) & (u + v <= 1)
    return grid[inside], cell * cell


def g4_stop_area(cam=None):
    """Area of real overlap between the four stop pads and the rail's back face."""
    cam = cam or design.CameraFrame()
    cells, area = _rail_back_face()
    if not len(cells):
        return dict(stop_area_mm2=0.0, note="no rail back face found in the mesh")
    # The pads are the brackets' +X faces at X = RAIL_BACK_X.  Take the solids'
    # own footprint there rather than the parameter rectangle: STOP_Y0, BORE_DIA and
    # YOKE_HALF_Z all trim it, and the bore's arc eats the inboard edge.
    covered = np.zeros(len(cells), bool)
    for part in (design.upper_bracket(cam), design.lower_bracket(cam)):
        mesh = to_mesh(part)
        faces = mesh.face_normals
        at = ((faces[:, 0] > 0.99)
              & (np.abs(mesh.triangles_center[:, 0] - P.RAIL_BACK_X) < 0.05))
        for a, b, c in mesh.triangles[at][:, :, 1:]:
            v0, v1 = b - a, c - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            rel = cells - a
            u = (rel[:, 0] * v1[1] - v1[0] * rel[:, 1]) / den
            v = (v0[0] * rel[:, 1] - rel[:, 0] * v0[1]) / den
            covered |= (u >= 0) & (v >= 0) & (u + v <= 1)
    return dict(stop_area_mm2=round(float(covered.sum()) * area, 1),
                rail_back_face_area_mm2=round(float(len(cells)) * area, 1),
                rail_back_face_in_key_band_mm2=round(float(
                    ((np.abs(cells[:, 1]) <= P.YOKE_HALF_Z)
                     & (np.abs(cells[:, 0]) >= P.STOP_Y0)).sum()) * area, 1),
                cell_mm=0.25,
                method="both brackets' +X faces at X = RAIL_BACK_X, rasterised "
                       "against the -X-facing triangles of gripper_link.STL at the "
                       "rail step; no parameter formula")


def g4_clocking_slop(cam=None, limit_deg=6.0, step_deg=0.01):
    """Residual rotation about X before the key ears touch the rail, measured.

    Rotate the real gripper meshes about the roll axis until the brackets' key ears
    interfere, in both directions.  The formula this replaces used the rail's Y
    half-width, 54.16 mm, as the lever arm; the ears only touch over |Z| <= 7 mm, so
    it read 0.317 deg where the truth is an order of magnitude more.
    """
    cam = cam or design.CameraFrame()
    # Only the key ears against only the rail.  Rotating the whole bracket against
    # the whole gripper measures the *stop pads*, which are in contact at zero by
    # design, and reports one step of slop -- it says nothing about the clocking.
    mesh = rb.gripper_meshes(0.0)["gripper_link"]
    keep = ((mesh.triangles_center[:, 0] > P.RAIL_BACK_X + 0.1)
            & (mesh.triangles_center[:, 0] <= P.RAIL_FRONT_X))
    sub = mesh.submesh([np.flatnonzero(keep)], append=True)
    # Surface samples, not vertices: the contact is on the *edge* of the rail's end
    # face, between its corners, and a vertex-only test finds nothing at all.
    rail = np.vstack([trimesh.sample.sample_surface_even(sub, 300000, seed=3)[0],
                      sub.vertices])
    out = {}
    for sign in (1.0, -1.0):
        angle = 0.0
        while angle < limit_deg:
            angle += step_deg
            theta = math.radians(sign * angle)
            rot = np.array([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta), math.cos(theta)]])
            turned = rail[:, 1:] @ rot.T
            # inside the key ears' pocket: |Y| >= KEY_INNER_Y over |Z| <= KEY_HALF_Z
            if ((np.abs(turned[:, 0]) >= P.KEY_INNER_Y)
                    & (np.abs(turned[:, 1]) <= P.KEY_HALF_Z)).any():
                break
        out["positive_deg" if sign > 0 else "negative_deg"] = round(angle, 3)
    total = out["positive_deg"] + out["negative_deg"]
    return dict(free_rotation_deg=out, total_slop_deg=round(total, 3),
                lever_arm_mm=P.KEY_HALF_Z, nominal_fit_mm=P.KEY_CLEARANCE,
                pupil_radius_mm=None,
                formula_the_first_revision_used_deg=round(math.degrees(math.atan2(
                    2 * P.KEY_CLEARANCE, 2 * P.RAIL_HALF_Y)), 3),
                method=f"the rail's own vertices from gripper_link.STL rotated about "
                       f"X in {step_deg} deg steps until they enter the key ears' "
                       f"pocket, both ways. The first revision computed "
                       f"atan(2 * fit / (2 * RAIL_HALF_Y)) instead, which uses the "
                       f"rail's 54.16 mm Y half-width as the lever arm; the ears only "
                       f"touch over |Z| <= {P.KEY_HALF_Z} mm, so it read an order of "
                       f"magnitude low",
                note="this is the slop with the grub screw NOT fitted. Running the "
                     "M3 in bears on the near rail end face and presses the far "
                     "ear's inner face flat against the far one, which is a plane "
                     "on a plane over 10.65 x 12.4 mm -- after that the clocking is "
                     "set by print flatness, not by the fit.")


def g5_swept_radius(meshes):
    """Largest distance of any payload point from the wrist-roll (X) axis."""
    per_part, worst, at = {}, 0.0, None
    for name, mesh in meshes.items():
        radii = np.hypot(mesh.vertices[:, 1], mesh.vertices[:, 2])
        per_part[name] = round(float(radii.max()), 1)
        if radii.max() > worst:
            worst = float(radii.max())
            at = mesh.vertices[int(np.argmax(radii))]
    return dict(swept_radius_mm=round(worst, 1),
                at_point_mm=[round(v, 1) for v in at], per_part=per_part)


CLAMP_PARTS = ("lower_bracket",)      # the strap has no camera on it


def _one_arm_audit(meshes, robot, samples, seed):
    """One hand's audit.  Returns the count and a per-part census of the hits."""
    payload = {name: rb.collision_object(mesh) for name, mesh in meshes.items()}
    rng = np.random.default_rng(seed)
    collisions, clear, examples = 0, 0, []
    per_part = {}
    poses = rng.uniform(robot.limits[:, 0], robot.limits[:, 1], (samples, 6))
    for i, q in enumerate(poses):
        jaw = (0.0, 20.0, 50.0)[i % 3]
        robot.set_pose(q, jaw)
        if robot.bare_collisions():
            continue
        clear += 1
        hits = [(part, link) for link in robot.names[:-1]
                for part, obj in payload.items()
                if rb.intersects(obj, robot.objects[link])]
        if hits:
            collisions += 1
            for part in {p for p, _ in hits}:
                per_part[part] = per_part.get(part, 0) + 1
            if len(examples) < 5:
                examples.append(dict(q_deg=np.rad2deg(q).round(3).tolist(),
                                     jaw_half_gap_mm=jaw, pairs=hits))
    return clear, collisions, per_part, examples


GATE_HAND = "right"     # the hand v4's 64 is comparable to; see below


def g5_arm_audit(meshes, robot=None, samples=P.ARM_AUDIT_SAMPLES,
                 seed=P.ARM_AUDIT_SEED, both_hands=True):
    """Seeded random full-arm audit, identical in method and seed to v4's.

    Run for both hands.  **The gate is the right hand**, and this function does not
    decide that -- ``verify.py`` does, from ``GATE_HAND`` -- but it used to say
    three different things about it: this docstring said "gated on the worse one",
    the returned ``gated_hand`` said "left", and verify.py gated the right.  Only
    one of the three was the code that ran.

    Why the right hand: v4's reference of 64 is one hand's number, because v4 only
    ever had one.  Comparing a "worse of two hands" figure with it is comparing a
    maximum of two draws with a single draw, which is biased upward by construction.
    So the like-for-like number is the right hand, the left is measured and reported
    beside it, and the left's excess is a disclosed miss rather than a renamed gate.

    The per-part census answers the question the first cut of this got backwards.
    "The clamp band alone accounts for 62 of them and the camera costs 1 more" is
    not possible -- a subset cannot collide in more configurations than the whole --
    and it is not what the numbers say.
    """
    robot = robot or rb.Robot()
    hands = {"right": meshes}
    if both_hands:
        mirror = {}
        for name, mesh in meshes.items():
            copy = mesh.copy()
            copy.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
            copy.fix_normals()
            mirror[name] = copy
        hands["left"] = mirror
    out = {}
    for hand, hand_meshes in hands.items():
        clear, collisions, per_part, examples = _one_arm_audit(
            hand_meshes, robot, samples, seed)
        out[hand] = dict(bare_arm_clear=clear, payload_collisions=collisions,
                         per_part_configs=dict(sorted(per_part.items(),
                                                      key=lambda kv: -kv[1])),
                         examples=examples)
    gated = out[GATE_HAND]["payload_collisions"]
    other = {h: v["payload_collisions"] for h, v in out.items() if h != GATE_HAND}
    return dict(samples=samples, seed=seed, per_hand=out,
                gate_hand=GATE_HAND,
                gate_hand_collisions=gated,
                reported_not_gated=other,
                v4_board_reference=P.G5_V4_BOARD_COLLISIONS,
                bare_arm_clear=out["right"]["bare_arm_clear"],
                counting_note=f"one number is the gate and it is the {GATE_HAND} "
                              f"hand's, because v4's "
                              f"{P.G5_V4_BOARD_COLLISIONS} is one hand's number. "
                              f"The other hand is measured and reported here, not "
                              f"folded into a worse-of-two figure that v4 has no "
                              f"counterpart for.",
                method="one jaw opening per configuration, (0, 20, 50)[i % 3]; a "
                       "configuration where the bare arm already self-collides is "
                       "skipped, not counted; the count is configurations, not "
                       "(configuration, jaw) pairs")


# --------------------------------------------------------------------------
# G6: stiffness
# --------------------------------------------------------------------------

def g5_audit_spread(meshes, robot=None, samples=P.ARM_AUDIT_SAMPLES,
                    seeds=P.ARM_AUDIT_SPREAD_SEEDS):
    """The same audit at several seeds, so the gated count has an error bar.

    Reported, not gated.  The gate is one 10,000-configuration count against v4's
    64, and a count of that size carries roughly sqrt(64) = 8 of sampling noise on
    its own -- so "74 against 64" is a difference of about one standard error, and
    reading it as a property of the mirrored bracket is reading the noise.  This
    measures the noise instead of assuming it: the same payload, the same method,
    a handful of different draws.

    It is what makes the left hand's miss reportable as a disclosed miss with a
    size rather than as a verdict.
    """
    robot = robot or rb.Robot()
    rows = []
    for seed in seeds:
        audit = g5_arm_audit(meshes, robot=robot, samples=samples, seed=seed)
        rows.append(dict(seed=seed,
                         bare_arm_clear=audit["bare_arm_clear"],
                         **{hand: audit["per_hand"][hand]["payload_collisions"]
                            for hand in audit["per_hand"]}))
    out = dict(seeds=list(seeds), samples=samples, per_seed=rows)
    for hand in ("right", "left"):
        values = np.array([r[hand] for r in rows], float)
        out[hand] = dict(mean=round(float(values.mean()), 1),
                         sd=round(float(values.std(ddof=1)), 1),
                         min=int(values.min()), max=int(values.max()),
                         under_v4_reference=int((values
                                                 < P.G5_V4_BOARD_COLLISIONS).sum()))
    delta = np.array([r["left"] - r["right"] for r in rows], float)
    out["left_minus_right"] = dict(mean=round(float(delta.mean()), 1),
                                   sd=round(float(delta.std(ddof=1)), 1),
                                   min=int(delta.min()), max=int(delta.max()))
    out["note"] = (
        f"the gated seed ({P.ARM_AUDIT_SEED}) is the first row. The mirrored "
        f"payload really does meet this arm a little worse -- the mean difference "
        f"is positive -- but the difference is smaller than the spread of either "
        f"hand, and on "
        f"{out['left']['under_v4_reference']} of {len(rows)} draws the left hand "
        f"comes in under v4's {P.G5_V4_BOARD_COLLISIONS} with no design change at "
        f"all.")
    return out


# The perturbations tried to bring the LEFT hand under v4's reference, and the
# gate each one spends.  They were four sentences of README prose with no code
# behind them, and a clean-room reviewer reran them: two of the four numbers were
# wrong and one perturbation the README said moved the count by 5 moved it by 0.
# So they are computed here, on every run, and rendered into the README from the
# file -- the same fix as the corridor census, applied to the paragraph that
# carries the argument for shipping a hand that misses the reference.
G5_WHAT_IF = (
    dict(name="camera 3 mm closer to the roll axis",
         override=dict(CAM_POS=(-29.0, 51.0, 45.0))),
    dict(name="camera 6 mm closer to the roll axis",
         override=dict(CAM_POS=(-29.0, 51.0, 42.0))),
    dict(name="camera 5 mm inboard in Y",
         override=dict(CAM_POS=(-29.0, 46.0, 48.0))),
    dict(name="camera pitch 5 deg shallower", override=dict(CAM_PITCH_DEG=20.0)),
    dict(name="USB keep-out 25 mm -> 18 mm",
         override=dict(USB_KEEPOUT=(12.0, 8.0, 18.0))),
)


def _what_if_gates(cam, meshes):
    """The four gates a camera-pose change can spend, measured on that change.

    Not a full verify run -- these are the ones that bind when the head moves, and
    each is the same computation ``verify.py`` gates.  They are *measured* because
    the previous version of this table carried them as typed claims, in the one
    paragraph that argues for shipping a hand that misses the G5 reference.
    """
    out = {}
    sight = g2_sightlines(cam, meshes)
    far = min(v["visible_fraction"] for k, v in sight["c_far_jaw"].items()
              if float(k.replace("opening", "")) >= 20.0)
    out["G2'c far jaw visible, openings >= 20 mm"] = dict(
        measured=round(far, 2), threshold=P.G2_FAR_JAW_FRACTION,
        ok=bool(far >= P.G2_FAR_JAW_FRACTION))
    fit = g7_module_fit(cam)["intersection_mm3"]
    out["G7 module fit"] = dict(measured=fit, threshold=P.G7_MODULE_FIT_MM3,
                                ok=bool(fit <= P.G7_MODULE_FIT_MM3))
    pitch = cam.pitch
    lo, hi = P.G3_PITCH_RANGE
    out["G3 pitch window"] = dict(measured=pitch, threshold=[lo, hi],
                                  ok=bool(lo <= pitch <= hi))
    swept = g5_swept_radius(meshes)["swept_radius_mm"]
    out["G5 swept radius"] = dict(measured=swept, threshold=P.G5_SWEPT_RADIUS,
                                  ok=bool(swept <= P.G5_SWEPT_RADIUS))
    return out


def _spends(gates):
    """One line naming the gates this change breaks, with their numbers."""
    bad = [f"{name}: {g['measured']} against {g['threshold']}"
           for name, g in gates.items() if not g["ok"]]
    return "; ".join(bad) if bad else "nothing"


def g5_what_if(robot=None, samples=P.ARM_AUDIT_SAMPLES, seed=P.ARM_AUDIT_SEED,
               cases=G5_WHAT_IF):
    """Rerun the audit with one parameter moved, for each thing that was tried.

    Each case rebuilds the payload from scratch with the override in place, so what
    is counted is the design that override produces -- not the shipped meshes with
    a number changed underneath them -- and the gates that change would spend are
    measured on it as well.  Reported, never gated: the gate is the shipped
    geometry.  What this is for is that the README's claim is "we tried and nothing
    reachable gets under the reference", and a claim like that has to carry the
    numbers that were actually measured.  It did not: of the four figures the
    previous version of this paragraph quoted, a clean-room reviewer reproduced
    one.
    """
    robot = robot or rb.Robot()
    rows = []
    for case in (None,) + tuple(cases):
        override = {} if case is None else case["override"]
        previous = {k: getattr(P, k) for k in override}
        try:
            for key, value in override.items():
                setattr(P, key, value)
            # _frame() reads the module attributes at call time.  CameraFrame's own
            # defaults are bound at import, so overriding params.CAM_POS and calling
            # CameraFrame() would build the shipped camera and quietly report the
            # shipped count for every case.
            cam = _frame()
            meshes = payload_meshes(cam)
            counts = _both_hands(meshes, robot, samples, seed)
            gates = _what_if_gates(cam, meshes)
        finally:
            for key, value in previous.items():
                setattr(P, key, value)
        rows.append(dict(name="as shipped" if case is None else case["name"],
                         override={k: list(v) if isinstance(v, tuple) else v
                                   for k, v in override.items()},
                         gates=gates, spends=_spends(gates), **counts))
    # The best count that does not break something, and the best at any price.
    clean = [r for r in rows[1:] if r["spends"] == "nothing"]
    best = min(r["left"] for r in rows)
    best_clean = min((r["left"] for r in clean), default=rows[0]["left"])
    return dict(seed=seed, samples=samples, per_case=rows,
                best_left=best, best_left_no_gate_spent=best_clean,
                reaches_reference=bool(best_clean < P.G5_V4_BOARD_COLLISIONS),
                reference=P.G5_V4_BOARD_COLLISIONS,
                note="one draw each, at the gated seed, and a draw of this size "
                     "carries about 8 of sampling noise -- so these are the size "
                     "of the move each change buys, not exact counts. The gates "
                     "column is measured on the perturbed design, not asserted. "
                     "Reported, not gated.")


def _frame():
    """A CameraFrame built from params as they stand *now*, not as they imported."""
    return design.CameraFrame(pos=P.CAM_POS, yaw=P.CAM_YAW_DEG,
                              pitch=P.CAM_PITCH_DEG, roll=P.CAM_ROLL_DEG,
                              side=P.HANDEDNESS)


def _both_hands(meshes, robot, samples, seed):
    """{'right': n, 'left': n} for one payload, the audit's own method."""
    audit = g5_arm_audit(meshes, robot=robot, samples=samples, seed=seed)
    return {hand: audit["per_hand"][hand]["payload_collisions"]
            for hand in ("right", "left")}


def g6_stiffness(cam=None, mass=None):
    """First lateral mode of the camera head, as a 2-DOF model.

    Three things the first cut of this left out, all of which matter more than the
    third figure it quoted:

      * the head's real mass.  It was a hand-entered 34 g labelled "measured
        below", which it was not; the pigtail and the M2 hardware were missing.  It
        now comes from ``mass_properties``, which places every gram.
      * the head's rotary inertia.  The module hangs off the end of the strut
        sideways -- the strut axis and the optical axis are 128 deg apart -- so the
        tip rotates as well as translating, and a translation-only model is
        optimistic.
      * the tongue.  The strut lands on the plate 30 mm off the module's centre and
        the plate carries the rest, so the load path is two springs in series.

    Euler-Bernoulli throughout, the strut taper handled by its mean second moment,
    and the same blunt 0.6 knockdown for the clamp band, the bolted split and the
    rubber liner, none of which are modelled.  Report it as "about this, several
    times the gate", not to three figures.
    """
    cam = cam or design.CameraFrame()
    mass = mass or mass_properties(cam)
    root, tip, axis, length, _ = design._strut_geometry(cam)

    def weak(w, h):
        return min(w * h ** 3, h * w ** 3) / 12.0

    # --- spring 1: the tapered strut, band to strut tip, on its weak axis
    i_root, i_tip = weak(P.STRUT_W, P.STRUT_H), weak(P.STRUT_TIP_W, P.STRUT_TIP_H)
    i_strut = 0.5 * (i_root + i_tip)
    k_strut = 3.0 * P.PETG_E * i_strut / length ** 3            # N/mm
    # --- spring 2: the plate, strut tip to the module's centre, as a T section
    span = float(np.hypot(*design.plate_tip_uv()))
    flange_w = P.STRUT_TIP_W + 8.0          # the tongue's width, not the whole plate
    a1, y1 = flange_w * P.PLATE_T, P.PLATE_T / 2
    a2, y2 = P.SPINE_W * P.SPINE_H, P.PLATE_T + P.SPINE_H / 2
    ybar = (a1 * y1 + a2 * y2) / (a1 + a2)
    i_plate = (flange_w * P.PLATE_T ** 3 / 12.0 + a1 * (ybar - y1) ** 2
               + P.SPINE_W * P.SPINE_H ** 3 / 12.0 + a2 * (y2 - ybar) ** 2)
    k_plate = 3.0 * P.PETG_E * i_plate / span ** 3
    k_series = 1.0 / (1.0 / k_strut + 1.0 / k_plate)
    # --- head: what hangs on the far end of that chain.  Measured, not typed: the
    # camera, its pigtail and its screws, plus the part of the upper bracket that is
    # forward of the strut's mid-point, which is the plate, tongue and spine.
    head = sum(v for k, v in mass["per_part_g"].items()
               if k in ("camera_module", "pigtail", "m2_hardware"))
    upper = to_mesh(design.upper_bracket(cam))
    along = (upper.vertices - root) @ axis
    forward_fraction = float((along > 0.5 * length).mean())
    head += forward_fraction * mass["per_part_g"]["upper_bracket_right"]
    # --- the head's CoM stands off the plate's mounting face, so the plate's end
    # slope rotates it as well as moving it.  e is that stand-off, measured along
    # the optical axis from the plate's front face to the module's own centroid.
    envelope = design.camera_envelope(cam)
    module = to_mesh(envelope["camera_body"].union(envelope["camera_holder"])
                     .union(envelope["camera_lens"]))
    offset = float(abs((np.asarray(module.center_mass) - cam.pos) @ axis
                       - P.BODY_BACK))
    total_span = length + span
    rotary = (1.0 + 1.5 * offset / total_span) ** 2
    beam_mass = (P.PETG_DENSITY * P.EFFECTIVE_FILL
                 * 0.5 * (P.STRUT_W * P.STRUT_H + P.STRUT_TIP_W * P.STRUT_TIP_H)
                 * length)
    effective = (head * rotary + 0.24 * beam_mass) / 1000.0     # kg
    knockdown = 0.6
    f_ideal = math.sqrt(k_series * 1000.0 / effective) / (2 * math.pi)
    f_joint = math.sqrt(knockdown * k_series * 1000.0 / effective) / (2 * math.pi)
    return dict(section_root_mm=[P.STRUT_W, P.STRUT_H],
                section_tip_mm=[P.STRUT_TIP_W, P.STRUT_TIP_H],
                strut_length_mm=round(length, 1),
                plate_span_mm=round(span, 1),
                I_strut_mean_mm4=round(i_strut, 1),
                I_plate_T_section_mm4=round(i_plate, 1),
                plate_T_section_mm=[flange_w, P.PLATE_T, P.SPINE_W, P.SPINE_H],
                E_MPa=P.PETG_E,
                k_strut_N_per_mm=round(k_strut, 1),
                k_plate_N_per_mm=round(k_plate, 1),
                k_series_N_per_mm=round(k_series, 1),
                head_mass_g=round(head, 1),
                head_bracket_fraction=round(forward_fraction, 3),
                beam_mass_g=round(beam_mass, 1),
                head_offset_from_plate_face_mm=round(offset, 1),
                rotary_mass_factor=round(rotary, 2),
                f1_no_knockdown_Hz=round(f_ideal, 1),
                f1_with_joint_knockdown_Hz=round(f_joint, 1),
                reported=f"about {round(f_joint / 10) * 10:.0f} Hz, "
                         f"{f_joint / P.G6_FIRST_MODE_HZ:.1f}x the "
                         f"{P.G6_FIRST_MODE_HZ:.0f} Hz gate",
                model="two Euler-Bernoulli springs in series -- the tapered strut on "
                      "its weak axis, then the plate's own T section (tongue flange "
                      "plus rear spine) over the 31 mm from the strut's tip to the "
                      "module's centre. Head mass from mass_properties, including "
                      "the pigtail, the M2 screws and the measured share of the "
                      "bracket forward of the strut's mid-point. The head's stand-off "
                      "from the plate face is carried as a rotary-inertia factor "
                      "(1 + 1.5 e / L)^2, plus Dunkerley 0.24 * beam mass and a 0.6 "
                      "stiffness knockdown for the clamp band, the bolted split and "
                      "the rubber liner, none of which are modelled. Not an FE "
                      "result: two significant figures at most.")


# --------------------------------------------------------------------------
# G7 extras: top-down picks and the pour wrist roll
# --------------------------------------------------------------------------

def top_down_pick(meshes, tcp_above_table=45.0):
    """Gripper pointing straight down, TCP 45 mm above the table.

    Gripper +X points at the table, so height above the table is
    (TCP height + TCP_X) - x.  The payload must stay well above the plane the
    fingertips are working in.
    """
    origin_height = tcp_above_table + P.TCP_X
    payload_max_x = float(max(m.vertices[:, 0].max() for m in meshes.values()))
    payload_low = origin_height - payload_max_x
    fingertip_plane = origin_height - P.FINGERTIP_X
    return dict(tcp_above_table_mm=tcp_above_table,
                fingertip_plane_mm=round(fingertip_plane, 1),
                payload_lowest_mm=round(payload_low, 1),
                margin_mm=round(payload_low - fingertip_plane, 1))


def pour_roll_dip(meshes, pitches=(0.30, 0.45, 0.60), step=5.0):
    """Lowest point of payload vs bare gripper as the wrist rolls for a pour.

    Heights are along world up in the gripper frame, relative to the TCP plane,
    over rolls of 0..+-120 deg about the tool axis.
    """
    payload = np.concatenate([m.vertices for m in meshes.values()])
    gripper = np.concatenate([m.vertices for m in rb.gripper_meshes(50.0).values()])
    rolls = np.arange(-120.0, 120.0 + step, step)
    out = {}
    for pitch in pitches:
        up = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
        heights = {}
        for name, points in (("payload", payload), ("gripper", gripper)):
            values = []
            for phi in np.radians(rolls):
                rot = np.array([[1.0, 0.0, 0.0],
                                [0.0, math.cos(phi), -math.sin(phi)],
                                [0.0, math.sin(phi), math.cos(phi)]])
                values.append(float(((points @ rot.T) @ up).min()))
            heights[name] = np.array(values)
        out[f"pitch{pitch:.2f}"] = dict(
            gripper_lowest_mm=round(float(heights["gripper"].min()), 1),
            payload_lowest_roll_plus_mm=round(float(heights["payload"][rolls >= 0].min()), 1),
            payload_lowest_roll_minus_mm=round(float(heights["payload"][rolls <= 0].min()), 1),
            payload_at_zero_roll_mm=round(float(heights["payload"][rolls == 0][0]), 1))
    return out


# --------------------------------------------------------------------------
# mass properties and mesh validity
# --------------------------------------------------------------------------

M4_MASS_G = 3.4         # M4 x 25 cap screw with its nut
# 3 x 10 mm steel pin: pi * 1.5^2 * 10 = 70.7 mm3 at 7.85 g/cc.  It said 1.1 g and
# "4 mm x 10", which is neither the BOM's pin nor design.DOWEL_DIA -- and because
# only one of the two dowels was being built at all, the payload total came out
# right by cancellation (1 x 1.1 where it should have been 2 x 0.55) while the
# centre of mass sat 0.39 mm off in Y.
DOWEL_MASS_G = round(math.pi * (P.DOWEL_DIA / 2) ** 2 * P.DOWEL_LENGTH * 7.85e-3, 2)
M2_MASS_G = 0.6         # M2 screw plus its printed standoff, each


def mass_properties(cam=None):
    """Payload mass, centre of mass and inertia about the gripper_link origin.

    Every gram is placed: the brackets and the camera module over their own
    solids, the bolts and dowels over the fastener envelope, the pigtail over
    the USB keep-out, and the four M2 screws as point masses at the plate's
    screw slots.  A lumped hardware *mass* with no position (what the first cut
    of this function did) makes the exported tensor describe a lighter body than
    the exported mass, which comes out as an inertia that no rigid body can
    have -- MuJoCo's ``fullinertia`` rejects it, and rightly."""
    cam = cam or design.CameraFrame()
    items = []
    for name, (solid, _) in design.printed_parts(cam).items():
        if "gauge" in name or "standoff" in name:
            continue
        mesh = to_mesh(solid)
        items.append((name, mesh, P.PETG_DENSITY * P.EFFECTIVE_FILL))
    envelope = design.camera_envelope(cam)
    # The bought module is mostly air; use its catalogue mass over its envelope.
    body = to_mesh(envelope["camera_body"].union(envelope["camera_holder"])
                   .union(envelope["camera_lens"]))
    items.append(("camera_module", body, P.CAMERA_MASS_G / float(body.volume)))
    # The cable's connector and the first stretch of pigtail live in the USB
    # keep-out; the rest of the lead is tied off to the arm and is not payload.
    usb = to_mesh(envelope["camera_usb"])
    items.append(("pigtail", usb, P.PIGTAIL_MASS_G / float(usb.volume)))
    for name, solid in design.fastener_envelope(cam.side).items():
        mesh = to_mesh(solid)
        nominal = M4_MASS_G if name.startswith("m4") else DOWEL_MASS_G
        items.append((name, mesh, nominal / float(mesh.volume)))

    total, first, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
    per_part = {}

    def add(name, mass, centre, local):
        nonlocal total, first, inertia
        per_part[name] = round(per_part.get(name, 0.0) + mass, 1)
        total += mass
        first += mass * centre
        inertia += local + mass * ((centre @ centre) * np.eye(3) - np.outer(centre, centre))

    for name, mesh, density in items:
        mass = float(mesh.volume) * density
        # Solid-body inertia about the part's own centre, scaled to that mass.
        local = np.asarray(mesh.moment_inertia, float) * (mass / float(mesh.volume))
        add(name, mass, np.asarray(mesh.center_mass, float), local)
    # M2 screws and standoffs: point masses at the four slots, mid-slot and
    # mid-standoff. Their own inertia is under 1000 g mm^2, which is noise here.
    w = design.plate_rear_offset() / 2.0
    for su, sv, half_lo, half_hi in design._m2_slot_centres():
        half = (half_lo + half_hi) / 2.0
        add("m2_hardware", M2_MASS_G, np.asarray(cam.point(su * half, sv * half, w), float),
            np.zeros((3, 3)))

    com = first / total
    about_com = inertia - total * ((com @ com) * np.eye(3) - np.outer(com, com))
    eig = np.linalg.eigvalsh(0.5 * (about_com + about_com.T))
    assert eig.min() > 0 and eig[0] + eig[1] > eig[2], \
        f"mass properties are not those of a rigid body: principal inertia {eig}"
    return dict(mass_g=round(total, 1), per_part_g=per_part,
                com_mm=[round(v, 2) for v in com],
                inertia_about_origin_g_mm2=[[round(v, 1) for v in row]
                                            for row in inertia],
                principal_about_com_g_mm2=[round(v, 1) for v in eig],
                effective_fill=P.EFFECTIVE_FILL,
                note="printed parts at PETG 1.27 g/cc x effective fill; the "
                     "camera module and the pigtail use their catalogue mass "
                     "over their envelopes; bolts, dowels and M2 screws are "
                     "nominal masses at their real positions")


def mesh_validity(stl_dir):
    """Every exported STL: watertight, one body, positive volume, on the bed."""
    report = {}
    for path in sorted(stl_dir.glob("*.stl")):
        mesh = trimesh.load(path)
        bodies = mesh.split(only_watertight=False)
        report[path.name] = dict(
            watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            single_body=len(bodies) == 1,
            volume_cm3=round(float(mesh.volume) / 1000.0, 2),
            size_mm=[round(v, 1) for v in mesh.extents],
            sits_on_bed=bool(abs(mesh.bounds[0, 2]) < 1e-3),
            fits_bed=bool(mesh.extents[0] <= P.BED[0] and mesh.extents[1] <= P.BED[1]))
    return report


def _inside_primitives(points, primitives, tolerance=0.0):
    """Boolean per point: inside the union of the primitives, grown by `tolerance`."""
    inside = np.zeros(len(points), bool)
    for prim in primitives:
        centre = np.array(prim["pos"]) * 1000.0
        rot = np.array(prim["rot"])
        # `rot` holds the local axes as columns, so global -> local is @ rot.
        local = (points - centre) @ rot
        if prim["type"] == "box":
            half = np.array(prim["size"]) * 1000.0 / 2.0 + tolerance
            inside |= np.all(np.abs(local) <= half, axis=1)
        else:
            radius, length = prim["size"][0] * 1000.0, prim["size"][1] * 1000.0
            inside |= ((np.hypot(local[:, 0], local[:, 1]) <= radius + tolerance)
                       & (np.abs(local[:, 2]) <= length / 2 + tolerance))
    return inside


def _arch_faces(mesh):
    """Which faces of a bracket STL lie on the crown of the bore, in the print pose.

    Both brackets reach their print pose by a rigid move that leaves X alone and
    puts the split plane on the bed (``export.print_pose``): the upper is
    translated down by SPLIT_GAP, the lower is turned 180 deg about X and then
    translated down by the same amount.  So in either exported file the bore's axis
    is the line y = 0, z = -SPLIT_GAP, and the arch is what lies within the bore's
    own radius of it over the clamp band's X span.  Everything else that overhangs
    is something else, which is the whole point of measuring the two apart.
    """
    centres = mesh.triangles.mean(axis=1)
    radius = np.hypot(centres[:, 1], centres[:, 2] + P.SPLIT_GAP)
    return ((radius <= P.BORE_DIA / 2 + P.BORE_RELIEF + 0.5)
            & (centres[:, 0] >= P.CLAMP_X0 - 0.5)
            & (centres[:, 0] <= P.CLAMP_X1 + 0.5))


def overhang_report(stl_dir, limit_deg=50.0):
    """Downward-facing area that overhangs more than `limit_deg` from vertical.

    Measured on the exported, pre-oriented STLs, so it is the number the slicer
    will see.  Anything reported here is where support (or a brim) is needed.

    Split into three numbers, because the README's "supports: none, the crown of the
    bore is an arch" attributed the whole figure to the arch and a review found
    that a third of it is not the arch at all: 654 mm2 of the upper bracket's
    1855 is the strut root and the camera plate cantilevering out over the bolt
    ear, of which 441 mm2 has nothing below it inside the part at all, up to 58 mm
    above the bed.  An arch is self-supporting; a ceiling in open air is not.

      * ``arch_area_mm2``     over the clamp band's own bore, where the overhang is
                              an arch and the crown is relieved on purpose;
      * ``other_area_mm2``    everything else that overhangs past ``limit_deg``;
      * ``free_area_mm2``     of that, the part with nothing under it: a ray cast
                              straight down from the face's centre hits no triangle
                              of the same part.  This is the gated number.
    """
    report = {}
    for path in sorted(stl_dir.glob("*.stl")):
        mesh = trimesh.load(path)
        normals, areas = mesh.face_normals, mesh.area_faces
        # The face sitting on the bed points straight down but is supported by
        # the bed, so it is not an overhang.
        on_bed = mesh.triangles[:, :, 2].max(axis=1) < P.LAYER_H
        downward = (normals[:, 2] < -1e-6) & ~on_bed
        # angle of the surface from vertical: 0 = a wall, 90 = a ceiling
        from_vertical = np.degrees(np.arcsin(np.clip(-normals[:, 2], 0.0, 1.0)))
        steep = downward & (from_vertical > limit_deg)
        arch = (_arch_faces(mesh) if path.stem.startswith(("upper_bracket",
                                                          "lower_bracket"))
                else np.zeros(len(areas), bool))
        other = steep & ~arch
        # Nothing below it: cast straight down from each face centre and see
        # whether the part itself is in the way.
        centres = mesh.triangles.mean(axis=1)
        free_area, free_top_z = 0.0, 0.0
        if other.any():
            origins = centres[other] + np.array([0.0, 0.0, -1e-3])
            directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
            hit = mesh.ray.intersects_any(origins, directions)
            free_area = float(areas[other][~hit].sum())
            if (~hit).any():
                free_top_z = float(centres[other][~hit][:, 2].max())
        report[path.name] = dict(
            downward_area_mm2=round(float(areas[downward].sum()), 1),
            unsupported_area_mm2=round(float(areas[steep].sum()), 1),
            arch_area_mm2=round(float(areas[steep & arch].sum()), 1),
            other_area_mm2=round(float(areas[other].sum()), 1),
            free_area_mm2=round(free_area, 1),
            free_highest_z_mm=round(free_top_z, 1),
            worst_overhang_deg=round(float(from_vertical[downward].max()), 1)
            if downward.any() else 0.0,
            height_mm=round(float(mesh.extents[2]), 1),
            footprint_mm=[round(float(mesh.extents[0]), 1),
                          round(float(mesh.extents[1]), 1)])
    report["limit_deg"] = limit_deg
    report["worst_free_area_mm2"] = round(
        max((v["free_area_mm2"] for v in report.values()
             if isinstance(v, dict)), default=0.0), 1)
    report["note"] = (
        f"Three numbers, not one. `arch_area_mm2` is the crown of the bore -- "
        f"unavoidable on any split clamp printed on its split face, self-"
        f"supporting as an arch, and relieved by {P.BORE_RELIEF} mm radially over "
        f"the top +-{P.BORE_RELIEF_DEG:.0f} deg (design._bore_cut) so the droop "
        f"lands in the relief and the clamp bears on the flanks. `other_area_mm2` "
        f"is everything else: the M4 counterbore ceilings, which bridge over "
        f"7.4 mm, and the underside of the strut root and camera plate where they "
        f"cantilever out over the bolt ear. `free_area_mm2` is the part of that "
        f"with nothing under it at all, and it is the gated one -- the README used "
        f"to charge the whole figure to the arch. The nut pockets have no ceiling "
        f"-- they open through the ear's outside face, so no pause at height is "
        f"needed. The gauges and the standoff have zero overhang in their poses.")
    return report


def first_hit(origin, targets, tris):
    """Nearest hit parameter t in (0, 1) per ray, or +inf.  Moller-Trumbore."""
    a = tris[:, 0]
    e1, e2 = tris[:, 1] - a, tris[:, 2] - a
    tvec = origin - a
    qvec = np.cross(tvec, e1)
    dot2 = np.einsum("ij,ij->i", e2, qvec)
    out = []
    for target in np.atleast_2d(targets):
        ray = target - origin
        pvec = np.cross(np.broadcast_to(ray, e2.shape), e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=abs(det) > 1e-10)
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        v = np.einsum("j,ij->i", ray, qvec) * inv
        t = dot2 * inv
        hit = ((abs(det) > 1e-10) & (u >= -1e-8) & (v >= -1e-8)
               & (u + v <= 1 + 1e-8) & (t > 1e-5) & (t < 1 - 1e-5))
        out.append(float(t[hit].min()) if hit.any() else np.inf)
    return np.array(out)


def self_occlusion(cam, meshes, fov, grid=48, opening=20.0):
    """Fraction of the image the mount takes up -- once the gripper is in front of it.

    The first cut of this cast against the mount alone and reported 6 % of the wide
    frame, which ``camera_spec.json`` then used to argue *against* the wide lens.
    Rendering the same pose in MuJoCo changes 0.1 % of pixels. The reason is not the
    near plane: it is that the gripper is in front of the strut on every one of
    those rays.  So the mount's share is now the rays where the mount is hit *and
    nothing is nearer*, which is what a camera sees.
    """
    tan_h = math.tan(math.radians(fov[0] / 2.0))
    tan_v = math.tan(math.radians(fov[1] / 2.0))
    u = tan_h * np.linspace(-1, 1, int(grid * fov[0] / fov[1]))
    v = tan_v * np.linspace(-1, 1, grid)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    directions = (cam.axis + uu.ravel()[:, None] * cam.right
                  + vv.ravel()[:, None] * cam.up)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    far = cam.pupil + directions * 900.0
    mount = rb.triangles([meshes[name] for name in design.PRINTED_PARTS])
    out = {}
    t_mount = first_hit(cam.pupil, far, mount)
    out["mount_alone_image_fraction"] = round(float(np.isfinite(t_mount).mean()), 3)
    # How far out the mount's own surface sits inside the frustum, measured along
    # the OPTICAL AXIS -- which is the axis a near plane is defined on, and so the
    # only depth that can be compared with one.  Recorded because the argument that
    # the bracket's absence from the render is geometry rather than a near-plane
    # artifact rests on it, and because sim/DATASET.md quoted this range ("3.8 to
    # 80.3 mm") from nothing at all.
    hit = np.isfinite(t_mount)
    depths = t_mount[hit] * 900.0 * (directions[hit] @ cam.axis)
    out["mount_depth_in_frustum_mm"] = (
        [round(float(depths.min()), 1), round(float(depths.max()), 1)]
        if depths.size else None)
    for opening in (20.0, 100.0):
        scene = rb.triangles(rb.gripper_meshes(opening / 2.0).values())
        t_grip = first_hit(cam.pupil, far, scene)
        out[f"gripper_image_fraction_opening{opening:.0f}"] = round(
            float(np.isfinite(t_grip).mean()), 3)
        visible = np.isfinite(t_mount) & (t_mount < t_grip)
        out[f"mount_image_fraction_opening{opening:.0f}"] = round(
            float(visible.mean()), 3)
    out["mount_image_fraction"] = out[f"mount_image_fraction_opening{opening:.0f}"]
    out["note"] = ("mount_image_fraction is the mount where nothing is nearer; "
                   "mount_alone_image_fraction is the same cast with the gripper "
                   "removed, which is the number the first revision published")
    return out


# --------------------------------------------------------------------------
# G7: wall thickness, the bought module's space, the plate's screw slots
# --------------------------------------------------------------------------

def wall_thickness(cam=None, samples=60000, seed=5, opposed=-0.7):
    """Wall thickness of every printed part, at even surface samples.

    Measured on the solids in the gripper frame rather than on the exported STLs:
    thickness does not care where a part sits, and the declared exception regions
    below are written in the frame the design is written in.

    G6's written clause is "min wall 2.4 mm" and nothing in the first revision
    measured it, which is how a 0.6 mm nut-pocket floor, a 2.0 mm tie-lug web, a
    0.6-0.8 mm sliver beside the M2 slots and a 1.0 mm gauge web all shipped under a
    "23/23 gates pass".

    A wall is two roughly opposed faces with material between them, so the ray cast
    inward along -normal only counts when the face it lands on faces back: a ray from
    a point on a convex edge crosses the wedge and hits a neighbour at 90 deg, which
    is a corner, not a wall.  Without that filter this measure reports 0.01 mm all
    over a perfectly sound part -- 4.5 % of this bracket's surface, essentially all
    of it edges -- and the gate becomes unreadable.  Both sets are reported.
    """
    cam = cam or design.CameraFrame()
    parts = {name: to_mesh(solid)
             for name, (solid, _) in design.printed_parts(cam).items()}
    report = {}
    for name, mesh in sorted(parts.items()):
        points, face = trimesh.sample.sample_surface_even(mesh, samples, seed=seed)
        normals = mesh.face_normals[face]
        origins = points - normals * 1e-4
        hits, index, tri = mesh.ray.intersects_location(
            origins, -normals, multiple_hits=False)
        if not len(index):
            continue
        raw = np.linalg.norm(hits - origins[index], axis=1)
        facing = np.einsum("ij,ij->i", mesh.face_normals[tri], normals[index])
        wall = facing <= opposed
        walls, at = raw[wall], points[index][wall]
        if not len(walls):
            continue
        excepted = np.zeros(len(walls), bool)
        by_exception = {}
        for entry in design.THIN_EXCEPTIONS:
            inside = entry["region"](at)
            excepted |= inside
            if inside.any():
                by_exception[entry["name"]] = dict(
                    samples=int(inside.sum()),
                    min_mm=round(float(walls[inside].min()), 2),
                    floor_mm=entry["floor_mm"], reason=entry["reason"],
                    ok=bool(walls[inside].min() >= entry["floor_mm"]))
        gated = walls[~excepted]
        report[name] = dict(
            samples=int(len(raw)), wall_samples=int(wall.sum()),
            min_wall_mm=round(float(gated.min()), 2) if len(gated) else None,
            p0_1_mm=round(float(np.percentile(gated, 0.1)), 2) if len(gated) else None,
            p1_mm=round(float(np.percentile(gated, 1.0)), 2) if len(gated) else None,
            fraction_under_min_wall=round(float((gated < P.MIN_WALL).mean()), 4)
            if len(gated) else 0.0,
            thinnest_at_mm=[round(v, 1) for v in at[~excepted][int(np.argmin(gated))]]
            if len(gated) else None,
            declared_exceptions=by_exception,
            # The unfiltered number, for comparison with a reviewer casting rays
            # without the opposed-face test.
            unfiltered_p0_1_mm=round(float(np.percentile(raw, 0.1)), 2),
            unfiltered_fraction_under_min_wall=round(
                float((raw < P.MIN_WALL).mean()), 4))
    report["min_wall_mm"] = P.MIN_WALL
    report["method"] = (f"even surface samples, ray cast inward along -normal, first "
                        f"hit; counted as a wall only where the face it lands on "
                        f"faces back (normal dot <= {opposed}), which excludes convex "
                        f"edges and chamfer wedges")
    return report


def split_plane_confinement(cam=None):
    """G9a/G9b: each printed half stays on its own side of the split plane, and
    the two halves do not intersect when assembled.

    The defect this exists for: the zip-tie lug was unioned onto the strut *after*
    the strut had been intersected with the upper half-space, so 111 mm3 of the
    upper bracket hung 2.74 mm below the plane -- 12.9 mm3 of it inside the lower
    strap, and the rest a nub the exported STL then stood on, with its 1182 mm2
    split face floating 2.65 mm above the bed.  Thirty gates measured neither the
    plane, the assembly nor the print pose.

    A feature is allowed across the plane only if it is listed in
    ``design.SPLIT_INTERLOCKS`` with the pocket that receives it; the list is empty,
    so both numbers here are hard zeros.
    """
    cam = cam or design.CameraFrame()
    parts = {"upper_bracket": design.upper_bracket(cam),
             "lower_bracket": design.lower_bracket(cam)}
    rows = {}
    for name, solid in parts.items():
        upper = design.SPLIT_HALVES[name] > 0
        leak = design.split_plane_leak(solid, upper)
        row = dict(side="+Z" if upper else "-Z",
                   allowed_from_mm=(P.SPLIT_GAP if upper else -P.SPLIT_GAP),
                   leak_mm3=0.0, leak_solids=0, leak_bbox_mm=None)
        if leak is not None:
            solids = leak.solids().vals()
            box = leak.val().BoundingBox()
            row.update(leak_mm3=round(float(leak.val().Volume()), 3),
                       leak_solids=len(solids),
                       leak_bbox_mm=[[round(box.xmin, 2), round(box.xmax, 2)],
                                     [round(box.ymin, 2), round(box.ymax, 2)],
                                     [round(box.zmin, 2), round(box.zmax, 2)]])
        rows[name] = row
    assembly = parts["upper_bracket"].intersect(parts["lower_bracket"])
    clash = (0.0 if not assembly.solids().vals()
             else round(float(assembly.val().Volume()), 3))
    return dict(per_part=rows, assembly_intersection_mm3=clash,
                total_leak_mm3=round(sum(r["leak_mm3"] for r in rows.values()), 3),
                declared_interlocks=[dict(i) for i in design.SPLIT_INTERLOCKS],
                note="both halves close onto a rubber liner and both print on this "
                     "face, so material past it collides with the other half and "
                     "lifts the print off its own split face")


def part_pair_intersections(cam=None):
    """G9c: no two payload solids share volume, except the declared pairs.

    Solids, not intent.  Every pair of everything that rides on the wrist -- the two
    brackets, the camera body, holder, lens and connector keep-out, the M4s, the
    nuts and the dowels -- is intersected, and anything non-empty has to appear in
    ``design.DECLARED_OVERLAPS`` with a reason and a bound.  There is one: the
    printed standoff pads stand inside the module's rear envelope, which is the
    point of them.
    """
    cam = cam or design.CameraFrame()
    parts = design.payload_parts(cam)
    declared = {tuple(sorted(d["pair"])): d for d in design.DECLARED_OVERLAPS}
    names = sorted(parts)
    rows, undeclared, over_limit = {}, {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = parts[a].intersect(parts[b])
            if not overlap.solids().vals():
                continue
            volume = round(float(overlap.val().Volume()), 3)
            if volume <= 1e-6:
                continue
            key = tuple(sorted((a, b)))
            row = dict(pair=list(key), volume_mm3=volume)
            spec = declared.get(key)
            if spec is None:
                undeclared[f"{a}^{b}"] = volume
            else:
                row["declared_limit_mm3"] = spec["limit_mm3"]
                row["reason"] = spec["reason"]
                if volume > spec["limit_mm3"]:
                    over_limit[f"{a}^{b}"] = [volume, spec["limit_mm3"]]
            rows[f"{a}^{b}"] = row
    # A missing part shrinks this gate instead of failing it, and one did: the +Y
    # dowel was never built, so the pair sweep tested 36 pairs of 9 parts where
    # there are 45 of 10, and nothing anywhere said a part had gone.
    expected = list(design.PAYLOAD_PART_NAMES)
    return dict(pairs_tested=len(names) * (len(names) - 1) // 2,
                parts_tested=names, parts_expected=expected,
                parts_missing=[n for n in expected if n not in names],
                parts_unexpected=[n for n in names if n not in expected],
                intersecting=rows, undeclared=undeclared, over_limit=over_limit,
                declared=[dict(d, pair=list(d["pair"]))
                          for d in design.DECLARED_OVERLAPS])


def clamped_assembly(cam=None):
    """G9f: the same invariants with the joint CLOSED, not as modelled.

    Every other G9 gate looks at the as-modelled pose, in which the two ears are
    2 x SPLIT_GAP = 1.6 mm apart.  That is not the pose the part is used in: the
    halves have to travel BORE_FIT_CLEARANCE / 2 = 0.3 mm each onto the rubber
    liner, which closes the ears to 1.0 mm, and anything rigid spanning the gap has
    to still fit when they have.

    The called-out 3 x 10 mm dowel did not.  Its holes were cut -DOWEL_DEPTH to
    +DOWEL_DEPTH with DOWEL_DEPTH = 5.0, so each half held 4.2 mm of blind hole and
    a 10 mm pin needed 1.6 mm between the ears -- exactly the open gap.  In the open
    pose the pin sits in free air in both holes and every gate passed it; in the
    closed pose it drove 0.60 mm into the two halves, which is the whole of the
    clamp travel, so the bore never reached the liner and the clamp did not grip.

    Reported per feature: the clash, and how much hole is left past each end of the
    pin once the joint is closed.
    """
    cam = cam or design.CameraFrame()
    upper, lower = design.clamped_pair(cam)
    clash = upper.intersect(lower)
    halves = round(float(clash.val().Volume()), 4) if clash.solids().vals() else 0.0
    rows, worst = {}, 0.0
    for name, pin in design.fastener_envelope(cam.side).items():
        if not name.startswith("dowel"):
            continue
        volume = 0.0
        for part_name, part in (("upper_bracket", upper), ("lower_bracket", lower)):
            hit = pin.intersect(part)
            if hit.solids().vals():
                volume += float(hit.val().Volume())
        # Hole left past each end of the pin, along Z, once the halves have closed.
        spare = round(P.DOWEL_DEPTH - design.CLAMP_TRAVEL_MM - P.DOWEL_LENGTH / 2, 3)
        rows[name] = dict(interference_mm3=round(volume, 4),
                          hole_past_each_end_mm=spare)
        worst = max(worst, volume)
    return dict(travel_per_half_mm=design.CLAMP_TRAVEL_MM,
                modelled_ear_gap_mm=round(2 * P.SPLIT_GAP, 2),
                clamped_ear_gap_mm=round(P.CLAMPED_EAR_GAP, 2),
                dowel_length_mm=P.DOWEL_LENGTH,
                hole_per_half_mm=round(P.DOWEL_DEPTH - P.SPLIT_GAP, 2),
                halves_intersection_mm3=halves,
                per_dowel=rows,
                worst_interference_mm3=round(max(worst, halves), 4),
                note="the bolts are not in this test: a screw's axial position "
                     "relative to each half is set by its thread or its nut and "
                     "follows the joint as it closes, so translating it rigidly "
                     "would invent an interference no assembly has. What is rigid "
                     "across the gap is the two halves and the two pins.")


def mirror_consistency(cam=None):
    """G9g: the native left-hand build is the mirror of the right-hand one.

    ``upper_bracket_left.stl``, the sim's left mesh and the audit's left hand are
    all ``design.mirrored()`` of the right-hand solid, so nothing shipped depends on
    building the left hand natively.  But ``payload_parts(side=-1)`` is a public
    argument and ``params.HANDEDNESS`` invites being set to -1, and that path was
    wrong: the camera plate read its image-v direction off a sign that is only
    correct for the +Y hand, so the tongue, the spine and the M2 lugs all went the
    other way.  4470 mm3 of the native-left part was not in mirror(right) and 3599
    of mirror(right) was not in it, and its swept radius was 106.8 mm against a
    100 mm gate.

    So the two are built and differenced both ways, and the gate is zero.
    """
    cam = cam or design.CameraFrame()
    other = design.CameraFrame(pos=P.CAM_POS, yaw=P.CAM_YAW_DEG,
                               pitch=P.CAM_PITCH_DEG, roll=P.CAM_ROLL_DEG,
                               side=-cam.side)
    rows, worst = {}, 0.0
    for name, build in (("upper_bracket", design.upper_bracket),
                        ("lower_bracket", design.lower_bracket)):
        native = build(other)
        mirror = design.mirrored(build(cam))
        pair = {}
        for label, a, b in (("native_not_in_mirror", native, mirror),
                            ("mirror_not_in_native", mirror, native)):
            diff = a.cut(b)
            volume = (round(float(diff.val().Volume()), 3)
                      if diff.solids().vals() else 0.0)
            pair[label + "_mm3"] = volume
            worst = max(worst, volume)
        pair["volume_mm3"] = round(float(native.val().Volume()), 3)
        rows[name] = pair
    return dict(per_part=rows, worst_difference_mm3=round(worst, 3),
                gated_side=cam.side, native_side=other.side,
                note="design.mirrored() of the gated hand against the same part "
                     "built natively for the other hand, differenced both ways")


def fastener_seating(cam=None, engage_floor=P.G9_FASTENER_ENGAGE_MM):
    """G9d: every bought fastener fits, sits in a real hole, and can be got in.

    Four things, because each one alone passes designs the other three catch:

      * **interference.** The fastener's solid does not intersect any printed part
        it passes through. A clearance hole modelled as an interference is a part
        that cannot be assembled; the closest approach is reported beside it.
      * **the hole.** The sleeve just outside the fastener *does* intersect that
        part, over a real length along the fastener's own axis. Without this a bolt
        drawn in free air -- no hole anywhere near it -- passes the first test
        perfectly, which is how the M3 grub screw's missing hole survived a whole
        revision.
      * **the way in.** Each head, nut and pin is swept from its seated position out
        along the direction it is fitted from, and whatever stands in that volume is
        measured. This is the one that mattered: the camera-side M4's nut passed the
        first two tests and could not be fitted at all, because the strut's root
        stands over its pocket with 937 mm3 of bracket in the way. A dowel has no
        path and is not supposed to -- it is captured before the halves close -- and
        says so.
      * **the thread**, where a joint is tapped rather than nutted.

    ``engage_floor`` is the least engagement along the axis that counts as a hole.
    """
    cam = cam or design.CameraFrame()
    printed = {"upper_bracket": design.upper_bracket(cam),
               "lower_bracket": design.lower_bracket(cam)}
    printed_meshes = {name: to_mesh(solid) for name, solid in printed.items()}
    rows, bad = {}, {}
    for name, probe in design.fastener_probes(cam).items():
        mesh = to_mesh(probe["solid"])
        axis = np.asarray(probe["axis"], float)
        wall = probe["sleeve"].cut(probe["solid"])
        row = dict(holes=list(probe["holes"]), interference_mm3=0.0,
                   engaged_mm={}, clearance_mm={})
        for part_name in probe["holes"]:
            clash = probe["solid"].intersect(printed[part_name])
            volume = (0.0 if not clash.solids().vals()
                      else round(float(clash.val().Volume()), 4))
            row["interference_mm3"] = round(row["interference_mm3"] + volume, 4)
            # Distance from the fastener's surface to the part's: > 0 means no
            # interference, and how much room the fit has.
            gap = float(trimesh.proximity.ProximityQuery(
                printed_meshes[part_name]).on_surface(mesh.vertices)[1].min())
            row["clearance_mm"][part_name] = round(gap, 3)
            # The wall around it: how far along the axis the hole runs.
            engaged = wall.intersect(printed[part_name])
            if not engaged.solids().vals():
                row["engaged_mm"][part_name] = 0.0
                continue
            points = np.array([[v.X, v.Y, v.Z] for v in engaged.val().Vertices()])
            along = points @ axis
            row["engaged_mm"][part_name] = round(float(along.max() - along.min()), 2)
        engaged = max(row["engaged_mm"].values()) if row["engaged_mm"] else 0.0
        row["engaged_best_mm"] = engaged
        # Can each head, nut and pin reach its seat from outside?  Sweep it along
        # the direction it is fitted from and measure what stands in the way.  The
        # sweep starts at the seated position, so anything it hits is material the
        # fastener would have to pass through.
        blocked = {}
        for label, piece, direction in probe["paths"]:
            swept = design.insertion_sweep(piece, direction)
            volume = 0.0
            for part_name in probe["holes"]:
                clash = swept.intersect(printed[part_name])
                if clash.solids().vals():
                    volume += float(clash.val().Volume())
            blocked[label] = round(volume, 2)
        # A fastener with no path -- the two dowels -- is captured between the
        # halves before they close and has nothing to be inserted through; what
        # proves it fits is the engagement test above.
        row["insertion_blocked_mm3"] = blocked or "captured before the halves close"
        # A tapped joint has no nut, so what has to be measured instead is how much
        # printed thread the bolt actually engages.
        side = {"m4_p": 1, "m4_n": -1}.get(name)
        thread = design.m4_thread_engagement(side, cam.side) if side else 0.0
        row["thread_engagement_mm"] = thread
        thread_ok = thread == 0.0 or thread >= P.M4_THREAD_ENGAGE_MIN
        row["ok"] = bool(row["interference_mm3"] <= 1e-3
                         and engaged >= engage_floor
                         and max(blocked.values(), default=0.0) <= 1e-2
                         and thread_ok)
        if not row["ok"]:
            bad[name] = dict(interference_mm3=row["interference_mm3"],
                             engaged_best_mm=engaged,
                             thread_engagement_mm=thread,
                             insertion_blocked_mm3=blocked)
        rows[name] = row
    return dict(per_fastener=rows, failing=bad,
                sleeve_mm=design.FASTENER_SLEEVE,
                engagement_floor_mm=engage_floor,
                note="clearance_mm is the closest approach of the whole fastener to "
                     "the part, so it reads 0.0 wherever a head or a nut is seated "
                     "on a face by design; interference_mm3 is the gate, and it is "
                     "a boolean volume, not a distance. The M2 camera screws are "
                     "probed here only: they are carried as point masses in "
                     "mass_properties and inside the standoff-pad primitives in "
                     "the collision cover, so making them payload parts would "
                     "double-count them.")


def bed_contact(stl_dir, cam=None):
    """G9e: how much of each bracket's split face actually lands on the bed.

    Measured on the exported STL, in its print pose, as flat downward-facing area at
    Z = 0 -- which is what a slicer sees and what the README's "prints split-face
    down, supports: none" claims.  The upper bracket as shipped rested on a single
    lowest vertex: 34 mm2 of cross-section at Z = 1.3 mm under a 1182 mm2 split face
    that floated 2.65 mm up.
    """
    cam = cam or design.CameraFrame()
    faces = {}
    for name, sign in design.SPLIT_HALVES.items():
        solid = (design.upper_bracket(cam) if sign > 0 else design.lower_bracket(cam))
        z = sign * P.SPLIT_GAP
        area = sum(f.Area() for f in solid.faces().vals()
                   if abs(f.Center().z - z) < 1e-6 and abs(f.normalAt().z) > 0.999)
        faces[name] = float(area)
    # The exported files: the upper bracket is one part per hand, the lower is shared.
    files = {"upper_bracket_right": "upper_bracket",
             "upper_bracket_left": "upper_bracket",
             "lower_bracket": "lower_bracket"}
    rows, bad = {}, {}
    for stem, part in files.items():
        path = stl_dir / f"{stem}.stl"
        if not path.exists():
            continue
        mesh = trimesh.load(path)
        normals, areas = mesh.face_normals, mesh.area_faces
        flat_down = normals[:, 2] < -0.999
        on_bed = mesh.triangles[:, :, 2].max(axis=1) <= P.LAYER_H
        contact = float(areas[flat_down & on_bed].sum())
        split = faces[part]
        rows[stem] = dict(
            bed_contact_mm2=round(contact, 1),
            split_face_mm2=round(split, 1),
            fraction=round(contact / split, 3) if split else 0.0,
            lowest_z_mm=round(float(mesh.bounds[0, 2]), 3),
            footprint_mm=[round(float(mesh.extents[0]), 1),
                          round(float(mesh.extents[1]), 1)])
        if rows[stem]["fraction"] < P.G9_BED_CONTACT_FRACTION:
            bad[stem] = rows[stem]["fraction"]
    return dict(per_file=rows, failing=bad,
                threshold=P.G9_BED_CONTACT_FRACTION,
                note="flat downward faces within one layer of Z = 0 in the "
                     "exported print pose, against the area of the part's own "
                     "split face as built")


def g7_module_fit(cam=None):
    """Volume of printed material inside the bought module's own envelope.

    Has to be zero, and was not: the strut prism was never trimmed, so 375 mm2 of
    bracket stood inside the PCB's footprint and material reached 1 mm past the
    lens's front vertex.  Nothing caught it, because ``payload_parts`` treats the
    bracket and the camera as two payload members and no gate tested them against
    each other.
    """
    cam = cam or design.CameraFrame()
    # Minus the standoff pads: the PCB stands on them, so they are the one thing
    # that belongs in the space between the plate's front face and the board.
    keepout = design.module_keepout(cam, grow=0.0).cut(design.standoff_pads(cam))
    rows, total = {}, 0.0
    for name, part in (("upper_bracket", design.upper_bracket(cam)),
                       ("lower_bracket", design.lower_bracket(cam))):
        overlap = part.intersect(keepout)
        volume = 0.0 if not overlap.solids().vals() else float(overlap.val().Volume())
        rows[name] = round(volume, 3)
        total += volume
    # And the space the README promises behind the PCB, which must be the space the
    # plate actually leaves.
    return dict(intersection_mm3=round(total, 3), per_part_mm3=rows,
                rear_envelope_mm=round(-P.BODY_BACK, 2),
                rear_envelope_in_pocket_mm=round(
                    -P.BODY_BACK + P.PLATE_RELIEF_DEPTH, 2),
                pocket_mm=P.PLATE_RELIEF,
                standoff_mm=P.STANDOFF_H,
                note="rear components may reach BODY_BACK anywhere, and "
                     "PLATE_RELIEF_DEPTH further inside the central pocket; those "
                     "two numbers are the plate, not a wish")


def g7_plate_slots(cam=None, steps=720, march=8.0):
    """Plate material between each M2 slot and the plate's outside, measured.

    Measured in the plate's own plane rather than derived from a formula: the
    binding wall is not the diagonal one to the corner (the formula the first cut
    used, which read 2.94 mm where the truth was 1.70), and once the four lugs are
    there it is not any single parameter either.  So: walk the slot's outline, step
    outward along its own normal, and record where the solid ends.
    """
    cam = cam or design.CameraFrame()
    mesh = to_mesh(design.upper_bracket(cam))
    query = trimesh.proximity.ProximityQuery(mesh)
    w = design.plate_rear_offset() + P.PLATE_T / 2
    half = (P.M2_PITCH_MIN + P.M2_PITCH_MAX) / 4.0
    length = design.m2_slot_length()
    worst, at = np.inf, None
    for su, sv, _, _ in design._m2_slot_centres():
        # The slot: a stadium of width 2 * M2_CAP_R along the 45 deg diagonal.
        axis = np.array([su, sv]) / math.sqrt(2.0)
        perp = np.array([-axis[1], axis[0]])
        centre = np.array([su * half, sv * half])
        outline, normals = [], []
        for t in np.linspace(-1, 1, steps // 4):
            for s in (-1, 1):
                outline.append(centre + t * (length / 2 - P.M2_CAP_R) * axis
                               + s * P.M2_CAP_R * perp)
                normals.append(s * perp)
        for a in np.linspace(0, 2 * math.pi, steps // 2, endpoint=False):
            direction = math.cos(a) * axis + math.sin(a) * perp
            for s in (-1, 1):
                # Only the outer half of each end cap is boundary; the inner half
                # faces back along the slot, into its own void.
                if s * (direction @ axis) < 0:
                    continue
                outline.append(centre + s * (length / 2 - P.M2_CAP_R) * axis
                               + P.M2_CAP_R * direction)
                normals.append(direction)
        for uv, n in zip(outline, normals):
            steps_out = np.arange(0.05, march, 0.05)
            points = np.array([cam.point(*(uv + d * n), w) for d in steps_out])
            inside = query.signed_distance(points) > 0
            if not inside[0]:
                continue        # a normal that points into the slot, not out of it
            if not inside.all():
                first = int(np.argmin(inside))
                if steps_out[first] < worst:
                    worst, at = float(steps_out[first]), uv
    return dict(wall_mm=round(worst, 2),
                thinnest_at_uv_mm=[round(v, 1) for v in at] if at is not None else None,
                slot_length_mm=round(length, 2),
                cap_radius_mm=P.M2_CAP_R, lug_radius_mm=P.M2_LUG_R,
                pitch_range_mm=[P.M2_PITCH_MIN, P.M2_PITCH_MAX],
                method=f"the slot's outline walked in {steps} steps, marched "
                       f"outward along its own normal in 0.05 mm steps against the "
                       f"built solid at mid-plate depth")


def gauge_fidelity():
    """Does the rail-key gauge reproduce the key pocket it exists to check?

    The first cut of it had ``depth = RAIL_BACK_X - KEY_FRONT_X``, i.e. -10.65, and
    ``box()`` sorts its corners, so nothing complained: the gauge came out 6.65 mm
    long with a 5.65 mm pocket and a 1.0 mm web -- 47 % short of the engagement, on
    the one part the README tells you to print first.
    """
    solid = design.rail_key_gauge()
    mesh = to_mesh(solid)
    lo, hi = mesh.bounds
    # Pocket depth: along X, where does material stop on the pocket's centreline?
    query = trimesh.proximity.ProximityQuery(mesh)
    xs = np.linspace(lo[0], hi[0], 400)
    probe = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    solid_at = query.signed_distance(probe) > 0
    depth = float(xs[solid_at].min() - lo[0]) if solid_at.any() else 0.0
    want_depth = design.key_engagement()
    across = 2 * P.KEY_INNER_Y
    # Across-flats: the pocket's Y extent at mid-depth
    ys = np.linspace(-P.YOKE_OUTER_Y, P.YOKE_OUTER_Y, 1200)
    mid = np.column_stack([np.full_like(ys, lo[0] + depth / 2), ys,
                           np.zeros_like(ys)])
    empty = query.signed_distance(mid) <= 0
    got_across = float(ys[empty].max() - ys[empty].min()) if empty.any() else 0.0
    ok = (abs(depth - want_depth) < 0.3 and abs(got_across - across) < 0.3
          and mesh.is_watertight)
    return dict(ok=bool(ok),
                gauge_pocket_depth_mm=round(depth, 2),
                key_engagement_mm=round(want_depth, 2),
                gauge_across_flats_mm=round(got_across, 2),
                key_across_flats_mm=round(across, 2),
                gauge_size_mm=[round(v, 2) for v in mesh.extents],
                web_mm=P.GAUGE_WEB,
                note="the pocket's back wall is also the axial stop plane, so the "
                     "one part checks the clocking fit and the datum together")


def driver_access(cam=None, reach=60.0, step=0.5):
    """Clear travel above each bolt's head, on the side it is driven from.

    A bolt you cannot get a key onto is not a fastening.  The strut's root stands
    over the camera-side ear and there is no direction at all with clear shaft above
    it, which is why that bolt is turned over (design.head_from_above); this is the
    number that says so, per bolt, measured on the built solids.
    """
    cam = cam or design.CameraFrame()
    x = (P.EAR_X0 + P.EAR_X1) / 2
    meshes = {"upper": to_mesh(design.upper_bracket(cam)),
              "lower": to_mesh(design.lower_bracket(cam))}
    queries = {k: trimesh.proximity.ProximityQuery(m) for k, m in meshes.items()}
    out = {}
    for side in (-1, 1):
        above = design.head_from_above(side, cam.side)
        sign = 1.0 if above else -1.0
        start = sign * (P.EAR_HALF_Z + 0.2)
        travel = np.arange(step, reach, step)
        points = np.array([[x, side * P.BOLT_Y, start + sign * t] for t in travel])
        free = reach
        for name, query in queries.items():
            inside = query.signed_distance(points) > 0
            if inside.any():
                free = min(free, float(travel[int(np.argmax(inside))]))
        out["+Y" if side > 0 else "-Y"] = dict(
            driven_from="above" if above else "below",
            clear_travel_mm=round(free, 1), capped_at_mm=reach)
    return out


def fastener_stack(cam=None):
    """Required screw lengths and driver access, from the solids not from the BOM."""
    cam = cam or design.CameraFrame()
    grip = 2 * P.EAR_HALF_Z - P.COUNTERBORE_DEPTH
    m2 = P.PLATE_T + P.STANDOFF_H + P.PCB_T + 1.6
    joints = {}
    for side in (1, -1):
        label = "+Y" if side > 0 else "-Y"
        tapped = design.tapped_joint(side, cam.side)
        joints[label] = dict(
            kind="tapped into the upper ear" if tapped else "steel nut",
            driven_from=("above" if design.head_from_above(side, cam.side)
                         else "below"),
            protrusion_mm=design.m4_protrusion(side, cam.side),
            thread_engagement_mm=design.m4_thread_engagement(side, cam.side),
            hole_left_past_the_tip_mm=design.m4_tip_clearance(side, cam.side),
            nut_entry=(None if tapped else
                       "slides in through the ear's outboard end face"))
    return dict(m4_head_underside_to_far_face_mm=round(grip, 1),
                m4_called_out_mm=P.M4_LENGTH,
                m4_protrusion_mm=round(design.m4_protrusion(), 1),
                m4_counterbore_depth_mm=P.COUNTERBORE_DEPTH,
                m4_head_height_mm=4.0,
                m4_tap_drill_mm=P.M4_TAP_DIA,
                m4_thread_engagement_floor_mm=P.M4_THREAD_ENGAGE_MIN,
                joints=joints,
                m4_driven_from={k: v["driven_from"] for k, v in joints.items()},
                driver_access=driver_access(cam),
                m2_stack_mm=round(m2, 1), m2_called_out_mm=12.0,
                nut_pocket="hex pocket plus a channel out through the ear's "
                           "outboard end face, so the nut slides in after "
                           "printing and no face above it can roof it",
                note="the M4 counterbore is deeper than an M4 SHCS head, so the head "
                     "sits flush; the M2 stack is plate + standoff + PCB + nut. The "
                     "camera-side bolt is turned over because the strut's root stands "
                     "over that ear -- measured, no direction above it has 70 mm of "
                     "clear shaft -- so it is driven from underneath, and for the "
                     "same reason its nut has nowhere to go: that joint is tapped.")


def primitive_cover(meshes, primitives, tolerance=1.5, volume_samples=400000,
                    seed=11):
    """Check the collision primitives cover the payload without hulling it.

    Every payload surface sample must lie inside the union of the primitives
    (within `tolerance`), and the union's volume -- measured by Monte Carlo, so
    overlaps are not double counted -- must not run away from the payload's.
    """
    points = payload_points(meshes, step=2.0)
    inside = _inside_primitives(points, primitives, tolerance)
    missed = points[~inside]
    corners = []
    for prim in primitives:
        centre = np.array(prim["pos"]) * 1000.0
        rot = np.array(prim["rot"])
        extent = np.array(prim["size"]) * 1000.0
        half = (extent / 2.0 if prim["type"] == "box"
                else np.array([extent[0], extent[0], extent[1] / 2.0]))
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1)
                          for c in (-1, 1)])
        corners.append(centre + (signs * half) @ rot.T)
    corners = np.concatenate(corners)
    low, high = corners.min(0) - 1.0, corners.max(0) + 1.0
    rng = np.random.default_rng(seed)
    probes = rng.uniform(low, high, (volume_samples, 3))
    fraction = float(_inside_primitives(probes, primitives).mean())
    union_volume = fraction * float(np.prod(high - low))
    payload_volume = float(sum(m.volume for m in meshes.values()))
    return dict(surface_samples=len(points), covered=int(inside.sum()),
                uncovered=int((~inside).sum()),
                worst_miss_mm=None if not len(missed) else
                [round(v, 1) for v in missed[0]],
                tolerance_mm=tolerance, n_primitives=len(primitives),
                union_volume_cm3=round(union_volume / 1000.0, 2),
                payload_volume_cm3=round(payload_volume / 1000.0, 2),
                volume_ratio=round(union_volume / payload_volume, 3),
                volume_method=f"Monte Carlo, {volume_samples} samples, seed {seed}")


def primitive_bottle_clearance(primitives, samples=P.G1_BOTTLE_SAMPLES,
                               seed=P.BOTTLE_SEED):
    """G1 again, this time on the simulator primitives rather than the CAD."""
    clouds = []
    for prim in primitives:
        centre = np.array(prim["pos"]) * 1000.0
        rot = np.array(prim["rot"])
        if prim["type"] == "box":
            half = np.array(prim["size"]) * 1000.0 / 2.0
            grid = [np.linspace(-h, h, max(2, int(2 * h / 2.0) + 1)) for h in half]
            a, b, c = np.meshgrid(*grid, indexing="ij")
            local = np.stack([a.ravel(), b.ravel(), c.ravel()], 1)
        else:
            radius, length = prim["size"][0] * 1000.0, prim["size"][1] * 1000.0
            angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            axial = np.linspace(-length / 2, length / 2, max(2, int(length / 2) + 1))
            local = np.array([[radius * math.cos(t), radius * math.sin(t), z]
                              for t in angles for z in axial])
        clouds.append(centre + local @ rot.T)
    points = np.concatenate(clouds)
    rng = np.random.default_rng(seed)
    bottles = [sample_bottle(rng) for _ in range(samples)] + corner_bottles()
    worst, collisions = np.inf, 0
    for label, z_rel, depth, pitch in GRASP_CANDIDATES:
        for b in bottles:
            if label == "neck" and not neck_graspable(b, pitch):
                continue
            gap = bottle_clearance(points, b, z_rel(b), depth, pitch)
            worst = min(worst, gap)
            collisions += int(gap < 0)
    return dict(points=len(points), min_clearance_mm=round(worst, 2),
                collisions=collisions)
