"""Print files, editable CAD, simulator collision primitives and camera_spec.

    python -m mount.export            # WRITES stl/ step/ collision/ camera_spec.json
    python -m mount.export --check    # writes nothing: regenerates into a temp
                                      # directory and diffs; exit 1 on drift

``stl/`` is pre-oriented on the bed.  Both hands are exported: the right-hand
part is the one the checks are run on, the left-hand one is its mirror in Y.

Everything written here is deterministic, byte for byte, which is what makes
``--check`` mean something: the STEP writer's ``FILE_NAME`` timestamp is
normalised away (``STEP_EPOCH``), the tessellation tolerances are fixed
arguments, and no sampler runs on this path.  ``set_output_root`` is how the
check run redirects; nothing in this module opens a path any other way.
"""
import argparse
import filecmp
import json
import math
import shutil
import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from . import checks, design, params as P, v4ref

ROOT = Path(__file__).resolve().parents[1]
STL_DIR = ROOT / "stl"
STEP_DIR = ROOT / "step"
COLLISION_DIR = ROOT / "collision"
_OUT = ROOT


def set_output_root(path):
    """Point every writer in this module somewhere else; returns the old root."""
    global _OUT, STL_DIR, STEP_DIR, COLLISION_DIR
    previous = _OUT
    _OUT = Path(path)
    STL_DIR, STEP_DIR, COLLISION_DIR = _OUT / "stl", _OUT / "step", _OUT / "collision"
    return previous


def output_root():
    return _OUT


# The files ``python -m mount.export`` owns; ``--check`` compares exactly these.
WRITTEN_FILES = (
    ["camera_spec.json", "collision/payload.json", "mount_on_gripper.glb",
     "step/assembly_right_hand.step", v4ref.CACHE_NAME]
    + [f"stl/{stem}.stl" for stem in
       ("upper_bracket_right", "upper_bracket_left", "lower_bracket",
        "bore_gauge", "rail_key_gauge", "m2_standoff_3mm")]
    + [f"step/{stem}.step" for stem in
       ("upper_bracket_right", "upper_bracket_left", "lower_bracket",
        "bore_gauge", "rail_key_gauge", "m2_standoff_3mm")])

# OCC stamps the wall clock into every STEP header, so two identical runs differ
# in one line and a diff is useless.  A fixed string there is what turns --check
# into a real comparison; the real date of the file is in git.
STEP_EPOCH = "1970-01-01T00:00:00"


def _normalise_step(path):
    """Replace the STEP header's wall-clock timestamp with a constant."""
    path = Path(path)
    text = path.read_text()
    marker = "FILE_NAME("
    start = text.find(marker)
    if start < 0:
        return path
    # FILE_NAME('<name>','<timestamp>',...): the timestamp is the second quoted
    # field after the marker.
    cursor, fields = start + len(marker), []
    for _ in range(4):
        cursor = text.find("'", cursor)
        if cursor < 0:
            return path
        fields.append(cursor)
        cursor += 1
    path.write_text(text[:fields[2] + 1] + STEP_EPOCH + text[fields[3]:])
    return path


# --------------------------------------------------------------------------
# print orientation
# --------------------------------------------------------------------------

def print_pose(solid, orientation):
    """Lay a part on the bed so its load-bearing layers run the right way.

    Both brackets print split-face down, which is their one big flat face and
    puts the clamp hoop, the locating yoke and the strut all in-plane.

    ``as_modelled``  the upper bracket: its split face is already the lowest.
    ``flip_x``       the lower bracket: turn it over onto its split face.
    ``flat``         gauges and small parts, as modelled.
    ``lay_flat``     the bore gauge: modelled as a ring in the YZ plane, which
                     "as modelled" would print standing on a 3 x 4 mm edge 40.8 mm
                     tall.  Turned 90 deg about Y it lies down on its 69.6 x 40.8
                     face, 3 mm high, with no overhang and no brim.
    """
    if orientation == "flip_x":
        solid = solid.rotate((0, 0, 0), (1, 0, 0), 180)
    elif orientation == "lay_flat":
        solid = solid.rotate((0, 0, 0), (0, 1, 0), 90)
    bounds = solid.val().BoundingBox()
    return solid.translate((0, 0, -bounds.zmin))


def export_parts(cam=None):
    """Write every STL and STEP, and return a per-file report."""
    cam = cam or design.CameraFrame()
    STL_DIR.mkdir(exist_ok=True)
    STEP_DIR.mkdir(exist_ok=True)
    report = {}
    parts = dict(design.printed_parts(cam))
    parts["upper_bracket_left"] = (design.mirrored(parts["upper_bracket_right"][0]),
                                   "as_modelled")
    for name, (solid, orientation) in parts.items():
        shape = solid.val()
        assert shape.isValid(), f"{name}: invalid solid"
        assert len(solid.solids().vals()) == 1, f"{name}: not a single solid"
        cq.exporters.export(solid, str(STEP_DIR / f"{name}.step"))
        _normalise_step(STEP_DIR / f"{name}.step")
        oriented = print_pose(solid, orientation)
        path = STL_DIR / f"{name}.stl"
        cq.exporters.export(oriented, str(path), tolerance=0.025, angularTolerance=0.08)
        mesh = trimesh.load(path)
        mesh.merge_vertices(digits_vertex=2)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
        mesh.export(path)
        report[name] = dict(orientation=orientation,
                            volume_cm3=round(float(shape.Volume()) / 1000.0, 2))
    # No colours on the assembly.  They were written as literal sRGB triples and
    # come back out of OCC's own XCAF reader as (0.71, 0.30, 0.03) and (0.01, 0.26,
    # 0.26) -- a linear/sRGB conversion applied on one side of the round trip only,
    # so the numbers in the file mean nothing in particular, and 300 entities of
    # STYLED_ITEM and MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION have
    # to be read to find that out. The two bodies are named `upper` and `lower` and
    # that is the whole information content; the renders carry the colours.
    assembly = cq.Assembly(name="g1_wrist_camera_mount")
    assembly.add(parts["upper_bracket_right"][0], name="upper")
    assembly.add(parts["lower_bracket"][0], name="lower")
    assembly.export(str(STEP_DIR / "assembly_right_hand.step"))
    _normalise_step(STEP_DIR / "assembly_right_hand.step")
    return report


# --------------------------------------------------------------------------
# camera_spec.json
# --------------------------------------------------------------------------

def camera_measurements(cam=None, meshes=None, occlusion=None, forward=None):
    """The six numbers ``camera_spec``'s lens recommendation argues from.

    One code path, and it is this one.  Before, ``verify.py`` computed them inline
    and handed them down, while ``python -m mount.export`` did not -- so the same
    file said "cuts the fraction of the image a held bottle covers to 0.28" after a
    verify run and "to not measured" after an export, and whichever ran last won.

    ``occlusion`` and ``forward`` are the reports verify.py has just taken; passing
    them in avoids re-casting a few hundred thousand rays, and leaving them out
    makes this compute them, which is what the export entry point does.
    """
    cam = cam or design.CameraFrame()
    meshes = meshes if meshes is not None else checks.payload_meshes(cam)
    lenses = (("wide", P.FOV_RECOMMENDED, "recommended_wide"),
              ("narrow", P.FOV_ALTERNATIVE, "alternative_standard"))
    out = {}
    for tag, fov, label in lenses:
        occ = (occlusion or {}).get(label)
        if occ is None:
            occ = checks.g3_occlusion(cam, fov)
            occ["self_and_gripper"] = checks.self_occlusion(cam, meshes, fov)
        body = [k for k in occ if k.startswith(("body-high", "body-mid"))]
        out[f"bottle_fraction_{tag}"] = round(
            sum(occ[k]["mean_image_fraction"] for k in body) / len(body), 2)
        out[f"mount_fraction_{tag}"] = occ["self_and_gripper"]["mount_image_fraction"]
        view = (forward or {}).get(label) or checks.g3_forward_view(cam, fov)
        out[f"margin_{tag}"] = view["gated_margin_deg"]
    return out


def camera_spec(cam=None, measured=None):
    """Lens pose in gripper_link, in metres, plus the OpenCV extrinsic.

    ``measured`` is ``camera_measurements``' six numbers, so the lens
    recommendation argues from a run rather than from a string typed once: the
    first revision's said "at the cost of 6 % of the frame being the mount's own
    strut", which was a cast with the gripper left out as an occluder and is 50x
    the truth.  Passing None measures them here rather than writing placeholders.
    """
    cam = cam or design.CameraFrame()
    m = camera_measurements(cam) if measured is None else dict(measured)

    def num(key, fmt="{:.2f}"):
        return "not measured" if m.get(key) is None else fmt.format(m[key])
    # OpenCV camera convention: z forward along the optical axis, x right in
    # the image, y down.  The rotation's columns are those three axes.
    rotation = np.column_stack([cam.right, -cam.up, cam.axis])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = cam.pupil / 1000.0
    return {
        "frame": "gripper_link",
        "units": "metres",
        "handedness": "right" if cam.side > 0 else "left",
        # Nine decimals, all of them, so that a simulator building its camera from
        # the pupil and the axes and comparing against T_gripper_camera_opencv
        # disagrees by rounding alone at 1e-9 rather than 8e-7.  At six decimals the
        # test's own 1e-6 tolerance was 78 % consumed by rounding, and one more
        # rounding change would have flipped it.
        "lens_entrance_pupil_m": [round(v / 1000.0, 9) for v in cam.pupil],
        "pcb_front_face_centre_m": [round(v / 1000.0, 9) for v in cam.pos],
        "optical_axis": [round(v, 9) for v in cam.axis],
        "image_up": [round(v, 9) for v in cam.up],
        "image_right": [round(v, 9) for v in cam.right],
        "T_gripper_camera_opencv": [[round(v, 9) for v in row] for row in transform],
        "pitch_below_tool_axis_deg": round(
            math.degrees(math.asin(-cam.axis[2])), 2),
        "yaw_inboard_deg": cam.yaw,
        "roll_deg": cam.roll,
        "lens": {
            "recommended": {
                "name": "M12 wide, ~2.1 mm",
                "fov_deg": list(P.FOV_RECOMMENDED),
                "fov_note": "the horizontal angle is not free: with square pixels "
                            "on this 4:3 frame it follows from the vertical one, "
                            "and MuJoCo drives a fixed camera from fovy alone. A "
                            "nominal 110 x 85 deg would need a 1.56:1 sensor; on "
                            "640 x 480 the same lens gives "
                            f"{P.FOV_RECOMMENDED[0]:.1f} x "
                            f"{P.FOV_RECOMMENDED[1]:.0f}.",
                "why": f"cuts the fraction of the image a held bottle covers to "
                       f"{num('bottle_fraction_wide')} from "
                       f"{num('bottle_fraction_narrow')} on body grasps and leaves "
                       f"{num('margin_wide', '{:.1f}')} deg of framing margin "
                       f"instead of {num('margin_narrow', '{:.1f}')}. It costs "
                       f"almost no frame: the mount is "
                       f"{num('mount_fraction_wide', '{:.1%}')} of the wide image "
                       f"once the gripper is in front of it. It does NOT recover "
                       f"the bottle mouth: on a body grasp the mouth is behind the "
                       f"camera plane, so no field of view reaches it.",
            },
            "alternative": {
                "name": "M12 standard, ~3.6 mm",
                "fov_deg": list(P.FOV_ALTERNATIVE),
                "why": f"more pixels on the grasp zone, but only "
                       f"{num('margin_narrow', '{:.1f}')} deg of framing margin and "
                       f"a held bottle covers {num('bottle_fraction_narrow')} of "
                       f"the image on body grasps.",
            },
        },
        "resolution_px": list(P.RESOLUTION),
        "sensor_module": {
            "kind": "32 x 32 mm UVC board camera, M2 holes on a 26-30 mm square",
            "mass_g": P.CAMERA_MASS_G,
        },
        "payload_mass_g": None,        # filled in by write_camera_spec
        "mouth_visibility": "On neck grasps the bottle mouth is in frame with "
                            "either lens. On body-high and body-mid grasps it "
                            "is behind the camera plane and no lens sees it: "
                            "close the pour loop on the base camera, or bias "
                            "the planner to neck grasps.",
        "notes": "Pose is nominal CAD, not a calibration. Calibrate extrinsics "
                 "on the real arm; this is the starting guess and the repeatable "
                 "mechanical datum that makes recalibration unnecessary after a "
                 "camera swap.",
    }


# --------------------------------------------------------------------------
# collision primitives
# --------------------------------------------------------------------------

def _oriented_box(centre, axes, half, name):
    """`axes` are the box's own three unit vectors; `rot` stores them as columns.

    A box is symmetric about each of its own mid-planes, so if the axes come in
    left-handed (the camera frame's `right` is `-side * R[:, 1]`, which flips
    the handedness for one hand) the first one is negated.  That names the same
    box and keeps `rot` a proper rotation, which is what a simulator needs to
    turn it into a pose -- MuJoCo's `mju_mat2Quat` on a reflection is silently
    wrong.
    """
    rot = np.column_stack([np.asarray(a, float) for a in axes])
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1.0
    assert np.linalg.det(rot) > 0.99, f"{name}: axes are not orthonormal"
    return dict(name=name, type="box",
                pos=[round(v / 1000.0, 6) for v in centre],
                rot=[[round(v, 6) for v in row] for row in rot],
                size=[round(2 * v / 1000.0, 6) for v in half])


def _cylinder(p0, p1, radius, name):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    axis = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    assert np.linalg.det(np.column_stack([e1, e2, axis])) > 0.99, f"{name}: bad frame"
    return dict(name=name, type="cylinder",
                pos=[round(v / 1000.0, 6) for v in (p0 + p1) / 2],
                rot=[[round(v, 6) for v in row]
                     for row in np.column_stack([e1, e2, axis])],
                size=[round(radius / 1000.0, 6), round(length / 1000.0, 6)])


def collision_primitives(cam=None):
    """Convex pieces that cover the payload without hulling over free space.

    A convex hull of the bracket would swallow the whole corridor between the clamp
    band and the camera -- which is exactly where a held bottle goes -- so the
    payload is covered by slabs that follow it instead.  ``checks.primitive_cover``
    proves the cover and measures how much free space it claims.

    The count is higher than the first revision's 25 because the boxes are smaller:
    24 band sectors instead of 12, four bolt-ear halves instead of two full-height
    boxes, three slabs per stop horn and six along the tapered strut.  That is the
    whole trade -- more geoms, less claimed air.
    """
    cam = cam or design.CameraFrame()
    out = []
    identity = np.eye(3)
    # Clamp band: tangential slabs around the annulus rather than one solid
    # cylinder.  A cylinder would be 9x the band's real volume and would claim the
    # bore, which belongs to the gripper.  A slab's inner face is a chord, so it has
    # to reach the bore's apothem, not its radius, or the bore surface pokes out
    # between slabs -- which is why more sectors claim less air.
    sectors = 24
    outer = P.CLAMP_OD / 2
    inner = P.BORE_DIA / 2 * math.cos(math.pi / sectors) - 0.2
    for k in range(sectors):
        angle = 2 * math.pi * k / sectors
        radial = np.array([0.0, math.cos(angle), math.sin(angle)])
        tangent = np.array([0.0, -math.sin(angle), math.cos(angle)])
        mid = (inner + outer) / 2
        out.append(_oriented_box(
            [(P.CLAMP_X0 + P.CLAMP_X1) / 2, mid * radial[1], mid * radial[2]],
            [np.array([1.0, 0.0, 0.0]), radial, tangent],
            [(P.CLAMP_X1 - P.CLAMP_X0) / 2, (outer - inner) / 2,
             outer * math.tan(math.pi / sectors)],
            f"clamp_band_{k}"))
    # Bolt ears, one box per half.  Inboard of Y = CLAMP_OD / 2 the ear is inside the
    # band's own slabs, and the 1.6 mm of air between the halves is nobody's.
    for side in (-1, 1):
        tag = "p" if side > 0 else "n"
        for half, z0, z1 in (("upper", P.SPLIT_GAP, P.EAR_HALF_Z + 1.0),
                             ("lower", -P.EAR_HALF_Z - 0.4, -P.SPLIT_GAP)):
            out.append(_oriented_box(
                [(P.EAR_X0 + P.EAR_X1) / 2,
                 side * (P.CLAMP_OD / 2 + P.EAR_Y1) / 2, (z0 + z1) / 2],
                identity,
                [(P.EAR_X1 - P.EAR_X0) / 2, (P.EAR_Y1 - P.CLAMP_OD / 2) / 2,
                 (z1 - z0) / 2],
                f"bolt_ear_{tag}_{half}"))
    # The M4's thread protrudes past the far face; a cylinder, not a taller ear box.
    # Which face depends on which way that bolt is driven -- see design.head_from_above.
    for side in (-1, 1):
        _, tip = design.m4_span_z(side, cam.side)
        far = -P.EAR_HALF_Z if design.head_from_above(side, cam.side) else P.EAR_HALF_Z
        out.append(_cylinder(
            [(P.EAR_X0 + P.EAR_X1) / 2, side * P.BOLT_Y, tip],
            [(P.EAR_X0 + P.EAR_X1) / 2, side * P.BOLT_Y, far],
            P.BOLT_DIA / 2, f"m4_thread_{'p' if side > 0 else 'n'}"))
    # Locating yoke: three slabs per stop horn following its taper, plus the key ear.
    # One slab over the horn would claim 2 cm3 of the gap beside the rail.
    for side in (-1, 1):
        tag = "p" if side > 0 else "n"
        steps = 3
        for k in range(steps):
            x0 = P.YOKE_X0 + (P.RAIL_BACK_X - P.YOKE_X0) * k / steps
            x1 = P.YOKE_X0 + (P.RAIL_BACK_X - P.YOKE_X0) * (k + 1) / steps
            # the horn's outline runs from 44 mm at YOKE_X0 to YOKE_OUTER_Y at the datum
            far = 44.0 + (P.YOKE_OUTER_Y - 44.0) * (k + 1) / steps
            out.append(_oriented_box(
                [(x0 + x1) / 2, side * (P.STOP_Y0 + far) / 2, 0.0], identity,
                [(x1 - x0) / 2, (far - P.STOP_Y0) / 2, P.YOKE_HALF_Z],
                f"stop_horn_{tag}_{k}"))
        out.append(_oriented_box(
            [(P.RAIL_BACK_X + P.KEY_FRONT_X) / 2,
             side * (P.KEY_INNER_Y + P.YOKE_OUTER_Y) / 2, 0.0],
            identity,
            [(P.KEY_FRONT_X - P.RAIL_BACK_X) / 2,
             (P.YOKE_OUTER_Y - P.KEY_INNER_Y) / 2, P.KEY_HALF_Z],
            f"rail_key_{tag}"))
    # strut-to-ear blend block
    out.append(_oriented_box(
        [(P.BLEND_X[0] + P.BLEND_X[1]) / 2,
         cam.side * (P.BLEND_Y[0] + P.BLEND_Y[1]) / 2,
         (P.SPLIT_GAP + P.BLEND_Z) / 2], identity,
        [(P.BLEND_X[1] - P.BLEND_X[0]) / 2,
         (P.BLEND_Y[1] - P.BLEND_Y[0]) / 2, (P.BLEND_Z - P.SPLIT_GAP) / 2],
        "strut_blend"))
    # Strut: six boxes along it, each sized to the local section of the taper.
    root, tip, axis, length, wide = design._strut_geometry(cam)
    tall = np.cross(axis, wide)
    steps = 6
    for k in range(steps):
        lo, hi = k / steps, (k + 1) / steps
        w = P.STRUT_W + (P.STRUT_TIP_W - P.STRUT_W) * lo
        h = P.STRUT_H + (P.STRUT_TIP_H - P.STRUT_H) * lo
        out.append(_oriented_box(root + (lo + hi) / 2 * length * axis,
                                 [wide, tall, axis],
                                 [w / 2 + 0.25, h / 2 + 0.25,
                                  (hi - lo) * length / 2 + 0.25],
                                 f"strut_{k}"))
    lug_centre, _, lug_axes, lug_half, _ = design._ziptie_geometry(cam)
    out.append(_oriented_box(lug_centre, lug_axes,
                             [v + 0.25 for v in lug_half], "ziptie_lug"))
    # Camera plate: the plate itself, then its four standoff pads, then the tongue
    # and the rear spine.  One box over plate plus pads claimed 3 cm3 of the air the
    # PCB sits in.
    rear = design.plate_rear_offset()
    plate_half = P.M2_PITCH_MAX / 2 + P.M2_LUG_R + 0.3
    out.append(_oriented_box(cam.pos + (rear + P.PLATE_T / 2) * cam.axis,
                             [cam.right, cam.up, cam.axis],
                             [plate_half, plate_half, P.PLATE_T / 2],
                             "camera_plate"))
    pad_len = design.m2_slot_length() + 2 * P.MIN_WALL
    for i, (su, sv, half_lo, half_hi) in enumerate(design._m2_slot_centres()):
        half = (half_lo + half_hi) / 2
        diag = np.array([su, sv]) / math.sqrt(2.0)
        e1 = diag[0] * cam.right + diag[1] * cam.up
        e2 = np.cross(e1, cam.axis)
        out.append(_oriented_box(
            cam.pos + (su * half) * cam.right + (sv * half) * cam.up
            + (rear + P.PLATE_T + P.STANDOFF_H / 2) * cam.axis,
            [e1, e2, cam.axis],
            [pad_len / 2 + 0.3, P.M2_LUG_R + 0.3, P.STANDOFF_H / 2 + 0.3],
            f"standoff_pad_{i}"))
    tip_u, tip_v = design.plate_tip_uv()
    lo, hi = design.plate_tongue_extent(spine=True)
    unit = np.array([tip_u, tip_v]) / np.hypot(tip_u, tip_v)
    along = unit[0] * cam.right + unit[1] * cam.up
    across = np.cross(along, cam.axis)
    # The tongue's inner half is inside the plate's own box, so its slab only has to
    # start where the plate's footprint ends.
    tongue_lo = plate_half - 2.0
    for name, span, half_w, w0, w1 in (
            ("plate_tongue", (tongue_lo, hi), (P.STRUT_TIP_W + 8.0) / 2,
             rear, rear + P.PLATE_T),
            ("plate_spine", (lo, hi), P.SPINE_W / 2, rear - P.SPINE_H, rear)):
        out.append(_oriented_box(
            cam.pos + sum(span) / 2 * along + (w0 + w1) / 2 * cam.axis,
            [along, across, cam.axis],
            [(span[1] - span[0]) / 2 + 0.3, half_w + 0.3, (w1 - w0) / 2], name))
    out.append(_oriented_box(cam.pos + P.BODY_BACK / 2 * cam.axis,
                             [cam.right, cam.up, cam.axis],
                             [P.PCB_SIZE / 2, P.PCB_SIZE / 2, -P.BODY_BACK / 2],
                             "camera_body"))
    out.append(_oriented_box(cam.pos + P.BODY_FRONT / 2 * cam.axis,
                             [cam.right, cam.up, cam.axis],
                             [P.HOLDER_SIZE / 2, P.HOLDER_SIZE / 2,
                              P.BODY_FRONT / 2],
                             "camera_holder"))
    out.append(_cylinder(cam.pos + P.BODY_FRONT * cam.axis,
                         cam.pos + P.LENS_FRONT * cam.axis, P.LENS_DIA / 2,
                         "camera_lens"))
    w, t, cable = P.USB_KEEPOUT
    usb_centre = cam.point(P.USB_U_OFFSET, -(P.PCB_SIZE / 2 + cable / 2), -1.0 + t / 2)
    out.append(_oriented_box(usb_centre, [cam.right, cam.up, cam.axis],
                             [w / 2, cable / 2, t / 2], "usb_keepout"))
    return out


def write_collision(cam=None, mass=None):
    """collision/payload.json: primitives plus mass properties, in SI."""
    cam = cam or design.CameraFrame()
    COLLISION_DIR.mkdir(exist_ok=True)
    primitives = collision_primitives(cam)
    mass = mass or checks.mass_properties(cam)
    inertia = np.array(mass["inertia_about_origin_g_mm2"]) * 1e-9   # g mm^2 -> kg m^2
    payload = {
        "frame": "gripper_link",
        "units": {"pos": "m", "size": "m (box: full extents; cylinder: radius, length)",
                  "mass": "kg", "inertia": "kg m^2"},
        "rot": "row-major 3x3, columns are the primitive's local axes in gripper_link",
        "primitives": primitives,
        "mass_kg": round(mass["mass_g"] / 1000.0, 5),
        "com_m": [round(v / 1000.0, 5) for v in mass["com_mm"]],
        "inertia_about_gripper_origin_kg_m2": [[round(v, 9) for v in row]
                                               for row in inertia],
        "material": f"PETG {P.PETG_DENSITY * 1000:.2f} g/cc at an effective fill "
                    f"of {P.EFFECTIVE_FILL}, plus the camera module and hardware",
        "note": "Deliberately not a convex hull: the free corridor between the "
                "clamp band and the camera is where a held bottle goes.",
    }
    (COLLISION_DIR / "payload.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_camera_spec(cam=None, payload_mass_g=None, measured=None, meshes=None,
                      occlusion=None, forward=None):
    """camera_spec.json, complete, whichever entry point called.

    Both numbers that used to be "not measured" or null after a bare export are
    filled in here: ``payload_mass_g`` from ``checks.mass_properties`` and the six
    lens numbers from ``camera_measurements``.  verify.py hands in what it has
    already computed; nothing else has to.
    """
    cam = cam or design.CameraFrame()
    if measured is None:
        measured = camera_measurements(cam, meshes, occlusion, forward)
    if payload_mass_g is None:
        payload_mass_g = round(checks.mass_properties(cam)["mass_g"], 1)
    spec = camera_spec(cam, measured)
    spec["payload_mass_g"] = payload_mass_g
    (_OUT / "camera_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def export_glb(cam=None):
    """One GLB of the assembly on the real wrist, for a 3D viewer. Metres."""
    cam = cam or design.CameraFrame()
    scene = trimesh.Scene()
    colours = {"upper_bracket": [219, 148, 47, 255], "lower_bracket": [21, 141, 140, 255],
               "camera_body": [38, 118, 80, 255], "camera_holder": [47, 138, 94, 255], "camera_lens": [28, 55, 71, 255],
               "camera_usb": [150, 150, 150, 255]}
    for name, mesh in checks.payload_meshes(cam).items():
        copy = mesh.copy()
        copy.apply_scale(0.001)
        copy.visual.face_colors = colours.get(name, [170, 182, 192, 255])
        scene.add_geometry(copy, node_name=name)
    for name, mesh in checks.rb.gripper_meshes(20.0).items():
        copy = mesh.copy()
        copy.apply_scale(0.001)
        copy.visual.face_colors = [113, 137, 150, 255]
        scene.add_geometry(copy, node_name=name)
    path = _OUT / "mount_on_gripper.glb"
    scene.export(path)
    return path


def write_v4_reference(block=None):
    """Write down v4's measured radii so a copy without the history can read them.

    ``mount/v4ref.py`` measures them with ``git show``, which needs the repository.
    Outside a checkout there is none, and the block came back "unavailable": that
    made ``verify.py --check`` exit 1 in any copy of this directory, and made a
    plain ``verify.py`` run there delete the v4 comparison out of the README. The
    numbers are a property of a commit, not of a run, so they are tracked.
    """
    block = block or v4ref.reference()
    variants = block.get("variants") or {}
    path = _OUT / v4ref.CACHE_NAME
    if not variants:
        # Nothing measured and nothing cached: leave whatever is there alone rather
        # than replacing a good file with an empty one.
        return path
    path.write_text(v4ref.cache_text(variants))
    return path


def write_all(cam=None, mass=None, measured=None, meshes=None, occlusion=None,
              forward=None, v4_reference=None):
    """Every file this module owns, in one call.  verify.py and main() share it."""
    cam = cam or design.CameraFrame()
    parts = export_parts(cam)
    mass = mass or checks.mass_properties(cam)
    write_collision(cam, mass)
    spec = write_camera_spec(cam, round(mass["mass_g"], 1), measured, meshes,
                             occlusion, forward)
    export_glb(cam)
    write_v4_reference(v4_reference)
    return parts, spec


def check_against_tree(cam=None, **kw):
    """Regenerate into a temp directory and diff against the tracked files.

    Returns the list of paths that differ.  Nothing under the package is opened
    for writing, which is the point: `verify.py --check` and
    `python -m mount.export --check` are read-only.
    """
    scratch = Path(tempfile.mkdtemp(prefix="g1_export_check_"))
    previous = set_output_root(scratch)
    try:
        write_all(cam, **kw)
        drift = []
        for name in WRITTEN_FILES:
            here, there = ROOT / name, scratch / name
            if not here.exists():
                drift.append(f"{name}: missing from the tree")
            elif not there.exists():
                drift.append(f"{name}: the run did not produce it")
            elif not filecmp.cmp(here, there, shallow=False):
                drift.append(f"{name}: {here.stat().st_size} B tracked vs "
                             f"{there.stat().st_size} B regenerated")
        return drift
    finally:
        set_output_root(previous)
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="write nothing; regenerate into a temp directory "
                             "and report any tracked file that differs")
    args = parser.parse_args(argv)
    cam = design.CameraFrame()
    if args.check:
        drift = check_against_tree(cam)
        for line in drift:
            print(f"  DRIFT {line}")
        print(f"{len(WRITTEN_FILES) - len(drift)}/{len(WRITTEN_FILES)} "
              f"exported files match the tree")
        return 1 if drift else 0
    parts, _ = write_all(cam)
    print(json.dumps(parts, indent=2))
    print(json.dumps(checks.mesh_validity(STL_DIR), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
