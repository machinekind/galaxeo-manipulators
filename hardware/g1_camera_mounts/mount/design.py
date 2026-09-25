"""CadQuery model of the G1 wrist-camera mount.

Two printed PETG parts plus hardware:

    upper_bracket()   top clamp half + locating yoke + strut + camera plate
    lower_bracket()   bottom clamp half + locating yoke + captured M4 nuts

They close on the 60 mm gripper housing at X = -52..-24, butt the rail's back
face at X = -15.65 with four stop pads, and key across the rail's two end
faces.  The camera sits outboard on +Y, low and behind the rail, because a held
bottle owns the corridor above the housing forward of X = -24.

All coordinates are gripper_link millimetres (see params.py).
"""
import math

import cadquery as cq
import numpy as np

from . import params as P


# --------------------------------------------------------------------------
# small primitives
# --------------------------------------------------------------------------

def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from corner to corner; the pairs may be in either order."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    return (cq.Workplane("XY")
            .box(x1 - x0, y1 - y0, z1 - z0)
            .translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)))


def cyl_x(x0, x1, radius, y=0.0, z=0.0):
    """Cylinder with its axis along X."""
    return cq.Workplane("YZ", origin=(x0, y, z)).circle(radius).extrude(x1 - x0)


def cyl_z(x, y, z0, z1, radius):
    """Cylinder with its axis along Z."""
    return cq.Workplane("XY", origin=(x, y, z0)).circle(radius).extrude(z1 - z0)


def hex_prism(x, y, z0, z1, across_flats):
    """Hex pocket/nut solid with its axis along Z, flats normal to X.

    The 30 degree rotation is what makes the docstring true.  Without it cadquery
    puts vertices on X, so the prism is ``across_flats / cos(30) = 1.155 x`` wider in
    X than in Y -- and the nut channel in ``_nut_pocket``, which is a straight slot
    of the across-flats width, would be 0.8 mm too narrow for the nut it has to pass.
    With the flats normal to X the nut slides along Y between two flat walls, which
    is also what stops it turning.
    """
    return (cq.Workplane("XY", origin=(x, y, z0))
            .polygon(6, across_flats / math.cos(math.pi / 6))
            .extrude(z1 - z0)
            .rotate((x, y, 0), (x, y, 1), 30.0))


def oriented_box(centre, axes, half):
    """Box centred at `centre` with the three unit vectors `axes` as its axes."""
    e1, _, normal = [np.asarray(a, float) for a in axes]
    origin = np.asarray(centre, float) - half[2] * normal
    plane = cq.Plane(origin=cq.Vector(*origin), xDir=cq.Vector(*e1),
                     normal=cq.Vector(*normal))
    return cq.Workplane(plane).rect(2 * half[0], 2 * half[1]).extrude(2 * half[2])


# --------------------------------------------------------------------------
# camera frame
# --------------------------------------------------------------------------

def camera_rotation(yaw_deg=P.CAM_YAW_DEG, pitch_deg=P.CAM_PITCH_DEG,
                    roll_deg=P.CAM_ROLL_DEG, side=P.HANDEDNESS):
    """Columns are (optical axis, -image right, image up) in gripper_link.

    Yaw turns the axis inboard (toward -Y for a +Y mount), pitch turns it down,
    roll spins the image about the optical axis.
    """
    cy, sy = math.cos(math.radians(side * yaw_deg)), math.sin(math.radians(side * yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    rz = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


class CameraFrame:
    """Where the lens is and which way it looks."""

    def __init__(self, pos=P.CAM_POS, yaw=P.CAM_YAW_DEG, pitch=P.CAM_PITCH_DEG,
                 roll=P.CAM_ROLL_DEG, side=P.HANDEDNESS):
        self.side = side
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        # pos is given for the +Y hand; mirror it in Y for the other one.
        self.pos = np.array([pos[0], side * pos[1], pos[2]], dtype=float)
        self.R = camera_rotation(yaw, pitch, roll, side)

    @property
    def axis(self):
        """Unit optical axis."""
        return self.R[:, 0]

    @property
    def right(self):
        """Image +u direction (to the right in the picture)."""
        return -self.side * self.R[:, 1]

    @property
    def up(self):
        """Image +v direction (upward in the picture)."""
        return self.R[:, 2]

    @property
    def pupil(self):
        """Entrance pupil: what the sightline checks cast from."""
        return self.pos + P.PUPIL_AHEAD * self.axis

    def plane(self, offset=0.0):
        """A CadQuery plane parallel to the PCB, `offset` mm along the axis."""
        origin = self.pos + offset * self.axis
        return cq.Plane(origin=cq.Vector(*origin), xDir=cq.Vector(*self.right),
                        normal=cq.Vector(*self.axis))

    def point(self, u, v, w):
        """Gripper-frame point at image-right u, image-up v, w along the axis."""
        return self.pos + u * self.right + v * self.up + w * self.axis


OVERLAP = 1.0   # how far unioned features reach into each other, in mm


def plate_rear_offset():
    """Distance from the PCB front face back to the camera plate's rear face."""
    return -(P.PCB_T + P.STANDOFF_H + P.PLATE_T)


# --------------------------------------------------------------------------
# clamp band and ears
# --------------------------------------------------------------------------

def _half_space(upper):
    """Everything above (or below) the split plane, with the closing gap."""
    if upper:
        return box(-120, 120, -120, 120, P.SPLIT_GAP, 120)
    return box(-120, 120, -120, 120, -120, -P.SPLIT_GAP)


def _band(upper):
    ring = (cq.Workplane("YZ", origin=(P.CLAMP_X0, 0, 0))
            .circle(P.CLAMP_OD / 2).circle(P.BORE_DIA / 2)
            .extrude(P.CLAMP_X1 - P.CLAMP_X0))
    return ring.intersect(_half_space(upper))


def _ears(upper):
    z0, z1 = (P.SPLIT_GAP, P.EAR_HALF_Z) if upper else (-P.EAR_HALF_Z, -P.SPLIT_GAP)
    part = None
    for side in (-1, 1):
        ear = box(P.EAR_X0, P.EAR_X1, side * P.EAR_Y1, side * P.EAR_Y0, z0, z1)
        ear = ear.edges("|Z").fillet(3.0)
        part = ear if part is None else part.union(ear)
    return part


def _yoke(upper):
    """Axial stop pads + anti-rotation key ears, one Z-half of them.

    The horn is a tapered web from the band out to the rail back face, held
    inside |Z| <= YOKE_HALF_Z so it passes under/over the finger carriages with
    5 mm to spare instead of the 2 mm a full-height stop would get.  The key ears
    sit *beside* the rail rather than in front of it and may run to KEY_HALF_Z,
    1 mm higher, before the same 4 mm gate binds.
    """
    z0 = P.SPLIT_GAP if upper else -P.YOKE_HALF_Z
    z1 = P.YOKE_HALF_Z if upper else -P.SPLIT_GAP
    k0 = P.SPLIT_GAP if upper else -P.KEY_HALF_Z
    k1 = P.KEY_HALF_Z if upper else -P.SPLIT_GAP
    part = None
    for side in (-1, 1):
        outline = [(P.YOKE_X0, side * P.STOP_Y0),
                   (P.YOKE_X0, side * 44.0),
                   (P.RAIL_BACK_X, side * P.YOKE_OUTER_Y),
                   (P.RAIL_BACK_X, side * P.STOP_Y0)]
        horn = (cq.Workplane("XY").polyline(outline).close()
                .extrude(z1 - z0).translate((0, 0, z0)))
        ear = box(P.RAIL_BACK_X - OVERLAP, P.KEY_FRONT_X,
                  side * P.KEY_INNER_Y, side * P.YOKE_OUTER_Y, k0, k1)
        # No web is needed to carry the extra millimetre of ear height: the ear is
        # 4.2 x 11.65 mm in plan and sits on the horn over its whole length.  A web
        # forward of RAIL_BACK_X at |Y| < KEY_INNER_Y would come within 3.6 mm of a
        # carriage, which is what the 4 mm gate is there to stop.
        # 1.2 mm lead-in chamfer so the key finds the rail end face on assembly
        lead = (cq.Workplane("XY")
                .polyline([(P.KEY_FRONT_X, side * P.KEY_INNER_Y),
                           (P.KEY_FRONT_X, side * (P.KEY_INNER_Y + 1.2)),
                           (P.KEY_FRONT_X - 1.2, side * P.KEY_INNER_Y)])
                .close().extrude(k1 - k0).translate((0, 0, k0)))
        piece = horn.union(ear).cut(lead)
        part = piece if part is None else part.union(piece)
    return part


# --------------------------------------------------------------------------
# strut and camera plate
# --------------------------------------------------------------------------

def _band_root(cam, towards):
    """A point just inside the clamp band's OD, at the azimuth of `towards`."""
    azimuth = math.atan2(towards[2], abs(towards[1]))
    radius = P.CLAMP_OD / 2 - 2.0                    # start just inside the band OD
    return np.array([(P.CLAMP_X0 + P.CLAMP_X1) / 2 + 1.0,
                     cam.side * radius * math.cos(azimuth),
                     radius * math.sin(azimuth)]), azimuth


def plate_tip_uv():
    """(u, v) on the plate where the strut lands.  See params.STRUT_TIP_UV."""
    return np.array(P.STRUT_TIP_UV, float)


def _strut_geometry(cam):
    """(root, tip, axis, length, side direction) of the strut."""
    u, v = plate_tip_uv()
    tip = cam.point(u, v, plate_rear_offset())
    root, azimuth = _band_root(cam, tip)
    delta = tip - root
    length = float(np.linalg.norm(delta))
    axis = delta / length
    # Wide axis of the section lies in the plane containing the strut and X.
    wide = np.cross(axis, [0.0, -cam.side * math.sin(azimuth), math.cos(azimuth)])
    norm = np.linalg.norm(wide)
    wide = wide / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return root, tip, axis, length, wide


def ziptie_flanks(cam):
    """Every candidate flank for the tie lug, scored and with its verdict.

    The lug wants to be on the strut flank that faces the USB connector's exit, so
    the cable runs from the connector down the strut to the clamp band rather than
    across the bottle's corridor.  But "faces the connector" on its own has now put
    this lug in the wrong place twice, and each time the thing it broke was written
    down somewhere else as a rule:

      * on the +X flank near the tip, the first cut reached X = -19.97, forward of
        the KEEPOUT_X plane the README promises nothing crosses;
      * on the -Z flank near the root, this revision's cut hung to Z = -1.94, which
        is 111 mm3 of the *upper* bracket below the split plane, 12.9 mm3 of it
        inside the lower strap, and an exported STL standing on a nub.  It also
        stood squarely in the only straight line by which the camera-side M4's nut
        can reach its pocket.

    So the two rules are applied first and the preference second: a flank is
    admissible only if the whole lug stays above +SPLIT_GAP and behind KEEPOUT_X,
    and among the admissible ones the most connector-facing wins.  The verdicts are
    returned rather than swallowed, so ``validation.json`` records why the chosen
    flank was chosen and what the alternatives cost.
    """
    root, _, axis, length, wide = _strut_geometry(cam)
    tall = np.cross(axis, wide)
    # The strut tapers, so the flank the lug sits on is narrower here than STRUT_W /
    # STRUT_H.  Using the nominal section put the slot's inner face inside the strut
    # and left a 2.3 mm lamina over it.
    f = P.ZIPTIE_ALONG / length
    local_w = P.STRUT_W + (P.STRUT_TIP_W - P.STRUT_W) * f
    local_h = P.STRUT_H + (P.STRUT_TIP_H - P.STRUT_H) * f
    half = P.PCB_SIZE / 2
    exit_point = cam.point(P.USB_U_OFFSET, -(half + P.USB_KEEPOUT[2]), -1.0)
    at = root + P.ZIPTIE_ALONG * axis
    towards = exit_point - at
    towards -= axis * (towards @ axis)
    slot_t = P.ZIPTIE_SLOT[1] + 0.8
    thickness = 1.0 + 2 * P.ZIPTIE_WEB + slot_t
    lug_half = [thickness / 2, 7.0, P.ZIPTIE_SLOT[0] / 2 + P.ZIPTIE_WEB]
    out = []
    for label, normal, span in (("+X", wide, local_w), ("-X", -wide, local_w),
                                ("-Z", tall, local_h), ("+Z", -tall, local_h)):
        other = tall if abs(normal @ wide) > 0.5 else wide
        centre = at + (span / 2 - 1.0 + thickness / 2) * normal
        axes = [normal, other, axis]
        # The lug is a box, so its extent along any world axis is the sum of each
        # of its own half-extents projected onto it.
        reach = [sum(h * abs(a[k]) for h, a in zip(lug_half, axes)) for k in range(3)]
        z_min = centre[2] - reach[2]
        x_max = centre[0] + reach[0]
        out.append(dict(
            flank=label, normal=[round(v, 4) for v in normal],
            span_mm=round(span, 2),
            centre_mm=[round(v, 2) for v in centre],
            z_min_mm=round(float(z_min), 2), x_max_mm=round(float(x_max), 2),
            clears_split_plane=bool(z_min >= P.SPLIT_GAP),
            behind_keepout=bool(x_max <= P.KEEPOUT_X),
            faces_connector=round(float(normal @ towards), 2),
            slot_to_connector_mm=round(float(np.linalg.norm(
                at + (span / 2 + P.ZIPTIE_WEB + slot_t / 2) * normal - exit_point)), 1)))
    for row in out:
        row["admissible"] = bool(row["clears_split_plane"] and row["behind_keepout"])
    return out, at, axes, lug_half, dict(wide=wide, tall=tall, axis=axis,
                                         local_w=local_w, local_h=local_h,
                                         thickness=thickness, slot_t=slot_t)


def _ziptie_geometry(cam):
    """(lug centre, slot centre, lug axes, lug half-extents, flank normal)."""
    rows, at, _, lug_half, geo = ziptie_flanks(cam)
    admissible = [r for r in rows if r["admissible"]]
    assert admissible, ("no strut flank keeps the tie lug above the split plane "
                        "and behind KEEPOUT_X; see design.ziptie_flanks")
    best = max(admissible, key=lambda r: r["faces_connector"])
    normal = np.array(best["normal"], float)
    span = best["span_mm"]
    other = geo["tall"] if abs(normal @ geo["wide"]) > 0.5 else geo["wide"]
    # Thickness from the flank: 1 mm embedded, ZIPTIE_WEB of inner wall, the slot,
    # ZIPTIE_WEB of outer wall.  Both walls and the two end walls are >= MIN_WALL,
    # which the 2.0 mm outer web of the first cut was not.
    lug_centre = at + (span / 2 - 1.0 + geo["thickness"] / 2) * normal
    slot_centre = at + (span / 2 + P.ZIPTIE_WEB + geo["slot_t"] / 2) * normal
    return lug_centre, slot_centre, [normal, other, geo["axis"]], lug_half, normal


def module_keepout(cam=None, grow=0.4):
    """Everything the bought camera module owns: PCB, body, lens barrel, connector.

    Deliberately starts at BODY_BACK -- the plate's front face -- because that is
    what the README promises a module: rear components anywhere down to there.  The
    four standoff pads are the one exception and they are added to the plate after
    this is cut, not exempted here.
    """
    cam = cam or CameraFrame()
    half = P.PCB_SIZE / 2 + grow
    board = (cq.Workplane(cam.plane(P.BODY_BACK - grow)).rect(2 * half, 2 * half)
             .extrude(-P.BODY_BACK + 2 * grow))
    holder = (cq.Workplane(cam.plane(0.0))
              .rect(P.HOLDER_SIZE + 2 * grow, P.HOLDER_SIZE + 2 * grow)
              .extrude(P.BODY_FRONT + grow))
    lens = (cq.Workplane(cam.plane(P.BODY_FRONT)).circle(P.LENS_DIA / 2 + grow)
            .extrude(P.LENS_FRONT - P.BODY_FRONT + 4.0))
    return (board.union(holder).union(lens)
            .union(camera_envelope(cam)["camera_usb"]))


def _strut(cam):
    root, tip, axis, length, wide = _strut_geometry(cam)
    # Solid section, not a tube.  Hollowing 42 mm of strut would save 6 g and cost a
    # third of the bending stiffness, and a sealed internal void is not something a
    # slicer or an inspector can do anything with.  It tapers from STRUT_W x STRUT_H
    # at the band to STRUT_TIP_W x STRUT_TIP_H at the plate, which is what lets it
    # thread past the board: the offset alone would need 6 mm more.
    strut = (cq.Workplane(cq.Plane(origin=cq.Vector(*root), xDir=cq.Vector(*wide),
                                   normal=cq.Vector(*axis)))
             .rect(P.STRUT_W, P.STRUT_H)
             .workplane(offset=length + OVERLAP)
             .rect(P.STRUT_TIP_W, P.STRUT_TIP_H).loft())
    # Stop the prism inside the spine's back face.  It overshoots the plate by
    # OVERLAP along its own axis, and because the strut is oblique to the plate that
    # 1 mm spreads into a thin lip hanging off the back, which the wall-thickness gate
    # finds and which nothing needs.  The spine (see _camera_plate) carries the joint.
    # Cut 1 mm *inside* the spine's rear face, not flush with it: a cut plane
    # coincident with another face unions into coplanar faces, and that is what makes
    # a tessellation leak -- the upper bracket's STL came out non-watertight.
    outside = (cq.Workplane(cam.plane(plate_rear_offset() - P.SPINE_H + 1.0))
               .rect(300, 300).extrude(-200))
    strut = strut.cut(outside)
    # Belt and braces: the module's own envelope is cut out of the strut, so that if
    # the tip offset is ever reduced the part loses material rather than silently
    # growing through the camera.  ``checks.g7_module_fit`` asserts this cut removes
    # nothing, i.e. that the offset is doing the work.
    strut = strut.cut(module_keepout(cam))
    # And it is an *upper*-half part.  At the section and root radius this is built
    # with the prism clears the split plane on its own -- its lowest corner is at
    # Z = +3.45, 2.65 mm above +SPLIT_GAP -- so this intersect removes nothing today.
    # It stays because the margin is 2.65 mm on an 18 x 16 mm section whose root sits
    # 11.27 mm up: a wider strut or a lower band root would put the prism into the
    # lower strap, and a silent clip here is better than a part that collides.  What
    # must not be clipped is anything unioned *after* it -- see the zip-tie lug below.
    strut = strut.intersect(_half_space(True))
    # Zip-tie strain relief: a lug on the strut flank with a through slot, not a
    # cut into the section -- the closed section stays closed and there is no
    # sub-millimetre sliver of wall left behind.
    #
    # This union comes AFTER the half-space intersect above, and deliberately is not
    # clipped by it: the lug has to clear the split plane on its own, by being far
    # enough along the strut (see params.ZIPTIE_ALONG), because clipping it would
    # shave the outer 2.9 mm web the tie bears against down to nothing and leave the
    # part printing on a truncated stub.  Clipping here would also make
    # ``checks.split_plane_confinement`` vacuous instead of a gate.  At the 11.0 mm
    # this lug was first placed at, it hung 2.74 mm below the plane and into the
    # lower strap; that is the defect the gate exists for.
    lug_centre, slot_centre, axes, lug_half, normal = _ziptie_geometry(cam)
    lug = oriented_box(lug_centre, axes, lug_half)
    slot = oriented_box(slot_centre, axes,
                        [P.ZIPTIE_SLOT[1] / 2 + 0.4, 10.0, P.ZIPTIE_SLOT[0] / 2])
    return strut.union(lug).cut(slot)


def _m2_slot_centres():
    """The four screw slots run diagonally so one plate covers 26..30 mm pitch."""
    half_lo, half_hi = P.M2_PITCH_MIN / 2, P.M2_PITCH_MAX / 2
    for su in (-1, 1):
        for sv in (-1, 1):
            yield su, sv, half_lo, half_hi


def m2_slot_length():
    """Overall length of one diagonal screw slot, end cap to end cap.

    A hole at half-pitch h sits at (h, h), so going from M2_PITCH_MIN to
    M2_PITCH_MAX moves it sqrt(2) * (max - min) / 2 along the diagonal -- not
    (max - min), which is what the first cut of this used.  The 1.17 mm of extra
    slot that bought was exactly what pushed the slot out through the plate's
    chamfered corner.
    """
    return 2.6 + math.sqrt(2.0) * (P.M2_PITCH_MAX - P.M2_PITCH_MIN) / 2


def standoff_pads(cam=None, grow=0.0):
    """The four pads the PCB sits on: the one thing allowed inside BODY_BACK."""
    cam = cam or CameraFrame()
    rear = plate_rear_offset()
    face = rear + P.PLATE_T
    slot_length = m2_slot_length()
    out = None
    for su, sv, half_lo, half_hi in _m2_slot_centres():
        pad = (cq.Workplane(cam.plane(face - OVERLAP))
               .center(su * (half_lo + half_hi) / 2, sv * (half_lo + half_hi) / 2)
               .slot2D(slot_length + 2 * P.MIN_WALL + 2 * grow,
                       2 * P.M2_PAD_HALF_W + 2 * grow,
                       45.0 if su * sv > 0 else -45.0)
               .extrude(P.STANDOFF_H + OVERLAP + grow))
        out = pad if out is None else out.union(pad)
    return out


def plate_tongue_extent(spine=False):
    """(along_min, along_max) of the tongue or the spine, in the tip's direction.

    One definition for the solid and for the collision primitives, so the two cannot
    disagree -- the first cut built the solid from ``slot2D``'s overall length and the
    primitive from the reach, and the primitive came out 10 mm short at the far end,
    leaving 211 payload samples uncovered.

    The tongue stops just past the module's centre (it only has to get from the plate
    to the strut); the spine runs back further, because it is the flange that makes
    the whole span a T section.
    """
    reach = float(np.hypot(*plate_tip_uv())) + P.PLATE_TONGUE
    # The spine stops 1 mm short of the tongue's far end: concentric end caps at the
    # same station are one more tangency, and it was the last unclosed patch.
    return ((-P.SPINE_BACK, reach - 1.0) if spine else (-6.0, reach))


def _camera_plate(cam):
    rear = plate_rear_offset()
    plate = (cq.Workplane(cam.plane(rear))
             .rect(P.PLATE_SIZE, P.PLATE_SIZE).extrude(P.PLATE_T))
    # Clip the corners, if there is room to: on this plate there is not -- see
    # params.PLATE_SIZE -- and the loop is a no-op at PLATE_CHAMFER = 0.
    for su in (-1, 1):
        for sv in (-1, 1):
            if P.PLATE_CHAMFER <= 0.0:
                continue
            half = P.PLATE_SIZE / 2
            corner = [(su * half, sv * (half - P.PLATE_CHAMFER)),
                      (su * half, sv * half),
                      (su * (half - P.PLATE_CHAMFER), sv * half)]
            wedge = (cq.Workplane(cam.plane(rear - 1.0))
                     .polyline(corner).close().extrude(P.PLATE_T + 2.0))
            plate = plate.cut(wedge)
    # A rounded lug around each slot's outer end cap.  Without them the wall between
    # the cap and the plate's edge is 1.7 mm; the lug makes it MIN_WALL by putting
    # the material where it is needed instead of growing the whole plate, which G1's
    # arm-link bound cannot afford.  m2_slot_wall() is the number.
    for su, sv, half_lo, half_hi in _m2_slot_centres():
        # Round, not square: a square lug's outer corner is the payload's most
        # negative X and costs 0.6 mm of G1's arm-link bound, which has half a
        # millimetre to give.
        lug = (cq.Workplane(cam.plane(rear))
               .center(su * P.M2_PITCH_MAX / 2, -sv * P.M2_PITCH_MAX / 2)
               .circle(P.M2_LUG_R).extrude(P.PLATE_T))
        plate = plate.union(lug)
    # Tongue out to the strut: the strut lands 30.7 mm off the module's centre,
    # which is past the plate's edge, so the plate runs on to meet it.  It reaches
    # *toward* the roll axis, so it costs no swept radius.
    #
    # A cq.Plane built from (xDir, normal) has yDir = normal x xDir, which on the
    # +Y hand is -image up.  So the plane's local y is -v there, and anything
    # asymmetric in v has to be flipped going in -- the first cut of the tongue did
    # not, and it grew upward out of the top of the plate instead of down toward the
    # strut.
    #
    # On the -Y hand `right` flips sign, so axis x right is +image up and the sign
    # is the other way.  It used to be the literal -1 below, which made
    # `payload_parts(side=-1)` a part that is not the mirror of the right hand at
    # all -- its tongue, spine and lugs all went the wrong way, 4470 mm3 different
    # from mirror(right) and 106.8 mm of swept radius against a 100 mm gate.  So it
    # is read off the frame, and checks.mirror_consistency is the gate.
    tip_u, tip_v = plate_tip_uv()
    lo, hi = plate_tongue_extent()
    width = P.STRUT_TIP_W + 8.0
    v_sign = float(np.sign(np.dot(np.cross(cam.axis, cam.right), cam.up)))
    unit = np.array([tip_u, v_sign * tip_v]) / math.hypot(tip_u, tip_v)
    angle = math.degrees(math.atan2(unit[1], unit[0]))
    tongue = (cq.Workplane(cam.plane(rear))
              .center(*(unit * (lo + hi) / 2))
              .slot2D(hi - lo, width, angle)
              .extrude(P.PLATE_T))
    plate = plate.union(tongue)
    # Spine on the plate's REAR face, running the whole span from past the module's
    # centre out to the strut's tip.  Without it the load path is 30 mm of 4.5 mm
    # plate between the camera's screws and the strut, and the first mode comes out
    # at 50 Hz -- under the gate.  With it the section is a T and it is 240 Hz.
    #
    # The rear face is the only side with room: the board covers all but 2 mm of the
    # front, and a rib there would also have to dodge the connector.  The cost is
    # X -- the rear normal is mostly -X -- but it lands at X = -45, where the plate's
    # own (+u, -v) corner is already at -52.
    slo, shi = plate_tongue_extent(spine=True)
    # Start 0.5 mm forward of the plate's rear face and reach back past it, rather
    # than starting on it: the plate, its four lugs and the tongue all grow forward
    # out of that plane, and a solid growing backward out of the same plane splits it
    # into coplanar faces. That is the tangency that leaks -- it left four unclosed
    # patches, all of them on w = plate_rear_offset().
    spine = (cq.Workplane(cam.plane(rear + 0.5))
             .center(*(unit * (slo + shi) / 2))
             .slot2D(shi - slo, P.SPINE_W, angle)
             .extrude(-(P.SPINE_H + 0.5)))
    plate = plate.union(spine)
    # 3 mm standoff pads, then the screw slots through pad and plate.  The pad is
    # MIN_WALL wider than its slot all round: the first cut gave it 1.9 mm, which is
    # the wall the M2 screw bears against.  Its cap radius is deliberately 0.3 mm
    # *less* than the lug's below, not equal to it: two coincident cylindrical faces
    # in a union is the tangential contact that makes a tessellation leak, and it did
    # -- three unclosed patches, one per lug.
    face = rear + P.PLATE_T
    slot_length = m2_slot_length()
    plate = plate.union(standoff_pads(cam))
    relief = (cq.Workplane(cam.plane(face - P.PLATE_RELIEF_DEPTH))
              .rect(P.PLATE_RELIEF, P.PLATE_RELIEF).extrude(P.PLATE_RELIEF_DEPTH + 1.0))
    plate = plate.cut(relief)
    for su, sv, half_lo, half_hi in _m2_slot_centres():
        hole = (cq.Workplane(cam.plane(rear - 1.0))
                .center(su * (half_lo + half_hi) / 2, sv * (half_lo + half_hi) / 2)
                .slot2D(slot_length, 2.6, 45.0 if su * sv > 0 else -45.0)
                .extrude(P.PLATE_T + P.STANDOFF_H + 2.0))
        plate = plate.cut(hole)
    return plate


# --------------------------------------------------------------------------
# the two printed parts
# --------------------------------------------------------------------------

def head_from_above(side, cam_side=P.HANDEDNESS):
    """Does this bolt's head go in from +Z?

    On the far side, yes.  On the camera side, no: the strut's root stands over that
    bolt ear, and there is no direction at all with 70 mm of clear shaft above it --
    measured, not guessed.  So the camera-side bolt is turned over: its head sits in
    the lower strap's counterbore, driven from underneath where nothing is in the
    way, and its nut drops into the upper bracket's ear from the top.
    """
    return side != cam_side


def bolt_hole_span():
    """(z0, z1) of the M4 clearance hole: far enough for the bolt, and no further.

    It used to be a flat -20 .. +20, which is a magic number on both counts.  It
    is 8 mm longer than the called-out bolt needs at the top, and the camera-side
    bolt's hole therefore ended *inside the strut's root*, 2.2 mm under the strut's
    outer surface: a blind Ø4.4 hole with a 2.2 mm cap over it, under MIN_WALL,
    which the thickness gate found only once the zip-tie lug stopped covering it.

    So it is derived: the ear, plus the furthest the called-out M4's tip reaches,
    plus a clearance so the bolt cannot bottom out in its own hole.
    """
    reach = max(abs(z) for side in (-1, 1) for z in m4_span_z(side))
    limit = max(P.EAR_HALF_Z + 1.0, reach + 1.8)
    return -limit, limit


def _nut_pocket(x, y, z0, z1):
    """Hex pocket plus the channel that lets the nut in from the ear's end face.

    The channel is the lesson from the other ear.  A pocket that opens only through
    one *face* is one union away from being roofed, and that is exactly what
    happened to the camera-side nut: its pocket opened upward through the upper
    ear's top face and the strut's root stands over it -- 937 mm3 of bracket in the
    nut's way, so the nut could not be fitted at all, and no gate looked, because
    G7 measured that the bolt could be *driven* and never that its nut could be
    *placed*.  That joint is tapped now (``tapped_joint``); this one keeps its steel
    nut, and keeps it reachable by a channel rather than by the hope that nothing
    gets built over the face.

    The nut slides in along Y between the channel's two flat walls, which is also
    what stops it turning -- see ``hex_prism`` for why the flats are normal to X.
    ``checks.fastener_seating`` sweeps it out along that line and measures what is
    in the way.
    """
    pocket = hex_prism(x, y, z0, z1, P.NUT_ACROSS_FLATS)
    sign = 1.0 if y > 0 else -1.0
    half = P.NUT_ACROSS_FLATS / 2 + 0.2      # the nut slides between two flats
    channel = box(x - half, x + half, y, sign * (P.EAR_Y1 + 1.0), z0, z1)
    return pocket.union(channel)


def tapped_joint(side, cam_side=P.HANDEDNESS):
    """Is this bolt threaded into the upper bracket instead of into a nut?

    Yes on the camera side, and it is the same feature that turned that bolt over
    in the first place: the strut's root stands over that ear.  Reversing the bolt
    solved the driver, and left the nut -- which then had to reach a pocket under
    the strut's root.  Swept as a solid, every straight line into that pocket is
    blocked (see params.M4_TAP_DIA).  So the far-side joint has a steel nut and the
    camera-side one is an M4 threaded into the upper ear itself;
    ``m4_thread_engagement`` measures how much thread that is.
    """
    return not head_from_above(side, cam_side)


def _fasteners_cut(part, upper, cam_side=P.HANDEDNESS):
    x = (P.EAR_X0 + P.EAR_X1) / 2
    hole_z0, hole_z1 = bolt_hole_span()
    for side in (-1, 1):
        above = head_from_above(side, cam_side)
        tapped = tapped_joint(side, cam_side)
        # Clearance everywhere the bolt passes through; tapping-drill diameter in
        # the half it threads into, which is the upper one on the tapped joint.
        if tapped and upper:
            part = part.cut(cyl_z(x, side * P.BOLT_Y, P.SPLIT_GAP - 0.5, hole_z1,
                                  P.M4_TAP_DIA / 2))
        else:
            part = part.cut(cyl_z(x, side * P.BOLT_Y, hole_z0, hole_z1,
                                  P.BOLT_DIA / 2))
        if upper == above:
            # the head's end: a counterbore opening through this half's outer face
            z0 = P.EAR_HALF_Z - P.COUNTERBORE_DEPTH if upper else -P.EAR_HALF_Z
            z1 = P.EAR_HALF_Z + 1.0 if upper else -P.EAR_HALF_Z + P.COUNTERBORE_DEPTH
            part = part.cut(cyl_z(x, side * P.BOLT_Y, z0, z1, P.COUNTERBORE_DIA / 2))
        elif not tapped:
            # The nut's end.  The pocket stops exactly at the ear's outer face
            # rather than 1 mm past it: 1 mm past used to cut a 0.21 mm knife edge
            # out of the strut-to-ear blend block that stood over it.  The channel
            # out through the ear's end face is what lets the nut in: a pocket that
            # opens only through a *face* is one union away from being roofed.
            z0 = P.EAR_HALF_Z - P.NUT_DEPTH if upper else -P.EAR_HALF_Z
            z1 = P.EAR_HALF_Z if upper else -P.EAR_HALF_Z + P.NUT_DEPTH
            part = part.cut(_nut_pocket(x, side * P.BOLT_Y, z0, z1))
    # Two alignment dowels carry the clocking from one keyed half into the other
    # without relying on the bolts being a tight fit.
    for side in (-1, 1):
        pin = cyl_z(P.DOWEL_X, side * P.DOWEL_Y,
                    -P.DOWEL_DEPTH, P.DOWEL_DEPTH, P.DOWEL_DIA / 2 + 0.1)
        part = part.cut(pin)
    return part


def _bore_cut(upper):
    """The bore, plus a radial relief over the crown of the arch.

    The crown prints as a horizontal bridge over roughly a 30 mm chord and droops
    a few tenths.  That crown is also the radial datum that sets the camera's Z,
    so the droop is relieved away rather than hoped about: the bore is opened by
    BORE_RELIEF over the top +-BORE_RELIEF_DEG and the clamp bears on the flanks,
    which print as walls.
    """
    cut = cyl_x(P.CLAMP_X0 - 2, P.RAIL_BACK_X, P.BORE_DIA / 2)
    reach = (P.BORE_DIA / 2 + P.BORE_RELIEF) * math.sin(math.radians(P.BORE_RELIEF_DEG))
    wedge = box(P.CLAMP_X0 - 3, P.RAIL_BACK_X, -reach, reach,
                0.0 if upper else -P.CLAMP_OD, P.CLAMP_OD if upper else 0.0)
    relief = (cyl_x(P.CLAMP_X0 - 2, P.RAIL_BACK_X, P.BORE_DIA / 2 + P.BORE_RELIEF)
              .intersect(wedge))
    return cut.union(relief)


def upper_bracket(cam=None):
    """Top clamp half: carries the strut, the camera plate and half the key."""
    cam = cam or CameraFrame()
    part = _band(True).union(_ears(True)).union(_yoke(True))
    part = part.union(_strut(cam)).union(_camera_plate(cam))
    part = part.union(box(P.BLEND_X[0], P.BLEND_X[1],
                         cam.side * P.BLEND_Y[0], cam.side * P.BLEND_Y[1],
                         P.SPLIT_GAP, P.BLEND_Z))
    part = part.cut(_bore_cut(True))
    part = _fasteners_cut(part, True, cam.side)
    part = part.cut(_grub_cut(cam.side))
    return part.clean()


def _grub_cut(side):
    """M3 grub screw, tapped inward through the camera-side key ear.

    The hole runs from the ear's outer face at |Y| = YOKE_OUTER_Y *inward*.  The
    first cut of this part extruded it the other way, so the swept cylinder lay
    entirely outside the solid and the hole the README tells the fitter to use did
    not exist -- probed on the built part and on the STL, 0 voids in 1242 samples.
    """
    return (cq.Workplane("XY", origin=(P.GRUB_X, side * P.YOKE_OUTER_Y,
                                       (P.SPLIT_GAP + P.KEY_HALF_Z) / 2))
            .transformed(rotate=(90, 0, 0))
            .circle(P.GRUB_DIA / 2).extrude(side * P.GRUB_DEPTH))


def lower_bracket(cam=None):
    """Bottom clamp half: plain strap, captured nuts, the other half of the key."""
    cam = cam or CameraFrame()
    part = _band(False).union(_ears(False)).union(_yoke(False))
    part = part.cut(_bore_cut(False))
    part = _fasteners_cut(part, False, cam.side)
    return part.clean()


def bore_gauge():
    """Half-ring that checks the printed bore diameter and the housing's clearance.

    3 mm thick, printed flat (see ``export.print_pose``).  It measures the printed
    bore diameter and finds obstructions on the real housing; it does not prove
    the bracket's own arch, which prints on edge -- the arch is relieved over the
    crown instead (``_bore_cut``).
    """
    ring = cyl_x(0, 3, P.BORE_DIA / 2 + 4).cut(cyl_x(-1, 4, P.BORE_DIA / 2))
    tab = box(0, 3, P.BORE_DIA / 2, P.BORE_DIA / 2 + 4, -6, 6)
    return ring.intersect(box(-1, 4, -60, 60, 0, 60)).union(tab)


def key_engagement():
    """How far the key ears run along the rail's end faces."""
    return P.KEY_FRONT_X - P.RAIL_BACK_X


def rail_key_gauge():
    """Reproduces the key pocket AND the axial stop plane, as one part.

    Print it, push it on the real gripper, and you know whether shrink has eaten
    the 0.15 mm fit and whether the stop pads reach the rail's back face, before
    you print 70 g of bracket.

    ``depth`` was written the other way round in the first cut of this part.
    ``box()`` sorts its corners, so nothing complained: the gauge came out 6.65 mm
    long with a 5.65 mm pocket and a 1.0 mm web, 47 % short of the engagement it
    exists to check.
    """
    depth = key_engagement()
    assert depth > 0, f"key engagement must be positive, got {depth}"
    block = box(0, depth + P.GAUGE_WEB, -P.YOKE_OUTER_Y, P.YOKE_OUTER_Y,
                -P.KEY_HALF_Z, P.KEY_HALF_Z)
    pocket = box(-1.0, depth, -P.KEY_INNER_Y, P.KEY_INNER_Y,
                 -P.KEY_HALF_Z - 1.0, P.KEY_HALF_Z + 1.0)
    # The pocket's back wall is the axial stop plane: slid on from the front, the
    # gauge bottoms on the rail's back face exactly where the brackets' stop pads
    # do, so one part checks both the clocking fit and the datum.
    return block.cut(pocket)


def m2_standoff():
    """Spare insulating standoff, in case the printed pads are dressed off.

    3.6 mm of wall around the M2 clearance hole, not the 1.3 mm the first cut had:
    a printed tube whose wall is under MIN_WALL fails the thickness gate, and a
    standoff's outside diameter is free -- it sits under the PCB at a screw hole.
    """
    return (cyl_z(0, 0, 0, P.STANDOFF_H, 3.8)
            .cut(cyl_z(0, 0, -1, P.STANDOFF_H + 1, 1.2)))


# --------------------------------------------------------------------------
# non-printed envelopes, used by the gate checks and the collision export
# --------------------------------------------------------------------------

def camera_envelope(cam=None):
    """The bought parts: module body, lens barrel, USB connector keep-out."""
    cam = cam or CameraFrame()
    half = P.PCB_SIZE / 2
    # Two blocks, not one: the board is 32 x 32 but the M12 holder in front of it is
    # a boss.  Calling the whole thing 32 x 32 x 15 cost the strut 6 mm of the
    # offset it needs and inflated the swept radius and the collision cover.
    body = (cq.Workplane(cam.plane(P.BODY_BACK)).rect(P.PCB_SIZE, P.PCB_SIZE)
            .extrude(-P.BODY_BACK))
    holder = (cq.Workplane(cam.plane(0.0)).rect(P.HOLDER_SIZE, P.HOLDER_SIZE)
              .extrude(P.BODY_FRONT))
    lens = (cq.Workplane(cam.plane(P.BODY_FRONT)).circle(P.LENS_DIA / 2)
            .extrude(P.LENS_FRONT - P.BODY_FRONT))
    # The connector leaves the board's lower edge, offset toward image-left, and the
    # cable runs down the strut to the zip-tie slot and on to the clamp band: away
    # from the bottle slab (|Y| <= 42) and away from the finger sweep.  The plane's
    # own second axis is -image up, so the centre is placed with cam.point() rather
    # than with .center().
    w, t, length = P.USB_KEEPOUT
    centre = cam.point(P.USB_U_OFFSET, -(half + length / 2), -1.0)
    plane = cq.Plane(origin=cq.Vector(*centre), xDir=cq.Vector(*cam.right),
                     normal=cq.Vector(*cam.axis))
    usb = cq.Workplane(plane).rect(w, length).extrude(t)
    return {"camera_body": body, "camera_holder": holder, "camera_lens": lens,
            "camera_usb": usb}


def m4_span_z(side, cam_side=P.HANDEDNESS):
    """(head end, tip end) Z of one bolt, in the direction it is driven."""
    if head_from_above(side, cam_side):
        head = P.EAR_HALF_Z - P.COUNTERBORE_DEPTH
        return head, head - P.M4_LENGTH
    head = -P.EAR_HALF_Z + P.COUNTERBORE_DEPTH
    return head, head + P.M4_LENGTH


def m4_protrusion(side=None, cam_side=P.HANDEDNESS):
    """How far the called-out screw sticks out past the far face of the joint.

    The two joints are not the same any more: the far-side bolt runs through both
    ears into a steel nut and pokes out past it, and the camera-side one threads
    into the upper ear and stops inside the bracket, under the strut's root.  Called
    with no side it gives the through-bolt's number, which is the one the BOM's
    screw length is chosen from.
    """
    through = round(P.M4_LENGTH - (2 * P.EAR_HALF_Z - P.COUNTERBORE_DEPTH), 2)
    if side is None or not tapped_joint(side, cam_side):
        return through
    # Tapped: the tip ends inside the tapped hole, which is bored deeper than the
    # bolt reaches (see bolt_hole_span), so nothing protrudes into free space.
    return 0.0


def m4_tip_clearance(side, cam_side=P.HANDEDNESS):
    """Hole left past a tapped bolt's tip: how far it is from bottoming out."""
    if not tapped_joint(side, cam_side):
        return 0.0
    _, tip = m4_span_z(side, cam_side)
    return round(bolt_hole_span()[1] - abs(tip), 2)


def m4_thread_engagement(side, cam_side=P.HANDEDNESS):
    """Length of PETG thread a tapped M4 engages, in millimetres."""
    if not tapped_joint(side, cam_side):
        return 0.0
    _, tip = m4_span_z(side, cam_side)
    return round(min(abs(tip), bolt_hole_span()[1]) - P.SPLIT_GAP, 2)


def fastener_envelope(cam_side=P.HANDEDNESS):
    """M4 heads, shanks and nuts, plus the dowels: they stick out of the part.

    The shank runs to the real tip of the called-out screw, not to a round number
    just past the part, so G1 and G5 see the thread that protrudes below the ear.
    """
    out = {}
    x = (P.EAR_X0 + P.EAR_X1) / 2
    for side in (-1, 1):
        # There are two dowels and there always were, but this line used to sit
        # *after* the tapped joint's `continue`, so only one of them was ever built.
        # Nothing said so: the pair gate reported 36 pairs where there are 45, the
        # +Y pin's 1.1 g of mass sat entirely on -Y and pulled the exported CoM
        # 0.39 mm across, and the total came out right only because DOWEL_MASS_G was
        # a 4 mm pin's mass where the BOM calls out a 3 mm one.
        out[f"dowel_{side}"] = cyl_z(P.DOWEL_X, side * P.DOWEL_Y,
                                     -P.DOWEL_LENGTH / 2, P.DOWEL_LENGTH / 2,
                                     P.DOWEL_DIA / 2)
        head_z, tip_z = m4_span_z(side, cam_side)
        sign = 1.0 if head_from_above(side, cam_side) else -1.0
        head = cyl_z(x, side * P.BOLT_Y, head_z,
                     head_z + sign * (P.COUNTERBORE_DEPTH + 0.5),
                     P.COUNTERBORE_DIA / 2 - 0.2)
        if tapped_joint(side, cam_side):
            # Clearance shank through the half it passes, then the thread, which is
            # modelled at the tapping-drill diameter: in a tapped hole the material
            # is displaced into the thread, so a solid at the bolt's major diameter
            # would read as interference in every boolean and mean nothing.
            clear = cyl_z(x, side * P.BOLT_Y, head_z + sign * P.COUNTERBORE_DEPTH,
                          P.SPLIT_GAP, 2.2)
            thread = cyl_z(x, side * P.BOLT_Y, P.SPLIT_GAP, tip_z,
                           P.M4_TAP_DIA / 2 - 0.025)
            out[f"m4_{side}"] = clear.union(thread).union(head)
            continue
        shank = cyl_z(x, side * P.BOLT_Y, tip_z, head_z + sign * P.COUNTERBORE_DEPTH, 2.2)
        nut_face = -sign * P.EAR_HALF_Z
        nut = hex_prism(x, side * P.BOLT_Y, nut_face,
                        nut_face + sign * P.NUT_DEPTH, P.NUT_ACROSS_FLATS - 0.3)
        out[f"m4_{side}"] = shank.union(head).union(nut)
    return out


def payload_parts(cam=None, side=None):
    """Every solid that moves with the wrist, keyed by name.

    `locating` names the features that must run close to the gripper on
    purpose (bore, stop pads, key ears); the gates treat them separately.
    """
    cam = cam or CameraFrame(side=side if side is not None else P.HANDEDNESS)
    parts = {"upper_bracket": upper_bracket(cam), "lower_bracket": lower_bracket(cam)}
    parts.update(camera_envelope(cam))
    parts.update(fastener_envelope(cam.side))
    return parts


# Every solid ``payload_parts`` is supposed to return.  Named, not counted in
# passing: one of the two dowels was silently missing for a whole revision and the
# only symptom was a pair gate that tested 36 pairs where there are 45, which reads
# like a smaller design rather than like a lost part.
PAYLOAD_PART_NAMES = ("camera_body", "camera_holder", "camera_lens", "camera_usb",
                      "dowel_-1", "dowel_1", "lower_bracket", "m4_-1", "m4_1",
                      "upper_bracket")


# The two printed parts.  Used as the occluder set for the sightline checks and
# as the finger-clearance set -- NOT as a "locating features" exemption; see
# `locating_region` for that.
PRINTED_PARTS = ("upper_bracket", "lower_bracket")


# How far each half travels when the bolts are torqued: the halves have to close
# BORE_FIT_CLEARANCE / 2 each, radially, to reach the liner that actually grips.
# That is 0.3 mm a side, so the ear gap goes from the modelled 2 x SPLIT_GAP =
# 1.6 mm to CLAMPED_EAR_GAP = 1.0 mm.  Nothing rigid may span the ears at more
# than that, and something did: see params.DOWEL_DEPTH.
CLAMP_TRAVEL_MM = P.BORE_FIT_CLEARANCE / 2


def clamped_pair(cam=None):
    """The two halves as they sit with the joint closed onto the liner.

    The bolts are deliberately not in here.  A screw's axial position relative to
    each half is set by its thread or its nut and follows the joint as it closes,
    so translating it rigidly would invent an interference that no assembly has.
    What is rigid across the gap is the two halves and the two dowels, and those
    are what ``checks.clamped_assembly`` measures.
    """
    cam = cam or CameraFrame()
    return (upper_bracket(cam).translate((0, 0, -CLAMP_TRAVEL_MM)),
            lower_bracket(cam).translate((0, 0, +CLAMP_TRAVEL_MM)))


# --------------------------------------------------------------------------
# split plane, part-pair and fastener invariants  (gates G9a-d)
# --------------------------------------------------------------------------

# Which side of Z = 0 each printed half is confined to, and by how much.  Every
# solid of the upper bracket must be at Z >= +SPLIT_GAP and every solid of the
# lower one at Z <= -SPLIT_GAP: the halves close onto a rubber liner, not onto each
# other, and both print on that face.
SPLIT_HALVES = {"upper_bracket": +1, "lower_bracket": -1}

# Features that deliberately cross the split plane, each with the pocket in the
# other half that receives it and the print-pose consequence of the protrusion.
# There are none: both halves are keyed to the rail itself and located to each other
# by two dowels that sit in holes in *both* halves, so nothing has to interlock.  An
# entry here is a licence to leave material on the wrong side of the plane, and it
# has to say what receives it and what it does to the print pose.
SPLIT_INTERLOCKS = ()


def split_plane_leak(solid, upper):
    """The part of `solid` on the wrong side of the split plane, as a solid.

    ``upper`` picks which side is wrong.  Returns None when there is nothing there,
    which is what both brackets are supposed to give.
    """
    wrong = (box(-200, 200, -200, 200, -200, P.SPLIT_GAP) if upper
             else box(-200, 200, -200, 200, -P.SPLIT_GAP, 200))
    leak = solid.intersect(wrong)
    if not leak.solids().vals():
        return None
    return leak


# Payload part pairs allowed to share volume, with the reason and a bound.  The
# gate is "no pair of payload solids intersects", and every exception is listed
# here with a number, so a new overlap is a gate failure rather than a silent one.
DECLARED_OVERLAPS = (
    dict(pair=("upper_bracket", "camera_body"), limit_mm3=800.0,
         reason="the four printed standoff pads stand inside the module's rear "
                "envelope on purpose -- that envelope starts at the plate's own "
                "front face (BODY_BACK) because that is the clearance the README "
                "promises a module, and the PCB sits ON the pads.  They are the "
                "one thing allowed inside it; module_keepout() adds them back "
                "after the cut rather than exempting them, and G7_module_fit "
                "measures the rest at 0 mm3."),
)


FASTENER_SLEEVE = 0.6         # how far outside a fastener the gate looks for a wall
INSERTION_REACH = 60.0        # how far out a head or nut is swept to prove it fits


def _sweep(solid, direction, reach=INSERTION_REACH, steps=40):
    """The volume a solid passes through on its way out along `direction`.

    A stepped union rather than a true sweep: for the prisms and cylinders here the
    two agree, and a union of translated copies cannot fail the way an OCC sweep of
    a compound can.
    """
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    out = cq.Workplane("XY").add(solid.val().copy())
    for k in range(1, steps + 1):
        offset = direction * (reach * k / steps)
        out = out.union(cq.Workplane("XY").add(solid.val().copy())
                        .translate(tuple(offset)))
    return out


def _m4_probe(side, cam_side, sleeve):
    """One M4's solid, its sleeve, and the paths its head and nut have to come in on.

    The nut goes in sideways through the ear's outboard end face (see
    ``_nut_pocket``) and the head goes in along the bolt's own axis from the face it
    is driven at.  Both are swept and measured, because the camera-side nut used to
    be specified as dropping in from the top of the upper ear -- where the strut's
    root stands, 937 mm3 of it.
    """
    x = (P.EAR_X0 + P.EAR_X1) / 2
    head_z, tip_z = m4_span_z(side, cam_side)
    sign = 1.0 if head_from_above(side, cam_side) else -1.0
    tapped = tapped_joint(side, cam_side)
    # Seated flush in its pocket.  It used to be modelled 0.2 mm proud of the ear's
    # outer face, which only went unnoticed because the pocket was cut 1 mm past
    # that face -- into whatever stood above it.
    nut_face = -sign * P.EAR_HALF_Z

    def head_piece(grow):
        return cyl_z(x, side * P.BOLT_Y, head_z,
                     head_z + sign * (P.COUNTERBORE_DEPTH + 0.5),
                     P.COUNTERBORE_DIA / 2 - 0.2 + grow)

    def build(grow):
        if tapped:
            clear = cyl_z(x, side * P.BOLT_Y, head_z + sign * P.COUNTERBORE_DEPTH,
                          P.SPLIT_GAP, 2.2 + grow)
            thread = cyl_z(x, side * P.BOLT_Y, P.SPLIT_GAP, tip_z,
                           P.M4_TAP_DIA / 2 - 0.025 + grow)
            return clear.union(thread).union(head_piece(grow))
        shank = cyl_z(x, side * P.BOLT_Y, tip_z,
                      head_z + sign * P.COUNTERBORE_DEPTH, 2.2 + grow)
        nut = hex_prism(x, side * P.BOLT_Y, nut_face, nut_face + sign * P.NUT_DEPTH,
                        P.NUT_ACROSS_FLATS - 0.3 + 2 * grow)
        return shank.union(head_piece(grow)).union(nut)

    paths = [("head", head_piece(0.0), (0.0, 0.0, sign))]
    if not tapped:
        nut_only = hex_prism(x, side * P.BOLT_Y, nut_face,
                             nut_face + sign * P.NUT_DEPTH, P.NUT_ACROSS_FLATS - 0.3)
        paths.append(("nut", nut_only, (0.0, 1.0 if side > 0 else -1.0, 0.0)))
    return build(0.0), build(sleeve), paths


def _dowel_probe(side, sleeve):
    def build(grow):
        # The bought pin, at its called-out length -- not the hole it sits in.  It
        # used to be the hole (+-DOWEL_DEPTH), which made the probe exactly as long
        # as the space it had and so could never read as too long for it.
        return cyl_z(P.DOWEL_X, side * P.DOWEL_Y,
                     -P.DOWEL_LENGTH / 2, P.DOWEL_LENGTH / 2,
                     P.DOWEL_DIA / 2 + grow)
    # A dowel has no insertion path and is not supposed to: it is dropped into one
    # half's blind hole before the halves are closed, and the other half's blind
    # hole receives it.  What the gate checks for a dowel is that both holes are
    # real and deep enough -- the engagement test -- and, because the two halves
    # then travel 0.3 mm each onto the liner, that the pin still fits once they
    # have: see ``checks.clamped_assembly``.
    return build(0.0), build(sleeve), []


def _m2_probe(cam, su, sv, half, sleeve):
    """One camera screw: head on the plate's rear face, shank through the stack.

    The M2 x 12 the BOM calls out, as a solid, so the gate can ask whether it fits
    rather than trusting the slot it was drawn for.  Head Ø3.9 x 2.0 (M2 SHCS 3.8 +
    fit), shank Ø2.2 through plate + standoff + PCB, nut Ø3.9 x 1.6 on the board.
    """
    rear = plate_rear_offset()
    stack = P.PLATE_T + P.STANDOFF_H + P.PCB_T
    centre = (su * half, sv * half)

    def build(grow):
        shank = (cq.Workplane(cam.plane(rear)).center(*centre)
                 .circle(1.1 + grow).extrude(stack))
        head = (cq.Workplane(cam.plane(rear)).center(*centre)
                .circle(1.95 + grow).extrude(-2.0))
        nut = (cq.Workplane(cam.plane(rear + stack)).center(*centre)
               .circle(1.95 + grow).extrude(1.6))
        return shank.union(head).union(nut)

    head_only = (cq.Workplane(cam.plane(rear)).center(*centre)
                 .circle(1.95).extrude(-2.0))
    nut_only = (cq.Workplane(cam.plane(rear + stack)).center(*centre)
                .circle(1.95).extrude(1.6))
    # The screw goes in from the plate's rear face, the nut on from the board side.
    paths = [("head", head_only, -cam.axis), ("nut", nut_only, cam.axis)]
    return build(0.0), build(sleeve), paths


def fastener_probes(cam=None, sleeve=FASTENER_SLEEVE):
    """Every bought fastener: name -> dict(solid, sleeve, holes, axis, paths).

    ``holes`` names the printed parts the fastener is supposed to pass through, so
    ``checks.fastener_seating`` can insist on three separate things that any one of
    them alone does not give:

      * the solid does not interfere with the part;
      * the sleeve just outside it *does*, over a real length -- that is the hole.
        A bolt drawn in free air passes the first perfectly and fails this;
      * each head, nut and pin can reach its seat from outside along ``paths``.
        The camera-side M4's nut passed the first two and could not be fitted: the
        strut's root stood in 937 mm3 of its way.

    The four M2 camera screws are probed here but are deliberately not payload
    parts: they are 4 x 0.6 g already carried as point masses in
    ``checks.mass_properties`` and inside the standoff-pad primitives in the
    collision cover, so adding them as parts would double-count them.
    """
    cam = cam or CameraFrame()
    z = np.array([0.0, 0.0, 1.0])
    out = {}
    for side in (-1, 1):
        tag = "p" if side > 0 else "n"
        solid, sleeved, paths = _m4_probe(side, cam.side, sleeve)
        out[f"m4_{tag}"] = dict(solid=solid, sleeve=sleeved, axis=z, paths=paths,
                                holes=("upper_bracket", "lower_bracket"))
        solid, sleeved, paths = _dowel_probe(side, sleeve)
        out[f"dowel_{tag}"] = dict(solid=solid, sleeve=sleeved, axis=z, paths=paths,
                                   holes=("upper_bracket", "lower_bracket"))
    for su, sv, half_lo, half_hi in _m2_slot_centres():
        solid, sleeved, paths = _m2_probe(cam, su, sv, (half_lo + half_hi) / 2.0,
                                          sleeve)
        out[f"m2_{'p' if su > 0 else 'n'}{'p' if sv > 0 else 'n'}"] = dict(
            solid=solid, sleeve=sleeved, axis=cam.axis, paths=paths,
            holes=("upper_bracket",))
    return out


def insertion_sweep(path_solid, direction, reach=INSERTION_REACH):
    """Where a head, nut or pin travels on its way in from outside."""
    return _sweep(path_solid, direction, reach)


def locating_region(point):
    """Is this bracket point one of the three surfaces that *should* touch the G1?

    A region test, not a part name.  The exemption G1 grants is "clamp bore
    contact excepted", which is the clamp band's own annulus, the four stop pads
    on the rail's back face and the key ears' inner faces -- not the strut, the
    camera plate, the tie lug or the bolt ears, which the first cut of
    ``g1_gripper_meshes`` exempted along with them because it keyed off the part
    name.  ``point`` is (N, 3) in gripper_link mm.
    """
    x, y, z = point[:, 0], point[:, 1], point[:, 2]
    radius = np.hypot(y, z)
    # The clamp band's annulus: its bore is 0.8 mm off the housing by design and its
    # bolt ears reach inside the OD, so the whole ring counts.
    band = ((x >= P.CLAMP_X0 - 2.0) & (x <= P.RAIL_BACK_X)
            & (radius <= P.CLAMP_OD / 2 + 1e-6))
    # The locating yoke: the stop horns and the key ears, which is everything the
    # bracket has inside |Z| <= KEY_HALF_Z forward of the band.  They run up to the
    # rail's back face and its end faces on purpose.
    yoke = ((x >= P.YOKE_X0 - 1.0) & (x <= P.KEY_FRONT_X + 1.0)
            & (np.abs(z) <= P.KEY_HALF_Z + 0.5))
    return band | yoke


def grub_surround(point):
    """Points within the grub screw's own neighbourhood in the key ear.

    The one place on either bracket that is thinner than MIN_WALL, and the only
    declared exception to the thickness gate.  The key ear is 6.2 mm tall because
    the finger carriages cap it at |Z| = 7 mm, so an M3 tapped hole through it
    cannot have 2.4 mm above and below: it has 1.85.  That is enough PETG around
    a thread that has to push a bracket 0.15 mm sideways, and the alternative is
    losing the feature that makes G4's clocking deterministic.
    """
    x, y, z = point[:, 0], point[:, 1], point[:, 2]
    return ((np.abs(x - P.GRUB_X) <= P.GRUB_DIA / 2 + 4.0)
            & (np.abs(y) >= P.KEY_INNER_Y - 0.5)
            & (np.abs(z) <= P.KEY_HALF_Z + 0.5))


THIN_EXCEPTIONS = (
    dict(name="grub screw surround in the key ear", region=grub_surround,
         floor_mm=1.8,
         reason="the ear is 6.2 mm tall because the carriages cap it at |Z| = 7 mm, "
                "so a tapped M3 cannot have MIN_WALL above and below it"),
)


def printed_parts(cam=None):
    """name -> (solid, print orientation) for export."""
    cam = cam or CameraFrame()
    return {
        "upper_bracket_right": (upper_bracket(cam), "as_modelled"),
        "lower_bracket": (lower_bracket(cam), "flip_x"),
        "bore_gauge": (bore_gauge(), "lay_flat"),
        "rail_key_gauge": (rail_key_gauge(), "flat"),
        "m2_standoff_3mm": (m2_standoff(), "flat"),
    }


def mirrored(solid):
    """Left-hand variant: the camera moves to -Y."""
    return cq.Workplane("XY").add(solid.val().mirror("XZ"))
