"""Every dimension of the G1 wrist-camera mount, in millimetres.

Frame: ``gripper_link`` of ``galaxea_a1xy_description``.

    X  tool axis, forward, and the wrist-roll (joint 6) axis
    Y  jaw closing axis
    Z  "up" with the wrist at its home roll

Everything below is either measured off the vendor STLs (the ``G1 ...``
block -- see ``mount.robot.measure_gripper``, which re-derives the numbers and
``verify.py``, which asserts them) or a design choice with the reason next to
it.  Nothing here is a nominal from a drawing we do not have.
"""

# --------------------------------------------------------------------------
# G1 gripper, measured on meshes/gripper_link.STL and the two finger meshes
# --------------------------------------------------------------------------
HOUSING_DIA = 60.0            # clean cylinder, no bosses, X = -76.65 .. -15.65
HOUSING_X0 = -76.65           # rear end of the gripper_link mesh
RAIL_BACK_X = -15.65          # rail back face: the axial datum
RAIL_FRONT_X = 0.0            # rail front face
RAIL_HALF_Y = 54.16           # rail end ("side") faces, the only flats we can key to
RAIL_HALF_Z = 25.88           # rail top/bottom flats
RAIL_TILT_DEG = 0.47          # measured: the rail prism is rotated this much about X.
                              # Over the |Z| <= 6 mm the key uses, that is 0.1 mm.

# Finger carriages run *inside* the rail.  These bound both carriages over the
# whole 0..50 mm jaw half-opening, and they are what the locating features on
# this bracket have to dodge.
CARRIAGE_BACK_X = -13.60      # rearmost carriage point: 2.05 mm in front of the datum
CARRIAGE_MIN_ABS_Z = 10.62    # carriages never come closer than this to the Z=0 plane
CARRIAGE_MAX_ABS_Y = 51.89    # at jaw half-opening 50 mm
# The carriage's outboard face sits 54.16 - 51.89 = 2.27 mm inside the rail's end
# face, at every jaw opening, over |Z| = 10.62..22.99.  So anything keying on a
# rail end face above |Z| = 10.62 is within 2.3 mm of a carriage in Y alone: the
# 4 mm gate confines the anti-rotation key to |Z| <= ~7.5 mm.  That is why the
# key has a short lever arm about X and why the grub screw, not the fit, is what
# makes the clocking deterministic.  See G4 in checks.py.
FINGER_FRONT_X = -0.70        # forwardmost carriage point behind the rail front face
TCP_X = 45.0                  # sim TCP site (a1x.xml)
FINGERTIP_X = 78.36           # forwardmost point of the finger meshes
JAW_PAD_ABS_Z = 6.96          # gripping pad half-height over X = 60..78

# --------------------------------------------------------------------------
# Clamp band
# --------------------------------------------------------------------------
# Printed bore = housing + 0.6 mm diametral fit clearance + room for a 0.5 mm
# liner on each side.  The liner (rubber/TPU sheet) is what actually grips; the
# 0.6 mm lets the two halves pull down onto it instead of bottoming out.
LINER_T = 0.5                 # rubber/TPU sheet between the bore and the housing
BORE_FIT_CLEARANCE = 0.6      # diametral
BORE_DIA = HOUSING_DIA + BORE_FIT_CLEARANCE + 2 * LINER_T   # 61.6
CLAMP_WALL = 4.2              # >= min wall, and keeps the band OD at r = 35.0
CLAMP_OD = BORE_DIA + 2 * CLAMP_WALL                         # 70.0
CLAMP_X0 = -52.0              # band spans 28 mm of the 61 mm cylinder ...
CLAMP_X1 = -24.0              # ... and stops 8.35 mm behind the rail datum
# Each half has to travel (BORE_FIT_CLEARANCE / 2) = 0.3 mm radially to reach the
# liner.  At 0.4 mm per side that left 0.1 mm before the halves bottomed on each
# other -- inside normal PETG scatter on a 61.6 mm bore.  0.8 leaves 0.5 mm.
SPLIT_GAP = 0.8               # per side, so the halves close onto the liner
# The crown of the printed bore droops.  Relieve it radially over the top
# +-BORE_RELIEF_DEG of each arch so the droop lands in the relief and the clamp
# bears on the flanks, which print as walls.
BORE_RELIEF = 0.4
BORE_RELIEF_DEG = 25.0

# M4 clamp ears, on the Z = 0 split plane at +-Y: the one neighbourhood of the
# housing that neither a held bottle nor the fingers ever reach.
BOLT_Y = 43.0
BOLT_DIA = 4.4                # M4 clearance
EAR_Y0, EAR_Y1 = 31.5, 50.6   # inboard edge overlaps the band OD so the part is one solid
EAR_X0, EAR_X1 = -47.0, -29.0
EAR_HALF_Z = 9.0
NUT_ACROSS_FLATS = 7.3        # M4 hex nut 7.0 + 0.3 print fit
NUT_DEPTH = 3.4
# The nut pocket opens through the ear's *outside* face, so the nut drops in
# after printing.  A pocket with a 0.6 mm printed floor over it is a sealed void
# that needs a pause at height, which nothing in an STL can tell the printer.
COUNTERBORE_DEPTH = 4.2       # >= the 4.0 mm head of an M4 SHCS, so it sits flush
COUNTERBORE_DIA = 7.4         # M4 SHCS head 7.0 + 0.4; 8.4 left only 0.43 mm
                              # of wall to the alignment dowel beside it
M4_LENGTH = 16.0              # head underside to tip; see checks.fastener_stack
# The camera-side ear has no captive nut, and that is forced, not chosen.  The
# strut's root stands over that ear -- which is already why the bolt is turned over
# and driven from underneath -- and a nut has to get into its pocket as well as a
# key onto the head.  Swept as a solid, all six straight lines out of that pocket
# are blocked: +Z by 1044 mm3 of strut and plate, -Z by the ear's own floor, +Y and
# -Y by the tie lug and the clamp band, +X by the locating yoke's stop horn and -X
# by the alignment dowel, which sits 0.7 mm off the channel a nut would need.  So
# that joint is an M4 tapped straight into the upper ear.  The called-out M4 x 16
# engages 10.4 mm of it, 2.6 x the thread diameter, against the 2 x D that is the
# usual floor for a thermoplastic.  The far-side joint keeps its steel nut, and
# gains a channel so that nut can still be got in.  checks.fastener_seating
# measures the interference, the hole, the insertion path and the thread.
M4_TAP_DIA = 3.3              # tapping drill for an M4 thread in PETG
M4_THREAD_ENGAGE_MIN = 8.0    # >= 2 x D of printed thread, or the joint is not one
DOWEL_DIA = 3.0               # two alignment pins across the split plane. 3 mm,
                              # not 4: at 4 mm there is nowhere in the ear that
                              # keeps MIN_WALL to the counterbore and 4 mm to
                              # the housing at the same time. Both halves are
                              # keyed to the rail anyway; the pins are extra.
DOWEL_X = -42.5               # in the bolt ears, MIN_WALL from the ear's end
                              # face, from the bolt hole and from the bore.
                              # (-43.5, 38.0) left 1.4 mm to the end face.
DOWEL_Y = 36.5                # 4.1 mm to the bore, 2.6 mm to the counterbore,
                              # 5.0 mm to the housing
DOWEL_LENGTH = 10.0           # the pin the BOM calls out
DOWEL_SLACK = 0.2             # hole past the pin's two ends, once the joint closes
# The hole depth is NOT a round number, and the round number it used to be did not
# work.  At DOWEL_DEPTH = 5.0 each half had 5.0 - SPLIT_GAP = 4.2 mm of blind hole,
# 8.4 mm of hole for a 10 mm pin -- so the pin held the two ears 1.6 mm apart, which
# is the *as-modelled* gap, and the clamp never travelled the 0.3 mm a side it has
# to travel to reach the liner.  Measured in the clamped pose the pin drove 0.60 mm
# into the two halves, exactly the travel.  Every gate passed it, because every gate
# looked at the as-modelled pose, where the pin sits in free air in both holes.
#
# So the depth is derived from the pin, from the gap the joint closes to, and from a
# little slack, and ``checks.clamped_assembly`` measures the closed pose as well as
# the open one.
CLAMPED_EAR_GAP = 2 * SPLIT_GAP - BORE_FIT_CLEARANCE          # 1.0 mm, closed
DOWEL_DEPTH = round(SPLIT_GAP
                    + (DOWEL_LENGTH - CLAMPED_EAR_GAP + DOWEL_SLACK) / 2, 2)  # 5.4

# --------------------------------------------------------------------------
# Locating yoke: axial stop + anti-rotation key  (gate G4)
# --------------------------------------------------------------------------
# Two horns run forward from the band at +-Y, confined to |Z| <= YOKE_HALF_Z.
# That band of the rail back face is free of carriage (carriages start at
# |Z| = 10.62), so the stop clears the fingers by sqrt(2.05^2 + 4.62^2) = 5.05
# mm instead of the 2.05 mm a full-height stop would get.
YOKE_HALF_Z = 6.0
YOKE_X0 = -27.0               # overlaps the band so the horn is not a add-on
YOKE_OUTER_Y = 58.7           # outer face of the horn / key ear
STOP_Y0 = 30.2                # inboard edge of the stop pad (bore radius at |Z|=6)

# Anti-rotation key: ears hugging both rail end faces.  Across-flats on the
# only non-round feature of the gripper.  They stop 5 mm short of the finger
# blade roots that swing out past the rail at X > 0.
#
# KEY_HALF_Z is the ear's own Z half-height and is *not* YOKE_HALF_Z: the stop
# pads are limited to 6.0 mm by the carriage's back face at X = -13.60, but the
# ears sit beside the rail rather than in front of it, so they may run 1 mm
# higher before the 4 mm finger gate binds (measured: 4.44 mm at 7.0).
KEY_HALF_Z = 7.0
KEY_CLEARANCE = 0.15          # per side.  The printed gauge exists to prove it;
                              # the ear faces are meant to be dressed to fit.
KEY_INNER_Y = RAIL_HALF_Y + KEY_CLEARANCE                    # 54.31
KEY_FRONT_X = -5.0
# M3 grub screw through the camera-side ear.  Tapped straight into 4.2 mm of
# PETG, so 2.5 mm core, not the 3.2 mm clearance diameter.  Running it in presses
# the opposite ear's inner face flat against the rail's far end face, which is
# what actually fixes the clocking -- see G4.
GRUB_DIA = 2.5
GRUB_DEPTH = 8.0
GRUB_X = -10.5

# --------------------------------------------------------------------------
# Camera pose  (gates G1/G2'/G3/G5)
# --------------------------------------------------------------------------
# Position of the centre of the camera PCB's FRONT face (the face the M12 lens
# holder sits on).  +Y side, low and well behind the rail: a held bottle is a
# |Y| <= 42 mm slab that leans back over the housing, so this is outside it in
# Y and behind it in X, and the swept radius stays small.
CAM_POS = (-29.0, 51.0, 48.0)
CAM_YAW_DEG = 32.0            # optical axis yawed inboard (toward -Y for a +Y mount)
CAM_PITCH_DEG = 25.0          # ... and pitched down; 25..50 deg is the G3 window
CAM_ROLL_DEG = 0.0            # image up stays close to +Z
HANDEDNESS = 1                # +1 = camera on +Y.  Mirror in Y for the other hand.

# --------------------------------------------------------------------------
# Camera module envelope (32 x 32 mm UVC board camera, SO-101 style)
# --------------------------------------------------------------------------
PCB_SIZE = 32.0
PCB_T = 1.6
# Rear envelope, along the optical axis from the PCB *front* face.  This is not a
# wish: it is exactly where the camera plate's front face is, so a module built to
# it seats.  The central relief pocket gives PLATE_RELIEF_DEPTH more inside a
# PLATE_RELIEF square.  checks.g7_module_fit asserts the two agree, and the README
# quotes these two numbers rather than one round one.
STANDOFF_H = 3.0              # insulating standoff under the PCB
BODY_BACK = -(PCB_T + STANDOFF_H)          # -4.6: the plate's front face
BODY_FRONT = 6.0              # M12 holder block
HOLDER_SIZE = 22.0            # the holder block is a boss, not the whole board
LENS_DIA = 16.0
LENS_FRONT = 18.0             # front vertex of the barrel, from the PCB front face
PUPIL_AHEAD = 12.0            # entrance pupil, 6 mm behind the barrel front vertex
M2_PITCH_MIN, M2_PITCH_MAX = 26.0, 30.0
# Connector keep-out: off the board's lower edge, offset toward image-left.  The
# offset is load-bearing, not cosmetic: the strut has to reach the plate from
# below (every other direction runs into the bottle slab, the arm links or the
# swept-radius gate), so the connector cannot sit in the middle of that edge.
USB_KEEPOUT = (12.0, 8.0, 25.0)   # width x thickness x length
USB_U_OFFSET = -10.0              # image-u of the connector's centre

RESOLUTION = (640, 480)

# Field of view.  A 4:3 sensor cannot have an arbitrary (H, V) pair: with square
# pixels the horizontal angle follows from the vertical one and the aspect, and
# MuJoCo drives a fixed camera from ``fovy`` alone.  So only the vertical angle is
# a lens choice here, and the horizontal one is derived -- quoting a nominal
# "110 x 85" would be 8.6 deg wider than any 640 x 480 module or renderer gives.
ASPECT = RESOLUTION[0] / RESOLUTION[1]


def fov_from_vertical(vertical_deg, aspect=ASPECT):
    """(horizontal, vertical) full angles of a rectilinear lens on this sensor."""
    import math
    half = math.radians(vertical_deg / 2.0)
    return (round(2 * math.degrees(math.atan(aspect * math.tan(half))), 2),
            float(vertical_deg))


FOV_WIDE = fov_from_vertical(85.0)      # M12 ~2.1 mm: 101.4 x 85, the recommended lens
FOV_NARROW = fov_from_vertical(55.0)    # M12 ~3.6 mm: 69.53 x 55, the alternative
FOV_RECOMMENDED, FOV_ALTERNATIVE = FOV_WIDE, FOV_NARROW
CAMERA_MASS_G = 22.0          # board + M12 lens + holder, typical
PIGTAIL_MASS_G = 6.0

# --------------------------------------------------------------------------
# Camera plate and strut
# --------------------------------------------------------------------------
# 35 mm, square-cornered, and the four M2 lugs swallow its corners.  Both numbers
# are forced:
#
#   * the M2 slots for the 30 mm pattern reach 16.3 mm in u and in v, and MIN_WALL
#     around them needs material out to 18.7 -- which the lugs provide locally
#     rather than the plate providing everywhere, because
#   * the plate's (+u, -v) corner is the payload's most negative X, and G1's 10 mm
#     bound on arm links 3-6 has a quarter of a millimetre to give.
#
# 35 mm also puts the plate's corner (17.5, 17.5) *inside* a lug, so the lug's arc
# crosses the plate's edges transversally instead of grazing its corner.  At 36 mm
# the corner sat 0.24 mm outside the arc and the near-tangency left three unclosed
# patches in the tessellation -- the STL came out non-watertight.
PLATE_SIZE = 35.0
PLATE_T = 4.5
PLATE_CHAMFER = 0.0
M2_CAP_R = 1.3                # half the slot's width: the end cap's radius
PLATE_RELIEF = 18.0           # central pocket for rear components, inboard of the pads
PLATE_RELIEF_DEPTH = 2.0      # <= PLATE_T - MIN_WALL, so the pocket floor is a wall
STRUT_W = 18.0                # root section, wide axis
STRUT_H = 16.0
STRUT_TIP_W = 12.0            # tip section: tapered, so it threads past the board
STRUT_TIP_H = 10.0
# Where on the plate the strut lands, in image (u, v) millimetres.
#
# NOT the plate's centre, and this is the one thing about the layout that is not
# free.  The clamp band sits inside the camera's own forward hemisphere -- the lens
# looks down and inboard, which is where the band is -- so a straight strut aimed
# at the plate's centre runs through the module to get there.  Measured on the
# first cut of this part: 375 mm2 of bracket inside the PCB's own footprint and
# material 1 mm past the lens's front vertex.  No plate thickness or standoff
# height fixes it; the strut has to land off to one side.
#
# Which side is decided by three gates, not by taste: +u walks into the bottle
# slab (|Y| <= 42 mm), -u and +v cross forward of KEEPOUT_X and cost swept radius,
# and -v is the only direction that moves the payload *away* from the roll axis's
# -X end, where G1's 10 mm arm-link bound has 0.1 mm to spare.  So: below the
# camera, slightly image-left.  checks.g7_module_fit measures what is left: the
# bracket's intersection with the module's own envelope, which must be zero.
STRUT_TIP_UV = (-10.0, -29.0)
PLATE_TONGUE = 9.0            # how far the plate runs on past the strut's tip
# Spine on the plate's rear face.  The plate is the soft link, not the strut: 30 mm
# of 4.5 mm plate between the camera's four screws and the strut puts the first mode
# at 50 Hz, under G6's 80.  A 12 x 6 mm rib along the span makes the section a T and
# takes it to about 180 Hz.  See checks.g6_stiffness.
SPINE_W = 12.0
SPINE_H = 6.0
SPINE_BACK = 22.0             # how far past the module's centre the spine runs
ZIPTIE_SLOT = (4.0, 2.2)      # length along the strut x width, of the tie slot
ZIPTIE_WEB = 2.9              # material outboard of the slot; >= MIN_WALL
# The tie lug sits on the strut flank that faces the USB exit, so the cable runs
# from the connector down the strut to the clamp band rather than across the
# bottle's corridor.  The first cut put it on the far flank near the tip: 39 mm
# from the connector, in the opposite direction, and 4 mm forward of the keep-out
# the README claimed nothing crossed.
#
# WHICH flank is no longer a free choice either: design.ziptie_flanks applies two
# rules -- the whole lug above +SPLIT_GAP, and behind KEEPOUT_X -- before it looks
# at the connector at all.  Each rule is there because a revision broke it: the
# first cut sat on the +X flank and reached X = -19.97, forward of the keep-out the
# README promised nothing crossed, and this one sat on the -Z flank at 11.0 mm
# along, where it reached Z = -1.94.
#
# HOW FAR along is then forced by the same split plane.  The -Z flank hangs off the
# underside of a strut whose root is only 11.27 mm up, the lug stands 10.5 mm off
# the strut's axis and its own Z half-extent is 6.51 mm, so it clears +SPLIT_GAP
# only from 17.7 mm along, and runs into the camera module's envelope from 21.6 mm.
# 19.0 is the middle of that window: 0.52 mm of air under the lug, 2.6 mm to the
# module, and the cable run from the connector to the slot drops from 24.9 mm to
# 17.0.  checks.split_plane_confinement and checks.g7_module_fit are the two gates.
ZIPTIE_ALONG = 19.0           # distance from the strut root, along the strut
# The strut's root grazes the top-outboard corner of the camera-side bolt ear, and a
# graze leaves a 1 mm sliver of material that no nozzle can lay down.  This block
# fills it: it is inside both the ear and the strut, so the union is a fillet rather
# than a tangency, and it braces the strut's root against the ear on the way.
BLEND_X = (-44.0, -34.0)
BLEND_Y = (44.0, EAR_Y1)
BLEND_Z = 13.0
KEEPOUT_X = -24.0             # nothing above the housing forward of here ...
KEEPOUT_R = 30.0              # ... outside this radius from the roll axis ...
BOTTLE_SLAB_Y = 42.0          # ... unless it is outside the held bottle's slab

# --------------------------------------------------------------------------
# Print / material
# --------------------------------------------------------------------------
LAYER_H = 0.2                 # also the bed-contact tolerance in the overhang check
NOZZLE = 0.4                  # 4 perimeters -> 1.6 mm of solid wall everywhere
MIN_WALL = 2.4                # G6's written clause; checks.wall_thickness gates it
# The solids are tessellated at 0.05 mm before anything is measured on them, and a
# ray cast inward from a sampled point is normal to a triangle, not to the true
# surface.  A wall whose nominal size is exactly MIN_WALL therefore reads 2.36-2.40.
# The gate carries that tolerance explicitly rather than by rounding.
TESSELLATION = 0.05
# Features allowed to be thinner than MIN_WALL live in design.THIN_EXCEPTIONS, as
# regions with their own floor and their own reason. There is one.
BED = (220.0, 220.0)
GAUGE_WEB = 2.6               # back web of the rail-key gauge; >= MIN_WALL
# The wall that really binds around an M2 slot is not the diagonal one to the plate's
# corner -- the formula the first cut used, which read 2.94 mm where the truth was
# 1.70 -- it is the straight-line distance from the slot's outer end cap to the
# plate's own edge.  Growing the plate until that is MIN_WALL would break G1's 10 mm
# arm-link bound, so each slot's outer end sits in its own lug instead.
# checks.g7_plate_slots walks the slot's outline against the built solid.
M2_PAD_HALF_W = M2_CAP_R + MIN_WALL + 0.1     # wall the M2 screw bears against
M2_LUG_R = M2_PAD_HALF_W + 0.3                # 0.3 more, so the lug's cylinder and
                                              # the pad's are not coincident faces
PETG_DENSITY = 1.27e-3        # g/mm^3, solid
EFFECTIVE_FILL = 0.92         # 4 perimeters + 40 % gyroid on these wall thicknesses
PETG_E = 1800.0               # MPa, printed PETG along the layers (conservative)

# --------------------------------------------------------------------------
# Gate thresholds (verify.py asserts against these)
# --------------------------------------------------------------------------
G1_BOTTLE_CLEARANCE = 4.0
G1_BOTTLE_SAMPLES = 320
G1_MESH_CLEARANCE = 4.0       # non-locating payload vs real gripper/finger meshes
G1_ARM_CLEARANCE = 10.0
G2_OBJECT_ZONE_Z = 10.0       # G2'b sample height above the blades
G2_FAR_JAW_FRACTION = 0.90    # G2'c
G2_GRASP_ZONE_FRACTION = 0.60 # G2'd, openings >= 40 mm
G3_FORWARD_MM = 250.0         # past the fingertips, on the tool axis
G3_PITCH_RANGE = (25.0, 50.0)
G4_STOP_AREA_MM2 = 300.0
G5_SWEPT_RADIUS = 100.0
G5_V4_BOARD_COLLISIONS = 64
G6_FIRST_MODE_HZ = 80.0
G7_TOPDOWN_MARGIN = 25.0      # payload above the fingertip plane on a top-down pick
G7_MODULE_FIT_MM3 = 1.0       # printed material inside the bought module's envelope
DRIVER_REACH_MM = 40.0        # clear shaft above a bolt head, for a hex key
G8_SWEPT_RADIUS = 125.0       # the webcam variant's allowance
# G9: the assembly invariants.  Added after a review found that thirty gates
# measured every clearance to the robot and none of the payload against itself:
# the zip-tie lug hung 2.74 mm below the split plane, 12.9 mm3 of the upper bracket
# sat inside the lower strap, and the exported STL stood on that nub with its
# 1182 mm2 split face 2.65 mm off the bed.  All four numbers are hard.
G9_SPLIT_LEAK_MM3 = 0.0       # material of a printed half on the wrong side of Z = 0
G9_ASSEMBLY_CLASH_MM3 = 0.0   # upper_bracket ^ lower_bracket, assembled
G9_BED_CONTACT_FRACTION = 0.60  # of the part's own split-face area, on the STL
G9_FASTENER_ENGAGE_MM = 3.0   # least hole a bolt, nut or dowel may sit in
# G9f: the same three invariants again with the joint CLOSED.  Added after a review
# found that every G9 gate measured the as-modelled pose, in which the halves are
# 1.6 mm apart and a 10 mm dowel in 8.4 mm of hole sits in free air -- so the pin
# that stopped the clamp reaching the liner passed all of them.
G9_CLAMPED_CLASH_MM3 = 0.0
# G9g: the left hand is shipped as a mirror of the right, so the native-left build
# has to be that mirror and not a second, differently-wrong part.  It was not: the
# camera plate's tongue, spine and M2 lugs read their v direction off a hard-coded
# sign that is only right for the +Y hand, and design.payload_parts(side=-1) came
# out 4470 mm3 different from mirror(right), with a 106.8 mm swept radius against
# the 100 mm gate.  Nothing shipped used that path; the gate is so nothing does.
G9_MIRROR_DIFF_MM3 = 0.0
# G9h: overhang that is NOT the bore arch and has nothing underneath it.  The arch
# is self-supporting and is reported separately; this is the part of "supports:
# none" that is a claim about the strut root and the camera plate, so it is the part
# that is gated.  Revision 6 measures 441 mm2 -- the number is a ceiling on that
# revision, not a target.
G9_FREE_OVERHANG_MM2 = 500.0
# Free space the collision cover may claim.  Revision 5's cover measured
# 90.8 cm3; see the note in verify.py for why this is an absolute bound and
# not the ratio revision 5 quoted.
COVER_UNION_CM3 = 90.0
JAW_OPENINGS = (10.0, 20.0, 40.0, 70.0, 100.0)
ARM_AUDIT_SAMPLES = 10000
ARM_AUDIT_SEED = 20260919     # same seed and method as v4/arm_check.py
# The gate is one count of about 60 out of 10,000, which carries ~8 of sampling
# noise on its own.  These are the extra draws checks.g5_audit_spread takes so the
# gated number is reported with the spread it sits in, rather than as if 74 and 64
# were 10 apart in any meaningful sense.
ARM_AUDIT_SPREAD_SEEDS = (ARM_AUDIT_SEED, 1, 2, 3, 4, 5, 6, 7)
BOTTLE_SEED = 20260920
# Every surface sampler gets a seed off this one.  trimesh's
# ``sample_surface_even`` draws from the global numpy RNG when it is not given
# one, and three call sites were not: validation.json's corridor census moved a
# few tenths of a millimetre per run, which is how the README came to quote
# 47.2 / 61.1 mm where the file said 46.78 / 60.25.  A rendered README can only
# be asserted against a file that does not move.
SAMPLE_SEED = 20260921
