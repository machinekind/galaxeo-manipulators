#!/usr/bin/env python3
"""Run every gate, write validation.json, exit non-zero if any gate fails.

    python verify.py                 # WRITES validation.json, stl/, step/,
                                     #   collision/, camera_spec.json, the GLB
    python verify.py --quick         # the same, minus the 10k arm audit, into
                                     #   validation.quick.json (about 3 minutes)
    python verify.py --check         # writes NOTHING: runs every gate, exports
                                     #   into a temp directory and diffs it
                                     #   against the tracked files; exit 1 on
                                     #   drift as well as on a failed gate

Which command writes what is the whole of it: `verify.py` and
`python -m mount.export` are the two writers, `verify.py --check` and
`python -m mount.export --check` are their read-only twins, and
`python -m mount.render` rewrites the four README images.  `sim/` has its own
pair; see its docstring.

Nothing here prints a pass it did not measure: each gate records the number it
measured, the threshold it was held to and the verdict, and the exit code is
the AND of the verdicts.  Gates the design does not meet are reported as
failures, not renamed.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mount import checks, design, export, params as P, robot as rb, v4ref   # noqa: E402

RESULTS = []


def gate(name, ok, measured, threshold, detail=None):
    """Record one gate's verdict and return it."""
    RESULTS.append(dict(gate=name, passed=bool(ok), measured=measured,
                        threshold=threshold, detail=detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {measured} (threshold {threshold})", flush=True)
    return bool(ok)


def check_measurements():
    """The params the design is built on must still match the vendor meshes."""
    measured = rb.measure_gripper()
    expected = dict(housing_dia=P.HOUSING_DIA, housing_x0=P.HOUSING_X0,
                    rail_back_x=P.RAIL_BACK_X, rail_front_x=P.RAIL_FRONT_X,
                    carriage_back_x=P.CARRIAGE_BACK_X,
                    carriage_min_abs_z=P.CARRIAGE_MIN_ABS_Z,
                    carriage_max_abs_y=P.CARRIAGE_MAX_ABS_Y,
                    finger_front_x=P.FINGER_FRONT_X, fingertip_x=P.FINGERTIP_X)
    drift = {k: [measured[k], v] for k, v in expected.items()
             if abs(measured[k] - v) > 0.02}
    # rail half-sizes are quoted to 2 dp in params
    for key, value in (("rail_half_y", P.RAIL_HALF_Y), ("rail_half_z", P.RAIL_HALF_Z)):
        if abs(measured[key] - value) > 0.02:
            drift[key] = [measured[key], value]
    gate("mesh measurements match params.py", not drift, measured, expected, drift)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the 10,000 configuration full-arm audit")
    parser.add_argument("--bottles", type=int, default=P.G1_BOTTLE_SAMPLES)
    parser.add_argument("--out", help="where to write the report "
                                     "(default: validation.json, or "
                                     "validation.quick.json for a reduced run)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing: run every gate, export into a temp "
                             "directory and report any tracked file that differs")
    args = parser.parse_args()
    reduced = args.quick or args.bottles != P.G1_BOTTLE_SAMPLES
    scratch = None
    if args.check:
        if args.out:
            parser.error("--check writes nothing, so --out means nothing with it")
        if reduced:
            # A reduced run writes validation.quick.json, which .gitignore keeps
            # out of the tree on purpose -- so there is nothing for --check to
            # compare it with and it could only ever exit 1.
            parser.error("--check compares against the tracked validation.json, "
                         "and a reduced run does not produce one; drop --quick "
                         "(and --bottles) or drop --check")
        scratch = Path(tempfile.mkdtemp(prefix="g1_verify_check_"))
        export.set_output_root(scratch)

    started = time.time()
    cam = design.CameraFrame()
    meshes = checks.payload_meshes(cam)
    report = {"revision": 6,
              # A --quick run skips the 10,000-configuration audit and may reduce
              # the bottle count, so it must not be mistaken for the record of a
              # full one. It says so here and it writes a different file.
              "reduced_run": bool(reduced), "quick": bool(args.quick),
              "bottle_samples": args.bottles}

    print("measurements")
    report["gripper_measurements"] = check_measurements()

    print("G1  held bottle, real meshes, arm links")
    bottles = checks.g1_bottle_in_hand(meshes, samples=args.bottles)
    report["G1_bottle_in_hand"] = bottles
    gate("G1 bottle collisions", bottles["collisions"] == 0,
         bottles["collisions"], 0)
    gate("G1 bottle clearance", bottles["min_clearance_mm"] >= P.G1_BOTTLE_CLEARANCE,
         bottles["min_clearance_mm"], f">= {P.G1_BOTTLE_CLEARANCE}")

    gripper = checks.g1_gripper_meshes(meshes)
    report["G1_gripper_meshes"] = gripper
    gate("G1 payload vs gripper and finger meshes",
         gripper["min_free_clearance_mm"] >= P.G1_MESH_CLEARANCE,
         gripper["min_free_clearance_mm"], f">= {P.G1_MESH_CLEARANCE}")

    locating = checks.g1_locating_features(meshes)
    report["G1_locating_features"] = locating
    gate("G1 locating features vs the finger meshes",
         locating["min_clearance_mm"] >= P.G1_MESH_CLEARANCE,
         locating["min_clearance_mm"], f">= {P.G1_MESH_CLEARANCE}")

    arm = checks.g1_arm_links(meshes)
    report["G1_arm_links"] = arm
    gate("G1 payload vs arm links 3-6 over the full wrist range",
         arm["min_mm"] >= P.G1_ARM_CLEARANCE, arm["min_mm"],
         f">= {P.G1_ARM_CLEARANCE}")

    # Reported, not gated: see checks.bottle_corridor_census.  G1 above is the gate.
    report["bottle_corridor"] = checks.bottle_corridor_census(meshes)
    print(f"       (corridor census: "
          f"{len(report['bottle_corridor']['parts_forward_of_keepout'])} parts "
          f"forward of X = {P.KEEPOUT_X:.0f} outside r = {P.KEEPOUT_R:.0f}, "
          f"closest to the bottle slab at |Y| = "
          f"{report['bottle_corridor']['min_abs_y_forward_mm']} mm)")

    print("G2' sightlines")
    sight = checks.g2_sightlines(cam, meshes)
    report["G2prime_sightlines"] = sight
    tips_ok = all(entry["visible"] for opening in sight["a_blade_tips"].values()
                  for entry in opening.values())
    gate("G2'a both blade tips visible at every opening", tips_ok,
         "all visible" if tips_ok else "one or more blocked", "all visible")
    zone_blocked = sum(v["blocked"] for v in sight["b_object_zone"].values())
    gate("G2'b object zone above the blades", zone_blocked == 0, zone_blocked, 0)
    far = {k: v["visible_fraction"] for k, v in sight["c_far_jaw"].items()}
    far_gated = min(v for k, v in far.items() if float(k[7:]) >= 20)
    gate("G2'c far jaw visible, openings >= 20 mm",
         far_gated >= P.G2_FAR_JAW_FRACTION, far_gated,
         f">= {P.G2_FAR_JAW_FRACTION}")
    grasp = {k: v["visible_fraction"] for k, v in sight["d_grasp_zone_z0"].items()}
    grasp_gated = min(v for k, v in grasp.items() if float(k[7:]) >= 40)
    gate("G2'd Z=0 grasp zone visible, openings >= 40 mm",
         grasp_gated >= P.G2_GRASP_ZONE_FRACTION, grasp_gated,
         f">= {P.G2_GRASP_ZONE_FRACTION}")
    print(f"       (ungated: Z=0 visible fraction {grasp['opening10']} at 10 mm, "
          f"{grasp['opening20']} at 20 mm)")

    print("G3  forward view, both candidate lenses")
    report["G3_forward_view"] = {}
    report["G3_occlusion"] = {}
    lenses = (("recommended_wide", P.FOV_RECOMMENDED),
              ("alternative_standard", P.FOV_ALTERNATIVE))
    for label, fov in lenses:
        view = checks.g3_forward_view(cam, fov)
        report["G3_forward_view"][label] = view
        report["G3_occlusion"][label] = checks.g3_occlusion(cam, fov)
        report["G3_occlusion"][label]["self_and_gripper"] = checks.self_occlusion(
            cam, meshes, fov)
        gate(f"G3 forward point and fingertips in frame ({label}, "
             f"{fov[0]:.1f} x {fov[1]:.0f} deg)",
             view["forward_point_in_frame"] and view["fingertips_in_frame"],
             f"margin {view['gated_margin_deg']} deg on the gated targets "
             f"({view['worst_margin_deg']} deg including the jaw-100 extras)",
             "both in frame")
    # Two readings of "camera pitch relative to the tool axis": the component below
    # the axis, and the total angle off it.  Both are restatements of params -- this
    # gate is a range check on a design input, not a measurement -- and the first
    # sits exactly on the window's lower bound, so both are recorded.
    view = report["G3_forward_view"]["recommended_wide"]
    pitch, off_axis = view["pitch_deg"], view["total_off_axis_deg"]
    report["G3_pitch"] = dict(
        pitch_below_tool_axis_deg=pitch, total_off_axis_deg=off_axis,
        window_deg=list(P.G3_PITCH_RANGE),
        margin_deg=[round(pitch - P.G3_PITCH_RANGE[0], 2),
                    round(P.G3_PITCH_RANGE[1] - pitch, 2)],
        note="a range check on params.CAM_PITCH_DEG, not an independent "
             "measurement; the pitch component sits on the window's lower bound "
             "with no margin, the total off-axis angle has 14.8 deg")
    gate("G3 camera pitch below the tool axis",
         P.G3_PITCH_RANGE[0] <= pitch <= P.G3_PITCH_RANGE[1]
         and P.G3_PITCH_RANGE[0] <= off_axis <= P.G3_PITCH_RANGE[1],
         f"{pitch} deg below the axis, {off_axis} deg off it",
         f"both in {P.G3_PITCH_RANGE[0]}..{P.G3_PITCH_RANGE[1]} deg")

    print("G4  repeatable pose")
    stop = checks.g4_stop_area(cam)
    slop = checks.g4_clocking_slop(cam)
    pupil_radius = float((cam.pupil[1] ** 2 + cam.pupil[2] ** 2) ** 0.5)
    report["G4_datum"] = dict(
        axial_stop=f"four pads on the rail back face at X = {P.RAIL_BACK_X}, "
                   f"|Z| <= {P.YOKE_HALF_Z} mm, "
                   f"|Y| = {P.STOP_Y0}..{P.KEY_INNER_Y} mm",
        anti_rotation=f"ears across both rail end faces (|Y| = {P.RAIL_HALF_Y} mm), "
                      f"the only non-round feature on the gripper, over "
                      f"|Z| <= {P.KEY_HALF_Z} mm",
        nominal_fit_mm=P.KEY_CLEARANCE,
        grub_screw=f"M{3} tapped into the camera-side ear at X = {P.GRUB_X} "
                   f"({P.GRUB_DIA} mm core, {P.GRUB_DEPTH} mm deep). Running it in "
                   f"presses the far ear's inner face flat against the far rail end "
                   f"face; that plane-on-plane contact, not the fit, is what fixes "
                   f"the clocking. Mirrored to -Y on the left-hand build.",
        clearance_to_fingers_mm=locating["min_clearance_mm"],
        pupil_radius_from_roll_axis_mm=round(pupil_radius, 1),
        pupil_displacement_per_slop_mm=round(
            pupil_radius * slop["total_slop_deg"] * 3.14159265 / 180.0, 2),
        **stop, clocking=slop)
    report["G4_datum"]["clocking"]["pupil_radius_mm"] = round(pupil_radius, 1)
    gate("G4 positive axial stop and anti-rotation key",
         locating["min_clearance_mm"] >= P.G1_MESH_CLEARANCE
         and stop["stop_area_mm2"] > P.G4_STOP_AREA_MM2,
         f"{stop['stop_area_mm2']} mm2 of measured stop contact, "
         f"{locating['min_clearance_mm']} mm to the fingers, "
         f"{slop['total_slop_deg']} deg of clocking slop without the grub screw",
         f"> {P.G4_STOP_AREA_MM2} mm2 and >= {P.G1_MESH_CLEARANCE} mm")

    print("G5  swept radius and full-arm audit")
    swept = checks.g5_swept_radius(meshes)
    report["G5_swept_radius"] = swept
    # v4's 118 / 143 mm, recomputed from v4's own collision STLs out of the commit
    # that added them, rather than quoted from a README nobody can re-derive.
    report["v4_reference"] = v4ref.reference()
    if report["v4_reference"]["available"]:
        print("       (v4 at " + report["v4_reference"]["commit"] + ": "
              + ", ".join(f"{k} {v['swept_radius_mm']} mm"
                          for k, v in report["v4_reference"]["variants"].items())
              + ")")
    gate("G5 payload swept radius about the roll axis",
         swept["swept_radius_mm"] <= P.G5_SWEPT_RADIUS, swept["swept_radius_mm"],
         f"<= {P.G5_SWEPT_RADIUS}")
    if args.quick:
        print("  [SKIP] G5 full-arm audit (--quick)")
    else:
        audit = checks.g5_arm_audit(meshes)
        report["G5_arm_audit"] = audit
        # v4's 64 is one hand's number -- it only ever had one -- so the gate is the
        # same hand, and the mirrored one is measured and reported beside it rather
        # than folded into a "worse hand" figure that v4 has no counterpart for.
        # It is not symmetric: the arm is not, so the mirrored payload meets it
        # differently. The left hand's 74 is above v4's 64 and that is a real cost of
        # building the left one; see the handedness note in the README.
        right = audit["per_hand"]["right"]["payload_collisions"]
        left = audit["per_hand"]["left"]["payload_collisions"]
        gate(f"G5 payload collisions in {P.ARM_AUDIT_SAMPLES:,} random "
             f"configurations ({checks.GATE_HAND} hand, like for like with v4; "
             f"the other hand is reported, not gated)",
             right < P.G5_V4_BOARD_COLLISIONS,
             f"{right} right / {left} left",
             f"< {P.G5_V4_BOARD_COLLISIONS} (v4) on the {checks.GATE_HAND} hand")
        print(f"       (per part, right hand: "
              f"{audit['per_hand']['right']['per_part_configs']})")
        # Reported, not gated: how much of the right/left difference is sampling.
        spread = checks.g5_audit_spread(meshes)
        report["G5_arm_audit_spread"] = spread
        print(f"       (over {len(spread['seeds'])} seeds: right "
              f"{spread['right']['mean']} +- {spread['right']['sd']}, left "
              f"{spread['left']['mean']} +- {spread['left']['sd']}, "
              f"left - right {spread['left_minus_right']['mean']} +- "
              f"{spread['left_minus_right']['sd']})")
        # Reported, not gated: what each attempt to bring the left hand under the
        # reference actually buys, and the gate it spends. These were four
        # sentences of README prose that no run produced, and a reviewer who
        # reran them found two of the four figures wrong and a third unmoved.
        what_if = checks.g5_what_if()
        report["G5_arm_audit_what_if"] = what_if
        print(f"       (what-if: best left {what_if['best_left']} vs "
              f"{P.G5_V4_BOARD_COLLISIONS}; "
              + ", ".join(f"{r['name']} {r['left']}"
                          for r in what_if["per_case"][1:]) + ")")

    print("G7  print files, mass, collision primitives")
    report["parts"] = export.export_parts(cam)
    mass = checks.mass_properties(cam)
    report["mass_properties"] = mass
    validity = checks.mesh_validity(export.STL_DIR)
    report["stl_validation"] = validity
    report["printability"] = checks.overhang_report(export.STL_DIR)
    gate("G7 every STL watertight, one body, on the bed and inside the build volume",
         all(v["watertight"] and v["winding_consistent"] and v["single_body"]
             and v["sits_on_bed"] and v["fits_bed"] for v in validity.values()),
         f"{len(validity)} files", "all true")

    thickness = checks.wall_thickness(cam)
    report["wall_thickness"] = thickness
    rows = [v for v in thickness.values() if isinstance(v, dict)]
    floor = P.MIN_WALL - P.TESSELLATION
    thin = {k: v["min_wall_mm"] for k, v in thickness.items()
            if isinstance(v, dict) and v["min_wall_mm"] < floor}
    bad_exception = {k: e for k, v in thickness.items() if isinstance(v, dict)
                     for k2, e in v["declared_exceptions"].items() if not e["ok"]}
    gate("G6/G7 minimum wall on every printed part",
         not thin and not bad_exception,
         f"thinnest wall {min(v['min_wall_mm'] for v in rows)} mm"
         + (f", under the gate: {thin}" if thin else "")
         + (f"; declared exception below its floor: {bad_exception}"
            if bad_exception else ""),
         f">= {P.MIN_WALL} mm (less {P.TESSELLATION} mm of tessellation) outside "
         f"the {len(design.THIN_EXCEPTIONS)} declared exception(s)")

    print("G9  the payload against itself: split plane, part pairs, fasteners, bed")
    split = checks.split_plane_confinement(cam)
    report["G9_split_plane"] = split
    gate("G9a each printed half confined to its side of the split plane",
         split["total_leak_mm3"] <= P.G9_SPLIT_LEAK_MM3,
         ", ".join(f"{k} {v['leak_mm3']} mm3 past {v['allowed_from_mm']}"
                   + (f" at {v['leak_bbox_mm']}" if v["leak_bbox_mm"] else "")
                   for k, v in split["per_part"].items()),
         f"<= {P.G9_SPLIT_LEAK_MM3} mm3 outside "
         f"{len(design.SPLIT_INTERLOCKS)} declared interlock(s)")
    gate("G9b the two halves do not intersect when assembled",
         split["assembly_intersection_mm3"] <= P.G9_ASSEMBLY_CLASH_MM3,
         f"{split['assembly_intersection_mm3']} mm3",
         f"<= {P.G9_ASSEMBLY_CLASH_MM3} mm3")

    pairs = checks.part_pair_intersections(cam)
    report["G9_part_pairs"] = pairs
    gate("G9c no two payload solids share volume, except the declared pairs",
         not pairs["undeclared"] and not pairs["over_limit"]
         and not pairs["parts_missing"] and not pairs["parts_unexpected"],
         f"{len(pairs['intersecting'])} intersecting pair(s) of "
         f"{pairs['pairs_tested']} over {len(pairs['parts_tested'])} parts, "
         f"{len(design.DECLARED_OVERLAPS)} declared"
         + (f"; UNDECLARED: {pairs['undeclared']}" if pairs["undeclared"] else "")
         + (f"; over its declared bound: {pairs['over_limit']}"
            if pairs["over_limit"] else "")
         # A part that stops being built shrinks this gate instead of failing it,
         # and one did: the +Y dowel was never created, so the sweep covered 36
         # pairs of 9 parts where there are 45 of 10 and nothing said so.
         + (f"; MISSING PARTS: {pairs['parts_missing']}"
            if pairs["parts_missing"] else "")
         + (f"; unexpected parts: {pairs['parts_unexpected']}"
            if pairs["parts_unexpected"] else ""),
         f"every overlap declared in design.DECLARED_OVERLAPS and under its "
         f"bound, over all {len(design.PAYLOAD_PART_NAMES)} payload parts")

    seating = checks.fastener_seating(cam, engage_floor=P.G9_FASTENER_ENGAGE_MM)
    report["G9_fastener_seating"] = seating
    rows = seating["per_fastener"].values()
    blocked = max((max(v["insertion_blocked_mm3"].values(), default=0.0)
                   for v in rows if isinstance(v["insertion_blocked_mm3"], dict)),
                  default=0.0)
    gate("G9d every bolt, nut, dowel and camera screw fits, sits in a real hole "
         "and can be got in",
         not seating["failing"],
         f"{len(seating['per_fastener'])} fasteners: worst interference "
         f"{max(v['interference_mm3'] for v in rows)} mm3, least hole "
         f"{min(v['engaged_best_mm'] for v in rows)} mm, most material in an "
         f"insertion path {blocked} mm3, least tapped thread "
         f"{min((v['thread_engagement_mm'] for v in rows if v['thread_engagement_mm']), default=0.0)} mm"
         + (f"; FAILING: {seating['failing']}" if seating["failing"] else ""),
         f"0 mm3 of interference, >= {P.G9_FASTENER_ENGAGE_MM} mm of hole, "
         f"nothing in the way of any head or nut, and >= "
         f"{P.M4_THREAD_ENGAGE_MIN} mm of thread where a joint is tapped")

    # Why the tie lug sits where it does: the two admissibility rules, the four
    # candidate flanks and what each one would have cost.  Recorded because the
    # rule that picks it was written after two revisions picked the wrong one.
    report["ziptie_flank"] = design.ziptie_flanks(cam)[0]

    bed = checks.bed_contact(export.STL_DIR, cam)
    report["G9_bed_contact"] = bed
    gate("G9e each bracket's exported print pose rests on its split face",
         not bed["failing"],
         ", ".join(f"{k} {v['fraction']:.0%} of {v['split_face_mm2']} mm2 "
                   f"(lowest Z {v['lowest_z_mm']})"
                   for k, v in bed["per_file"].items()),
         f">= {P.G9_BED_CONTACT_FRACTION:.0%} of the split face on the bed")

    # The as-modelled pose is not the pose the part is used in.  Every gate above
    # measures the halves 1.6 mm apart; they close to 1.0 mm on the liner, and the
    # called-out dowel used to bottom out over exactly that 0.6 mm.
    clamped = checks.clamped_assembly(cam)
    report["G9_clamped_assembly"] = clamped
    gate("G9f the joint closes onto the liner: nothing rigid spans the ears",
         clamped["worst_interference_mm3"] <= P.G9_CLAMPED_CLASH_MM3,
         f"{clamped['worst_interference_mm3']} mm3 with the ears closed from "
         f"{clamped['modelled_ear_gap_mm']} to {clamped['clamped_ear_gap_mm']} mm "
         f"({clamped['dowel_length_mm']:.0f} mm pin in "
         f"{2 * clamped['hole_per_half_mm']:.1f} mm of hole, "
         f"{clamped['per_dowel']['dowel_1']['hole_past_each_end_mm']} mm spare "
         f"each end)",
         f"<= {P.G9_CLAMPED_CLASH_MM3} mm3")

    # The left hand ships as a mirror of the right, so the native left build has to
    # be that mirror.  It was not, and only a public argument nothing shipped used
    # kept that from mattering.
    mirror = checks.mirror_consistency(cam)
    report["G9_mirror"] = mirror
    gate("G9g the native left-hand build is the mirror of the right-hand one",
         mirror["worst_difference_mm3"] <= P.G9_MIRROR_DIFF_MM3,
         ", ".join(f"{k} {v['native_not_in_mirror_mm3']}/"
                   f"{v['mirror_not_in_native_mm3']} mm3 either way"
                   for k, v in mirror["per_part"].items()),
         f"<= {P.G9_MIRROR_DIFF_MM3} mm3 in both directions")

    # "Supports: none" is a claim about the overhang that is *not* the bore arch.
    free = report["printability"]["worst_free_area_mm2"]
    gate("G9h overhang that is not the bore arch and has nothing under it",
         free <= P.G9_FREE_OVERHANG_MM2,
         ", ".join(f"{k} {v['arch_area_mm2']} arch + {v['other_area_mm2']} other, "
                   f"{v['free_area_mm2']} mm2 with nothing below"
                   for k, v in report["printability"].items()
                   if isinstance(v, dict) and v["unsupported_area_mm2"] > 0),
         f"<= {P.G9_FREE_OVERHANG_MM2} mm2 on any one file")

    module = checks.g7_module_fit(cam)
    report["G7_module_fit"] = module
    gate("G7 the bought camera module's envelope is empty of printed material",
         module["intersection_mm3"] <= P.G7_MODULE_FIT_MM3,
         f"{module['intersection_mm3']} mm3", f"<= {P.G7_MODULE_FIT_MM3} mm3")

    slots = checks.g7_plate_slots()
    report["G7_plate_slots"] = slots
    gate("G7 plate material between each M2 slot and the plate edge",
         slots["wall_mm"] >= P.MIN_WALL, slots["wall_mm"], f">= {P.MIN_WALL}")

    report["fasteners"] = checks.fastener_stack(cam)
    access = report["fasteners"]["driver_access"]
    worst = min(v["clear_travel_mm"] for v in access.values())
    gate("G7 both M4s can be reached with a key",
         worst >= P.DRIVER_REACH_MM,
         ", ".join(f"{k} {v['clear_travel_mm']} mm from {v['driven_from']}"
                   for k, v in access.items()),
         f">= {P.DRIVER_REACH_MM} mm of clear shaft")
    joints = report["fasteners"]["joints"]
    through = [j for j in joints.values() if j["kind"] == "steel nut"]
    tapped = [j for j in joints.values() if j["kind"] != "steel nut"]
    gate("G7 the called-out M4 suits both joints",
         all(0.0 <= j["protrusion_mm"] <= 4.0 for j in through)
         and all(j["thread_engagement_mm"] >= P.M4_THREAD_ENGAGE_MIN
                 for j in tapped),
         "; ".join(f"{k}: {j['kind']}, driven from {j['driven_from']}, "
                   + (f"{j['protrusion_mm']} mm proud of the nut"
                      if j["kind"] == "steel nut"
                      else f"{j['thread_engagement_mm']} mm of thread")
                   for k, j in joints.items())
         + f" (one M4 x {P.M4_LENGTH:.0f} suits both)",
         f"0..4 mm past a nut, >= {P.M4_THREAD_ENGAGE_MIN} mm of thread "
         f"where tapped")

    gauge = checks.gauge_fidelity()
    report["G7_gauges"] = gauge
    gate("G7 the rail-key gauge reproduces the key it is there to check",
         gauge["ok"], f"pocket {gauge['gauge_pocket_depth_mm']} mm deep x "
         f"{gauge['gauge_across_flats_mm']} mm across",
         f"{gauge['key_engagement_mm']} x {gauge['key_across_flats_mm']}")

    print("G6  stiffness")
    stiffness = checks.g6_stiffness(cam, mass)
    report["G6_stiffness"] = stiffness
    gate("G6 first lateral mode",
         stiffness["f1_with_joint_knockdown_Hz"] >= P.G6_FIRST_MODE_HZ,
         stiffness["reported"], f">= {P.G6_FIRST_MODE_HZ}")

    primitives = export.collision_primitives(cam)
    cover = checks.primitive_cover(meshes, primitives)
    report["collision_cover"] = cover
    gate("collision primitives cover the payload within 1.5 mm",
         cover["uncovered"] == 0, f"{cover['uncovered']} uncovered points", 0)
    gate("collision primitives add no more than 25 % volume",
         cover["volume_ratio"] <= 1.25, cover["volume_ratio"], "<= 1.25")
    # And in absolute terms, which is what a planner actually loses: revision 5's
    # cover claimed 90.8 cm3 of free space with 25 primitives, this one claims less
    # with 57 smaller ones.
    gate("collision primitives claim no more free space than revision 5's cover",
         cover["union_volume_cm3"] <= P.COVER_UNION_CM3,
         f"{cover['union_volume_cm3']} cm3 over {cover['n_primitives']} primitives "
         f"(rev 5: 90.8 cm3 over 25)", f"<= {P.COVER_UNION_CM3} cm3")
    prim_bottle = checks.primitive_bottle_clearance(primitives, samples=args.bottles)
    report["collision_primitive_bottles"] = prim_bottle
    gate("collision primitives pass G1 too",
         prim_bottle["min_clearance_mm"] >= P.G1_BOTTLE_CLEARANCE,
         prim_bottle["min_clearance_mm"], f">= {P.G1_BOTTLE_CLEARANCE}")
    # One code path for camera_spec.json, and this is not it: `export.write_all`
    # is.  verify.py hands down the occlusion and framing reports it has just
    # taken so nothing is cast twice, and `python -m mount.export` measures them
    # itself -- which is the fix for a camera_spec.json that was only complete if
    # verify.py happened to run last.  The block validation.json carries is the
    # spec that was written, not a second, emptier rendering of it.
    measured = export.camera_measurements(
        cam, meshes, occlusion=report["G3_occlusion"],
        forward=report["G3_forward_view"])
    report["camera_measurements"] = measured
    report["camera"] = export.write_camera_spec(
        cam, round(mass["mass_g"], 1), measured=measured)
    export.write_collision(cam, mass)
    export.export_glb(cam)
    # v4's radii, written down as well as measured: `git show` needs the history,
    # a copy of this directory does not have it, and the comparison is a property
    # of commit c80b163 rather than of this run.
    export.write_v4_reference(report["v4_reference"])

    print("G8  webcam variant")
    webcam = webcam_variant(cam)
    # The same search with no cradle at all, so "the bare body fits and the bracket
    # that holds it does not" is a measurement rather than a remembered sentence.
    webcam["bare_body_no_cradle"] = webcam_variant(cam, cradle=0.0)["best_placement"]
    report["G8_webcam"] = webcam
    # The gate asserts the condition, not the fact that a verdict string exists.  A
    # hard-coded True here counted toward "23/23 gates passed" and would have passed
    # a KEPT verdict at any swept radius.
    best = webcam["best_placement"]
    kept_ok = best is not None and (best["swept_radius_mm"] <= P.G8_SWEPT_RADIUS
                                    and best["bottle_clearance_mm"] >= P.G1_BOTTLE_CLEARANCE
                                    and best["gripper_clearance_mm"] >= P.G1_MESH_CLEARANCE)
    # The name said "G4" and the code tests G1's gripper-mesh clearance; the gate
    # is what the code does, so the name is now what the code does.
    gate("G8 webcam variant kept only if it passes G1 against a held bottle and "
         "against the gripper meshes, G3 framing and the swept radius",
         (best is None and webcam["verdict"].startswith("DROPPED")) or kept_ok,
         webcam["verdict"],
         f"DROPPED, or swept radius <= {P.G8_SWEPT_RADIUS} mm and clearances "
         f">= {P.G1_BOTTLE_CLEARANCE} mm")

    print("extras")
    top_down = checks.top_down_pick(meshes)
    report["top_down_pick"] = top_down
    gate("top-down pick: payload above the fingertip plane",
         top_down["margin_mm"] >= P.G7_TOPDOWN_MARGIN, top_down["margin_mm"],
         f">= {P.G7_TOPDOWN_MARGIN}")
    report["pour_roll_dip"] = checks.pour_roll_dip(meshes)

    report["gates"] = RESULTS
    report["elapsed_s"] = round(time.time() - started, 1)
    report["limitations"] = [
        "No physical prototype has been printed or fitted.",
        "Geometry comes from the vendor STLs; the real G1 may carry screws, "
        "cable glands or labels the meshes do not show. The two printed fit "
        "gauges exist to catch exactly that before the brackets are printed.",
        "Both finger meshes are open surfaces, so every check here is a "
        "triangle-surface test, never a solid-inside test.",
        "G6 is two hand-calculated Euler-Bernoulli springs in series with a "
        "blunt joint knockdown, not an FE model, and it does not contain the "
        "rubber liner, which is the softest thing in the load path. Two "
        "significant figures at most.",
        "The full-arm audit is 10,000 random samples with a fixed seed, not a "
        "workspace proof; unrestricted arm motion with any wrist payload still "
        "needs payload-aware collision checking -- and on this arm that means "
        "patching planner/motion.py as well as adding the collision geoms, "
        "because it treats an arm/-prefixed payload as self. See sim/DATASET.md.",
        f"The audit is gated on the {checks.GATE_HAND} hand, because v4's "
        f"{P.G5_V4_BOARD_COLLISIONS} is one hand's number -- v4 only ever had one "
        f"-- and a worse-of-two-hands figure against a single draw is biased "
        f"upward by construction. The other hand is measured and reported "
        f"(G5_arm_audit.reported_not_gated) and is above the reference: that is a "
        f"disclosed miss, not a renamed gate. G5_arm_audit_spread runs the same "
        f"audit at "
        f"{len(P.ARM_AUDIT_SPREAD_SEEDS)} seeds because a count of this size "
        f"carries about {P.G5_V4_BOARD_COLLISIONS ** 0.5:.0f} of sampling noise "
        f"on its own.",
        f"The camera envelope is a {P.PCB_SIZE:.0f} x {P.PCB_SIZE:.0f} mm board with "
        f"{-P.BODY_BACK:.1f} mm of rear components ({-P.BODY_BACK + P.PLATE_RELIEF_DEPTH:.1f} "
        f"inside an {P.PLATE_RELIEF:.0f} x {P.PLATE_RELIEF:.0f} mm central pocket), a "
        f"{P.HOLDER_SIZE:.0f} mm M12 holder block, a dia {P.LENS_DIA:.0f} x "
        f"{P.LENS_FRONT:.0f} mm barrel and a "
        f"{P.USB_KEEPOUT[0]:.0f} x {P.USB_KEEPOUT[1]:.0f} x {P.USB_KEEPOUT[2]:.0f} mm "
        f"connector keep-out off the board's lower edge, offset "
        f"{-P.USB_U_OFFSET:.0f} mm toward image-left. A different module changes "
        f"the numbers, and a connector in the middle of that edge does not fit "
        f"-- the strut is there.",
        "G4's clocking figure is the slop with the grub screw NOT fitted. With it "
        "in, the clocking is a plane on a plane and is set by print flatness, "
        "which nothing here measures.",
        f"G9f tests one assembled pose: both halves moved "
        f"{design.CLAMP_TRAVEL_MM} mm onto a liner of nominal thickness, "
        f"rigidly. It is not a tolerance stack -- a thicker liner, a printer "
        f"that runs the bore small or an ear that is not flat all change the "
        f"gap, and none of those is modelled. What it does catch is the class "
        f"that got through: a gate that only ever looks at the part as drawn.",
        "The camera-side joint is a thread cut in PETG, not a steel nut. It "
        "tolerates fewer assembly cycles than the far-side joint, and that is "
        "the price of putting a bolt ear under the strut's root. It is in the "
        "BOM and in assembly step 4.",
        "G9h's 'nothing below it' figure is a ray cast straight down from each "
        "overhanging face's centre against the part's own triangles. It says "
        "which faces bridge over air, not whether PETG will bridge them.",
    ]
    name = "validation.quick.json" if reduced else "validation.json"
    out = Path(args.out) if args.out else (scratch or ROOT) / name
    out.write_text(json.dumps(report, indent=2) + "\n")

    drift = []
    if scratch is not None:
        drift = compare_tree(ROOT / name, out)
        for path in export.WRITTEN_FILES:
            here, there = ROOT / path, scratch / path
            if not here.exists():
                drift.append(f"{path}: missing from the tree")
            elif not there.exists():
                drift.append(f"{path}: the run did not produce it")
            elif here.read_bytes() != there.read_bytes():
                drift.append(f"{path}: {here.stat().st_size} B tracked vs "
                             f"{there.stat().st_size} B regenerated")
        for line in drift:
            print(f"  DRIFT {line}")
        print(f"  {len(export.WRITTEN_FILES) + 1 - len(drift)}/"
              f"{len(export.WRITTEN_FILES) + 1} tracked files match this run")
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"  -> {out.name}")

    failed = [r["gate"] for r in RESULTS if not r["passed"]]
    print()
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} gates passed "
          f"in {report['elapsed_s']} s"
          + ("  [REDUCED RUN -- not the record of a full one]" if reduced else ""))
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed or drift else 0


# Two numbers in the report are the run and not the design: how long it took and
# what time it is.  A --check run must not call those drift.
VOLATILE = ("elapsed_s",)


def compare_tree(tracked, fresh):
    """Diff two validation.json files, ignoring the timing fields."""
    if not tracked.exists():
        return [f"{tracked.name}: missing from the tree"]
    a = json.loads(tracked.read_text())
    b = json.loads(fresh.read_text())
    for key in VOLATILE:
        a.pop(key, None)
        b.pop(key, None)
    if a == b:
        return []
    keys = sorted(set(a) | set(b))
    differing = [k for k in keys if a.get(k) != b.get(k)]
    return [f"{tracked.name}: {len(differing)} block(s) differ: "
            + ", ".join(differing[:8])]


def webcam_variant(cam, cradle=5.0):
    """G8: can a 1/4-20 webcam ride on this mount line at all?

    Search the 45 x 74 x 45 mm body over the region the clamp band can reach,
    carrying the strut it would need, and score it on the same gates the board
    camera has to meet: G1 against a held bottle and against the real gripper
    meshes, G3's framing, and the swept radius.  The answer decides whether the
    variant is built; it is a measurement, not a preference.
    """
    import math
    import numpy as np

    limit = P.G8_SWEPT_RADIUS
    CRADLE_T = cradle       # wall of the cradle that has to hold the body
    rng = np.random.default_rng(P.BOTTLE_SEED)
    # The 32 corners of the bottle parameter box as well as 40 random draws: the
    # worst case here is an all-maximum bottle, which random sampling never hits,
    # and adding them makes the drop conclusion stronger rather than weaker.
    bottles = [checks.sample_bottle(rng) for _ in range(40)] + checks.corner_bottles()
    gripper = checks.rb.gripper_meshes(50.0)
    gripper_points = np.concatenate([m.vertices for m in gripper.values()])
    best, nearest_miss = None, None

    def surface(centre, half):
        """Dense samples of an axis-aligned box's surface."""
        grids = [np.linspace(-h, h, max(2, int(2 * h / 4.0) + 1)) for h in half]
        out = []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            u, v = np.meshgrid(grids[others[0]], grids[others[1]], indexing="ij")
            for sign in (-half[axis], half[axis]):
                local = np.zeros((u.size, 3))
                local[:, others[0]] = u.ravel()
                local[:, others[1]] = v.ravel()
                local[:, axis] = sign
                out.append(centre + local)
        return np.concatenate(out)

    for x in np.arange(-34.0, -13.0, 3.0):
        for y in np.arange(50.0, 112.0, 3.0):
            for z in np.arange(24.0, 92.0, 4.0):
                for half in ((22.5, 37.0, 22.5), (22.5, 22.5, 37.0),
                             (37.0, 22.5, 22.5)):
                    centre = np.array([x, y, z])
                    # A webcam cannot be held by nothing: the envelope that
                    # actually sweeps is the body plus a 5 mm cradle and the
                    # 1/4-20 head under it.
                    grown = np.array(half) + CRADLE_T
                    points = surface(centre, grown)
                    # the strut that would have to reach it, from the band OD
                    azimuth = math.atan2(z, y)
                    root = np.array([-38.0, 35.0 * math.cos(azimuth),
                                     35.0 * math.sin(azimuth)])
                    steps = np.linspace(0.0, 1.0, 12)[:, None]
                    strut = root + steps * (centre - root)
                    payload = np.vstack([points, strut])
                    radius = float(np.hypot(payload[:, 1], payload[:, 2]).max())
                    if radius > limit or (best and radius >= best["swept_radius_mm"]):
                        continue
                    # G3: a lens on the inboard face, aimed like the board camera
                    lens = centre - np.array([0.0, half[1], 0.0])
                    probe = design.CameraFrame(pos=(lens[0], lens[1], lens[2]),
                                               yaw=P.CAM_YAW_DEG,
                                               pitch=P.CAM_PITCH_DEG)
                    view = checks.g3_forward_view(probe, P.FOV_RECOMMENDED)
                    if not (view["forward_point_in_frame"]
                            and view["fingertips_in_frame"]):
                        continue
                    # clearance to the real gripper: point-to-box distance
                    delta = np.abs(gripper_points - centre) - grown
                    gap = float(np.linalg.norm(np.maximum(delta, 0.0),
                                               axis=1).min())
                    if gap < P.G1_MESH_CLEARANCE:
                        continue
                    worst = min(
                        checks.bottle_clearance(payload, b, z_rel(b), depth, pitch)
                        for label, z_rel, depth, pitch in checks.GRASP_CANDIDATES
                        for b in bottles
                        if not (label == "neck"
                                and not checks.neck_graspable(b, pitch)))
                    if worst < P.G1_BOTTLE_CLEARANCE:
                        if nearest_miss is None or worst > nearest_miss["bottle_clearance_mm"]:
                            nearest_miss = dict(
                                centre_mm=[float(v) for v in centre],
                                body_half_extents_mm=list(half),
                                cradle_wall_mm=CRADLE_T,
                                swept_radius_mm=round(radius, 1),
                                gripper_clearance_mm=round(gap, 2),
                                bottle_clearance_mm=round(worst, 2))
                        continue
                    best = dict(centre_mm=[float(v) for v in centre],
                                body_half_extents_mm=list(half),
                                cradle_wall_mm=CRADLE_T,
                                swept_radius_mm=round(radius, 1),
                                bottle_clearance_mm=round(worst, 2),
                                gripper_clearance_mm=round(gap, 2))
    if best is None:
        verdict = ("DROPPED: no placement inside the 125 mm swept-radius "
                   "allowance clears a held bottle by 4 mm while keeping the "
                   "fingertips and the scene ahead in frame")
    else:
        verdict = (f"KEPT: swept radius {best['swept_radius_mm']} mm, bottle "
                   f"clearance {best['bottle_clearance_mm']} mm")
    return dict(verdict=verdict, best_placement=best,
                best_rejected_placement=nearest_miss,
                cradle_wall_mm=CRADLE_T,
                swept_radius_limit_mm=limit, body_envelope_mm=[45, 74, 45],
                method="lattice over the region this clamp band can reach, "
                       "three body orientations, each carrying the strut it "
                       f"would need; scored on G1 ({len(bottles)} bottles = 40 "
                       f"random draws plus the 32 parameter-box corners, x every "
                       f"planner grasp candidate, and the real gripper meshes at "
                       f"jaw 100), G3 framing with the wide lens, and swept radius",
                bottles=len(bottles))


if __name__ == "__main__":
    raise SystemExit(main())
