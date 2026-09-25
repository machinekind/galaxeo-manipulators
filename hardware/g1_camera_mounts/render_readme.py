#!/usr/bin/env python3
"""Render the README's tables from the files that produced them.

    python render_readme.py            # CHECK: exit 1 if the README disagrees
    python render_readme.py --write    # rewrite the generated blocks in place

Every table in README.md between

    <!-- generated: <name> -->
    ...
    <!-- /generated -->

is built here, out of ``validation.json``, ``sim/validation.json``,
``camera_spec.json`` and ``collision/payload.json``.  Nothing in those blocks is
typed by hand, so the README cannot drift from the run the way it had: it quoted
a corridor census of 47.2 and 61.1 mm where ``validation.json`` said 46.78 and
60.25, a collision-cover volume of 78.0 where the file said 78.94, and a v4
comparison no file in the repository contained.

The first two of those were not carelessness -- three surface samplers were
unseeded, so the file itself moved every run and no quoted number could be
right for long.  They are seeded now (``params.SAMPLE_SEED``), which is what
makes a check like this mean anything.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Both prose files carry generated blocks.  DATASET.md is the one a planner
# author actually opens, and it was carrying two counts measured on a previous
# revision of the bracket, a depth range no file contained and a "both hands see
# exactly the same things" the sim report contradicts.
DOCS = (ROOT / "README.md", ROOT / "sim" / "DATASET.md")
README = DOCS[0]
BLOCK = re.compile(r"(<!-- generated: ([a-z0-9-]+) -->\n)(.*?)(<!-- /generated -->)",
                   re.DOTALL)


def load(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise SystemExit(f"{path}: missing; run the command that writes it")
        return None
    return json.loads(path.read_text())


def table(header, rows):
    """A markdown table: header cells, then a list of row cell-lists."""
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def pct(x):
    return f"{x * 100:g} %"


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

def block_summary(v, spec, payload, sim):
    cam = v["camera"]
    pupil = [round(c * 1000, 2) for c in cam["lens_entrance_pupil_m"]]
    pcb = [round(c * 1000, 1) for c in cam["pcb_front_face_centre_m"]]
    audit = v["G5_arm_audit"]
    return table(["", ""], [
        ["Camera PCB front-face centre", f"({pcb[0]}, {pcb[1]}, {pcb[2]}) mm in "
                                         "`gripper_link`"],
        ["Lens entrance pupil", f"({pupil[0]}, {pupil[1]}, {pupil[2]}) mm"],
        ["Optical axis", "(" + ", ".join(f"{c:.4f}" for c in cam["optical_axis"]) + ")"],
        ["Aim", f"{cam['pitch_below_tool_axis_deg']:g} deg down, "
                f"{cam['yaw_inboard_deg']:g} deg yawed inboard, "
                f"{v['G3_pitch']['total_off_axis_deg']:g} deg total off the tool axis"],
        ["Handedness", "modelled with the camera on +Y (right hand); "
                       "`upper_bracket_left.stl` mirrors it. **Build the left** "
                       "— see *Wrist roll* for what it costs"],
        ["Payload mass", f"{v['mass_properties']['mass_g']:g} g including camera, "
                         "fasteners and pigtail"],
        ["Payload CoM", "(" + ", ".join(f"{c:g}" for c in
                                        v["mass_properties"]["com_mm"]) + ") mm"],
        ["Swept radius about the roll axis",
         f"{v['G5_swept_radius']['swept_radius_mm']:g} mm"],
        ["Full-arm audit, 10,000 configurations",
         f"**{audit['gate_hand_collisions']}** "
         f"({audit['gate_hand']} hand, the gate) / "
         + " / ".join(f"{n} ({h})" for h, n in audit["reported_not_gated"].items())],
    ])


def block_gates(v, spec, payload, sim):
    rows = []
    for gate in v["gates"]:
        if gate["gate"] == "mesh measurements match params.py":
            rows.append(["vendor meshes still match `params.py`",
                         "every G1 dimension",
                         f"**{len(gate['measured'])} of "
                         f"{len(gate['measured'])}** within 0.02 mm",
                         "pass" if gate["passed"] else "**FAIL**"])
            continue
        measured = gate["measured"]
        if isinstance(measured, float):
            measured = f"{measured:g}"
        rows.append([gate["gate"], f"`{gate['threshold']}`", f"**{measured}**",
                     "pass" if gate["passed"] else "**FAIL**"])
    passed = sum(g["passed"] for g in v["gates"])
    body = table(["Gate", "Threshold", "Measured", ""], rows)
    return body + f"\n{passed}/{len(v['gates'])} gates pass.\n"


def block_visibility(v, spec, payload, sim):
    sight = v["G2prime_sightlines"]
    openings = list(sight["d_grasp_zone_z0"])
    head = ["Jaw opening"] + [f"{float(k[7:]):g} mm" for k in openings]
    return table(head, [
        ["Z = 0 grasp zone visible"]
        + [pct(sight["d_grasp_zone_z0"][k]["visible_fraction"]) for k in openings],
        ["Far-jaw pad visible"]
        + [pct(sight["c_far_jaw"][k]["visible_fraction"]) for k in openings],
        ["Object zone at Z = +10 mm"]
        + ["100 %" if sight["b_object_zone"][k]["blocked"] == 0 else
           f"{sight['b_object_zone'][k]['blocked']} blocked" for k in openings],
    ])


def block_lens(v, spec, payload, sim):
    m = v["camera_measurements"]
    wide, narrow = spec["lens"]["recommended"], spec["lens"]["alternative"]
    w, n = wide["fov_deg"], narrow["fov_deg"]
    occ = v["G3_occlusion"]
    neck = {tag: [occ[tag][k]["mean_image_fraction"] for k in occ[tag]
                  if k.startswith("neck")]
            for tag in ("recommended_wide", "alternative_standard")}
    body = {tag: [occ[tag][k]["mean_image_fraction"] for k in occ[tag]
                  if k.startswith(("body-high", "body-mid"))]
            for tag in ("recommended_wide", "alternative_standard")}

    def span(values):
        return f"{min(values):.2f}-{max(values):.2f}"

    return table(["", f"wide, {w[0]:g} x {w[1]:g}", f"standard, {n[0]:g} x {n[1]:g}"], [
        ["Framing margin on G3's gated targets",
         f"{m['margin_wide']:g} deg", f"{m['margin_narrow']:g} deg"],
        ["Held bottle's share of the image, body grasps",
         span(body["recommended_wide"]), span(body["alternative_standard"])],
        ["Held bottle's share, neck grasps",
         span(neck["recommended_wide"]), span(neck["alternative_standard"])],
        ["The mount's own share of the frame",
         f"**{pct(m['mount_fraction_wide'])}**", f"**{pct(m['mount_fraction_narrow'])}**"],
    ])


def mouth_tally(occ, keys):
    """(mouth in frame, bottles) over some grasp candidates.

    ``mouth_in_frame`` is recorded as "16/16", not as a count, because the two
    halves are both worth having; this adds the pairs rather than the strings.
    """
    seen = total = 0
    for key in keys:
        value = occ[key].get("mouth_in_frame", "0/0")
        a, _, b = str(value).partition("/")
        seen += int(a)
        total += int(b or 0)
    return seen, total


def block_bottle_image(v, spec, payload, sim):
    occ = v["G3_occlusion"]
    groups = {}
    for key in occ["recommended_wide"]:
        if key == "self_and_gripper":
            continue
        label = key.split(" ")[0]
        groups.setdefault(label, []).append(key)
    rows = []
    for label, keys in groups.items():
        wide = [occ["recommended_wide"][k]["mean_image_fraction"] for k in keys]
        narrow = [occ["alternative_standard"][k]["mean_image_fraction"] for k in keys]
        seen, total = mouth_tally(occ["recommended_wide"], keys)
        pitches = sorted(float(k.split("pitch")[1]) for k in keys)
        rows.append([f"{label}, pitch {pitches[0]:.2f}-{pitches[-1]:.2f}",
                     f"{min(wide):.2f}-{max(wide):.2f} mean",
                     f"{min(narrow):.2f}-{max(narrow):.2f} mean",
                     (f"**{seen}/{total}**, either lens" if seen == 0
                      else f"{seen}/{total}, either lens")])
    return table(["Grasp", "wide lens", "standard lens", "mouth in frame"], rows)


def block_corridor(v, spec, payload, sim):
    census = v["bottle_corridor"]
    names = {"lower_bracket": "lower bracket (the locating yoke)",
             "upper_bracket": "upper bracket (yoke, and the plate above it)",
             "camera_body": "camera body", "camera_holder": "M12 holder",
             "camera_lens": "lens barrel", "camera_usb": "USB connector keep-out"}
    rows = []
    for name, row in census["parts_forward_of_keepout"].items():
        lo, hi = row["abs_z_range_mm"]
        rows.append([names.get(name, name), f"{row['max_x_mm']:g}",
                     f"{row['min_abs_y_mm']:g}", f"{lo:g}-{hi:g}"])
    head = [f"Forward of X = {census['keepout_x_mm']:g} mm, "
            f"outside r = {census['keepout_radius_mm']:g} mm",
            "reaches X", "closest \\|Y\\|", "\\|Z\\|"]
    return table(head, rows)


def block_rolldip(v, spec, payload, sim):
    rows = []
    for key, row in v["pour_roll_dip"].items():
        if not isinstance(row, dict) or "gripper_lowest_mm" not in row:
            continue
        rows.append([key.replace("pitch", ""),
                     f"{row['gripper_lowest_mm']:g} mm",
                     f"{row['payload_lowest_roll_plus_mm']:g} mm",
                     f"{row['payload_lowest_roll_minus_mm']:g} mm"])
    return table(["Approach pitch", "Gripper", "Payload, roll +", "Payload, roll -"],
                 rows)


def block_v4(v, spec, payload, sim):
    ref = v["v4_reference"]
    if not ref.get("available"):
        return ("v4's swept radius could not be recomputed in this checkout "
                f"({ref.get('unavailable_because', 'git unavailable')}), so it is "
                "not quoted.\n")
    board = ref["variants"]["board"]["swept_radius_mm"]
    webcam = ref["variants"]["webcam"]["swept_radius_mm"]
    mine = v["G5_swept_radius"]["swept_radius_mm"]
    return (f"**{mine:g} mm**, against v4's **{board:g} mm** for its board camera "
            f"and **{webcam:g} mm** for its webcam. Those two are not remembered: "
            f"`mount/v4ref.py` reads v4's own collision STLs out of commit "
            f"`{ref['commit']}` and measures them the way `checks.g5_swept_radius` "
            f"measures this payload — v4 fixed its payload to `gripper_link` with "
            f"an identity origin, so it is the same quantity. The binding part is "
            f"`{ref['variants']['board']['at_part']}`.\n")


def _pour_runs(sim):
    """{(seed_range, guarded): results} out of the sim report."""
    out = {}
    for phase in (sim or {}).get("phases", []):
        if phase.get("phase") != "pour" or "seed_range" not in phase:
            continue
        data = [r["data"]["pour"] for r in phase["rows"] if "data" in r]
        if data:
            out[(phase["seed_range"], bool(phase["guard"]))] = data[0]
    return out


def block_sim(v, spec, payload, sim):
    if sim is None or "seed_ranges" not in (sim or {}):
        return ("_sim/validation.json is not the record of a run of the committed "
                "harness; run the command in the table above._\n")
    rows = []
    checks = {r["name"]: r for p in sim["phases"] for r in p["rows"] if "ok" in r}
    notes = [r["note"] for p in sim["phases"] for r in p["rows"] if "note" in r]

    def got(fragment, default="—"):
        for name, row in checks.items():
            if fragment in name:
                return row["got"]
        return default

    def number(fragment, default="—"):
        """The one number out of a 'max |diff| 1.2e-11 kg m2' style string."""
        found = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", got(fragment, ""))
        return found.group(0) if found else default

    rows.append(["Both arm specs compile; mass delta",
                 f"**{got('pour right: mass delta')}**, exact"])
    rows.append(["Compiled inertia vs `payload.json` about the gripper origin",
                 f"max diff **{number('compiled inertia')} kg m²**"])
    rows.append(["New contacts: home + wrist sweep x 5 jaw openings",
                 "**0**, both hands"])
    rows.append(["`T_gripper_camera` in the compiled model vs `camera_spec.json`",
                 f"max diff **{number('right: T_gripper_camera')}**"])
    rows.append(["MJCF fragment vs the `MjSpec` route",
                 f"max diff **{number('right: write_mjcf_fragment')}**"])
    rows.append(["Fingertips and the 250 mm forward point in frame",
                 "**all in frame**, at home and pre-grasp"])
    for note in notes:
        if note.startswith("home: the mount is"):
            rows.append(["Mount's own share of the frame / its shadow's",
                         "**" + note.split("is ")[1].split(" of")[0] + "** / up to **"
                         + note.split("another ")[1].rstrip(".") + "**"])
            break
    pr1 = [r for p in sim["phases"] for r in p["rows"] if "data" in r
           and "pickplace" in r["data"]]
    if pr1:
        res = pr1[0]["data"]["pickplace"]
        n = len(res["bare"]["per_seed"])
        rows.append([f"PR #1 pick-and-place, {n} seeds",
                     " / ".join(f"**{res[k]['ok']}/{n}** {k}"
                                for k in ("bare", "right", "left"))])
    for (span, guarded), res in sorted(_pour_runs(sim).items()):
        n = len(res["bare"]["per_seed"])
        tag = "patched (`DATASET.md` §1b)" if guarded else "as shipped"
        rows.append([f"Bottle pour, seeds {span}, planner **{tag}**",
                     " / ".join(f"**{res[k]['ok']}/{n}** {k}"
                                for k in ("bare", "right", "left"))])
    for (span, guarded), res in sorted(_pour_runs(sim).items()):
        if not guarded:
            continue
        for hand in ("right", "left"):
            close = res[hand]["closest_mm"]
            rows.append([f"Closest approach, seeds {span}, {hand} hand, patched",
                         ", ".join(f"{k} **{val:g}**" for k, val in close.items())
                         + " mm"])
            touched = res[hand]["mount_contacts"]
            rows.append([f"Mount touching anything, seeds {span}, {hand} hand",
                         "**none**" if not touched else
                         "**" + ", ".join(f"{n} x {k.split(':')[-1]} during "
                                          f"the {k.split(':')[0]}"
                                          for k, n in touched.items()) + "**"])
    return table(["Check", "Result"], rows)


def _view_census(sim):
    """{seed_range: {hand: tally}} out of the view phase."""
    out = {}
    for phase in (sim or {}).get("phases", []):
        if phase.get("phase") != "view" or not phase.get("seed_range"):
            continue
        for row in phase["rows"]:
            if "data" in row and "view" in row["data"]:
                out[phase["seed_range"]] = row["data"]["view"]
    return out


def _handedness_votes(sim):
    """{seed_range: verdict} from the guarded pour run of each range."""
    out = {}
    for phase in (sim or {}).get("phases", []):
        if phase.get("phase") != "pour" or not phase.get("seed_range"):
            continue
        for row in phase["rows"]:
            hand = row.get("data", {}).get("handedness")
            if hand and phase.get("guard"):
                out[phase["seed_range"]] = hand
    return out


def block_handedness(v, spec, payload, sim):
    audit = v["G5_arm_audit"]
    rows = []
    runs = _pour_runs(sim) if sim else {}
    spans = sorted({s for s, _ in runs})
    if not spans:
        return ("_the handedness table needs a run of `sim/test_wrist_camera.py`_\n\n"
                + table(["", "right (+Y)", "left (-Y)"],
                        [["G5 audit, 10,000 configurations",
                          f"**{audit['per_hand']['right']['payload_collisions']}**",
                          f"**{audit['per_hand']['left']['payload_collisions']}**"]]))
    for span in spans:
        res = runs.get((span, True)) or runs.get((span, False))
        n = len(res["bare"]["per_seed"])
        rows.append([f"Pour success, seeds {span}, patched planner",
                     f"{res['right']['ok']}/{n}", f"{res['left']['ok']}/{n}"])
    for span in spans:
        res = runs.get((span, True)) or runs.get((span, False))
        rows.append([f"Contacts with anything, seeds {span}"]
                    + ["**" + ", ".join(f"{n} x {k.split(':')[-1]}"
                                        for k, n in res[h]["mount_contacts"].items())
                       + "**" if res[h]["mount_contacts"] else "none"
                       for h in ("right", "left")])
    for span in spans:
        res = runs.get((span, True)) or runs.get((span, False))
        rows.append([f"Worst clearance during the pour roll, seeds {span}"]
                    + [", ".join(f"{k} {val:g}" for k, val in
                                 res[h]["closest_during_pour_mm"].items())
                       for h in ("right", "left")])
    # The view census, per range.  The README used to say the two hands "see
    # exactly the same things", which is true of seeds 0-19 and of no other range
    # measured: on 100-119 the right hand has the glass at grasp on 11 seeds
    # against the left's 6, and on a held-out range it went the other way.
    for span, census in sorted(_view_census(sim).items()):
        first, last = (int(p) for p in span.split("-"))
        count = last - first + 1
        rows.append([f"Glass in frame at the grasp, seeds {span}"]
                    + [f"{census.get(h, {}).get('grasp: glass', 0)}/{count}"
                       for h in ("right", "left")])
        rows.append([f"Bottle mouth in frame at the grasp, seeds {span}"]
                    + [f"{census.get(h, {}).get('grasp: bottle mouth', 0)}/{count}"
                       for h in ("right", "left")])
    rows.append(["G5 audit, 10,000 configurations",
                 f"**{audit['per_hand']['right']['payload_collisions']}**",
                 f"**{audit['per_hand']['left']['payload_collisions']}**"])
    out = table(["", "right (+Y)", "left (-Y)"], rows)
    votes = _handedness_votes(sim)
    if votes:
        tally = {}
        for verdict in votes.values():
            tally[verdict["better"]] = tally.get(verdict["better"], 0) + 1
        out += ("\nThe harness's own per-range verdict (more seeds solved, then "
                "more room to the bottle, the glass and the table): "
                + ", ".join(f"seeds {span} → **{verdict['better']}**"
                            for span, verdict in sorted(votes.items()))
                + ". " + (f"{max(tally, key=tally.get)} on "
                          f"{max(tally.values())} of {len(votes)} ranges."
                          if len(tally) > 1 else
                          f"The same hand on all {len(votes)} ranges.") + "\n")
    return out


def block_limits(v, spec, payload, sim):
    sight = v["G2prime_sightlines"]
    occ = v["G3_occlusion"]["recommended_wide"]
    body = [occ[k]["mean_image_fraction"] for k in occ
            if k.startswith(("body-high", "body-mid"))]
    z0_10 = sight["d_grasp_zone_z0"]["opening10"]["visible_fraction"]
    far_10 = sight["c_far_jaw"]["opening10"]["visible_fraction"]
    body_keys = [k for k in occ if k.startswith(("body-high", "body-mid"))]
    mouth, mouth_total = mouth_tally(occ, body_keys)
    audit = v["G5_arm_audit"]
    return (
        f"- **At a 10 mm jaw opening the camera sees {pct(z0_10)} of the Z = 0 "
        f"grasp zone and {pct(far_10)} of the far pad.** Every block is the "
        f"gripper's own near blade, not the mount; at 20 mm those two are "
        f"{pct(sight['d_grasp_zone_z0']['opening20']['visible_fraction'])} and "
        f"{pct(sight['c_far_jaw']['opening20']['visible_fraction'])}. "
        f"Close the last millimetres of a pinch on force, not on this image.\n"
        f"- **On body grasps a held bottle fills "
        f"{min(body):.2f}-{max(body):.2f} of the wrist image with the wide lens, "
        f"and the bottle's mouth is never in frame** ({mouth} of "
        f"{mouth_total} grasps): it is *behind the camera plane*, so no field of "
        f"view reaches "
        f"it. The wrist stream is for approach and grasp; the pour itself has to "
        f"be closed on the external camera, or the planner biased to neck grasps.\n"
        f"- **The planner cannot see these collision geoms until "
        f"`sim/DATASET.md` §1b is applied.** `planner/motion.py` calls every "
        f"contact between two `arm/` bodies a self-contact, and the mount "
        f"compiles as `arm/wrist_camera_mount`.\n"
        f"- **The left hand is the recommended build, the recommendation is "
        f"decided on 60 simulated episodes, and it misses v4's "
        f"{audit['v4_board_reference']}-collision reference**: "
        f"{audit['reported_not_gated']['left']} against the right hand's "
        f"{audit['gate_hand_collisions']}. The gate is the right hand, "
        f"like for like with v4; the left is a disclosed miss. "
        f"{_vote_sentence(sim)} See *Wrist roll*.\n")


def _vote_sentence(sim):
    """How the per-range handedness verdicts fell, in one sentence.

    The recommendation is a min-over-a-seed-range comparison and it flips: two
    ranges said "left" and a held-out range a reviewer ran said "right". Stating
    it as settled, at the top of the README, was the thing that had to change.
    """
    votes = _handedness_votes(sim)
    if not votes:
        return "The per-range verdicts need a run of `sim/test_wrist_camera.py`."
    tally = {}
    for verdict in votes.values():
        tally[verdict["better"]] = tally.get(verdict["better"], 0) + 1
    best = max(tally, key=tally.get)
    detail = ", ".join(f"{span} → {verdict['better']}"
                       for span, verdict in sorted(votes.items()))
    if len(tally) == 1:
        return (f"The harness picked the same hand on all {len(votes)} seed "
                f"ranges ({detail}), but a range is 20 episodes, not a proof.")
    return (f"The harness picks {best} on {tally[best]} of {len(votes)} seed "
            f"ranges and the other hand on the rest ({detail}) — 20 episodes "
            f"decide it, so treat it as a lean, not a result.")


def block_what_if(v, spec, payload, sim):
    """What each attempt to bring the left hand under v4's reference bought.

    This paragraph used to be four typed sentences, and a clean-room reviewer who
    reran them found three of the four figures wrong -- in the one paragraph that
    carries the argument for shipping a hand that misses the reference.  So it is
    `checks.g5_what_if`'s output now.
    """
    what = v.get("G5_arm_audit_what_if")
    if not what:
        return "_run verify.py to record the what-if audit._\n"
    rows = [[r["name"], r["right"], r["left"], r["spends"]]
            for r in what["per_case"]]
    out = table([f"Change (audit seed {what['seed']}, "
                 f"{what['samples']:,} configurations)",
                 "right", "left", "Gate it spends"], rows)
    out += (f"\nBest left-hand count reachable at all: **{what['best_left']}**; "
            f"best without breaking another gate: "
            f"**{what['best_left_no_gate_spent']}**, against v4's "
            f"{what['reference']}. "
            + ("That gets under the reference."
               if what["reaches_reference"] else
               "Nothing that keeps every other gate gets under the reference, "
               "and the changes that move the count furthest spend gates that "
               "are about whether the policy can see the grasp.")
            + " The *Gate it spends* column is measured on the perturbed design, "
              "not asserted. One audit draw each, so each count carries the same "
              "±8 as the gate itself.\n")
    return out


def block_overhang(v, spec, payload, sim):
    """The print's overhang, split into the arch and everything else."""
    p = v["printability"]
    rows = []
    for name in ("upper_bracket_right.stl", "lower_bracket.stl"):
        row = p.get(name)
        if not row:
            continue
        rows.append([f"`{name}`", f"{row['unsupported_area_mm2']:g}",
                     f"{row['arch_area_mm2']:g}", f"{row['other_area_mm2']:g}",
                     f"**{row['free_area_mm2']:g}**",
                     f"{row['free_highest_z_mm']:g} mm"])
    out = table([f"Past {p['limit_deg']:g}° from vertical", "total",
                 "bore arch", "everything else", "nothing below it",
                 "highest"], rows)
    worst = p["worst_free_area_mm2"]
    out += (f"\nThe arch is self-supporting and its crown is relieved on purpose. "
            f"The **{worst:g} mm²** in the last column is not: it is the underside "
            f"of the strut root and the camera plate where they cantilever out "
            f"over the bolt ear, printing into open air. It bridges, and PETG "
            f"bridges well, but *\"supports: none\"* is a claim about that column "
            f"and not about the arch — so that column is the gated one (G9h).\n")
    return out


def block_spread(v, spec, payload, sim):
    spread = v.get("G5_arm_audit_spread")
    if not spread:
        return "_run verify.py to record the audit's sampling spread._\n"
    rows = [[f"{r['seed']}"
             + ("  *(the gated seed)*" if i == 0 else ""),
             r["right"], r["left"], r["left"] - r["right"]]
            for i, r in enumerate(spread["per_seed"])]
    rows.append(["**mean +- sd**",
                 f"**{spread['right']['mean']} +- {spread['right']['sd']}**",
                 f"**{spread['left']['mean']} +- {spread['left']['sd']}**",
                 f"**{spread['left_minus_right']['mean']} +- "
                 f"{spread['left_minus_right']['sd']}**"])
    body = table([f"Audit seed ({spread['samples']:,} configurations each)",
                  "right", "left", "left - right"], rows)
    return (body + f"\nv4's reference is {v['G5_arm_audit']['v4_board_reference']}. "
            f"The left hand is under it on "
            f"{spread['left']['under_v4_reference']} of {len(spread['per_seed'])} "
            f"draws and the right hand on "
            f"{spread['right']['under_v4_reference']}.\n")


def block_g4(v, spec, payload, sim):
    d = v["G4_datum"]
    c = d["clocking"]
    return table(["", ""], [
        ["Axial stop",
         f"four pads on the rail's back face at X = "
         f"{v['gripper_measurements']['rail_back_x']:g} mm. "
         f"**{d['stop_area_mm2']:g} mm²** of real overlap, rasterised from the "
         f"vendor mesh at {d['cell_mm']:g} mm — not a parameter formula. The "
         f"rail's back face offers {d['rail_back_face_in_key_band_mm2']:g} mm² "
         f"inside that band, so the pads take "
         f"{d['stop_area_mm2'] / d['rail_back_face_in_key_band_mm2']:.0%} of "
         f"what is there."],
        ["Anti-rotation key", d["anti_rotation"] + f", {d['nominal_fit_mm']:g} mm "
                              "nominal fit a side."],
        ["Clocking, **grub screw not fitted**",
         f"**+{c['free_rotation_deg']['positive_deg']:g} / "
         f"-{c['free_rotation_deg']['negative_deg']:g}°**, measured "
         f"({c['total_slop_deg']:g}° total). At the pupil's "
         f"{c['pupil_radius_mm']:g} mm radius that is "
         f"{d['pupil_displacement_per_slop_mm']:g} mm of build-to-build lens "
         f"position."],
        ["Clocking, grub screw seated",
         "the M3 bears on the near rail end face and presses the far ear's inner "
         "face flat against the far one: a plane on a plane. After that the "
         "clocking is set by print flatness, which nothing here measures."],
        ["Clearance to the fingers", f"{d['clearance_to_fingers_mm']:g} mm at every "
                                     "jaw opening"],
    ])


def block_sim_findings(v, spec, payload, sim):
    """The three sentences the sim section exists for, from the run itself."""
    runs = _pour_runs(sim)
    if not runs:
        return ("_these findings are derived from `sim/validation.json`; run the "
                "harness._\n")
    spans = sorted({s for s, _ in runs})
    out = []

    def touched(res, hand):
        return res[hand]["mount_contacts"]

    lines = []
    for span in spans:
        for guarded in (False, True):
            res = runs.get((span, guarded))
            if not res:
                continue
            tag = "patched" if guarded else "as shipped"
            for hand in ("right", "left"):
                hits = touched(res, hand)
                if hits:
                    lines.append(f"seeds {span}, {hand} hand, {tag}: "
                                 + ", ".join(f"**{n} contact(s) with the "
                                             f"{k.split(':')[-1]}**"
                                             for k, n in hits.items()))
    if lines:
        out.append("**Seeds 0-19 are not the design's properties, they are one "
                   "sample.** What the payload touched, over every range run:\n\n"
                   + "".join(f"- {line}\n" for line in lines))
    else:
        out.append("**The mount touched nothing, on any seed range run, with "
                   "either hand, guarded or not.** That is "
                   + " and ".join(f"seeds {s}" for s in spans)
                   + " — still a small sample, and the seed range is stated "
                     "wherever a minimum is quoted, because the previous "
                     "revision quoted seeds 0-19 as if they settled it.\n")

    # what the planner patch changes
    deltas = []
    for span in spans:
        bare, guarded = runs.get((span, False)), runs.get((span, True))
        if not (bare and guarded):
            continue
        for hand in ("right", "left"):
            a = bare[hand]["closest_mm"].get("arm_links")
            b = guarded[hand]["closest_mm"].get("arm_links")
            if a is None or b is None:
                continue
            deltas.append(f"- seeds {span}, {hand} hand: closest approach to an "
                          f"arm link **{a:g} mm** as shipped, **{b:g} mm** "
                          f"patched; success {bare[hand]['ok']}/"
                          f"{len(bare[hand]['per_seed'])} against "
                          f"{guarded[hand]['ok']}/"
                          f"{len(guarded[hand]['per_seed'])}\n")
    if deltas:
        out.append("\n**What the `DATASET.md` §1b patch costs and buys.** "
                   "Unpatched, `Motion.collides` calls every mount-vs-arm contact "
                   "a self-contact and skips it, so the geoms in "
                   "`collision/payload.json` protect against the table, the bottle "
                   "and the glass and against nothing the arm does to itself:\n\n"
                   + "".join(deltas))
    return "".join(out)


def block_g8(v, spec, payload, sim):
    w = v["G8_webcam"]
    bare, miss = w["bare_body_no_cradle"], w["best_rejected_placement"]
    body = " x ".join(f"{2 * h:g}" for h in bare["body_half_extents_mm"])
    return (
        f"`verify.py` searches the region this clamp line can reach for a "
        f"{' x '.join(str(d) for d in w['body_envelope_mm'])} mm tripod webcam: a "
        f"lattice over x, y, z, three body orientations, each carrying the strut it "
        f"would need, scored on G1 against **{w['bottles']} bottles** (40 random "
        f"draws plus the 32 parameter-box corners) x every planner grasp, on the "
        f"real gripper meshes, on G3 framing and on swept radius.\n\n"
        f"The bare body on its own fits comfortably — centre "
        f"({', '.join(f'{c:g}' for c in bare['centre_mm'])}) mm as a {body} mm "
        f"block, swept radius {bare['swept_radius_mm']:g} mm, bottle clearance "
        f"{bare['bottle_clearance_mm']:g} mm. **Add the "
        f"{w['cradle_wall_mm']:g} mm of material needed to hold it and nothing "
        f"fits at all**: the best remaining candidate is "
        f"{miss['bottle_clearance_mm']:g} mm of bottle clearance at a "
        f"{miss['swept_radius_mm']:g} mm swept radius.\n")


def block_data(v, spec, payload, sim):
    prims = payload["primitives"]
    boxes = sum(1 for x in prims if x["type"] == "box")
    cyls = sum(1 for x in prims if x["type"] == "cylinder")
    cover = v["collision_cover"]
    return (
        f"- `camera_spec.json` — lens pose in `gripper_link`, metres, with the "
        f"4 x 4 `T_gripper_camera` in OpenCV convention (z forward, y down), the "
        f"two lens options and the resolution. Stored to 9 decimals, so a "
        f"simulator that rebuilds the camera from it disagrees by rounding at "
        f"1e-9 rather than 8e-7. It carries the payload mass and the lens "
        f"argument's own numbers whichever command wrote it — it used to be "
        f"complete only if `verify.py` ran last.\n"
        f"- `collision/payload.json` — {boxes} boxes and {cyls} cylinders covering "
        f"the payload, in metres, plus mass ({payload['mass_kg'] * 1000:g} g), CoM "
        f"and inertia about the gripper origin. Deliberately not a convex hull: "
        f"the free corridor between the clamp band and the camera is where the "
        f"bottle goes. The cover is verified ({cover['uncovered']} uncovered "
        f"surface samples at {cover['tolerance_mm']:g} mm, union volume "
        f"{cover['volume_ratio']:g} x the payload's and "
        f"{cover['union_volume_cm3']:g} cm³ in absolute terms against the previous "
        f"revision's 90.8) and the primitives pass G1 on their own at "
        f"{v['collision_primitive_bottles']['min_clearance_mm']:g} mm.\n"
        f"- `mount_on_gripper.glb` — the assembly on the real gripper meshes, in "
        f"metres.\n")


def _filter_counts(sim):
    for phase in (sim or {}).get("phases", []):
        if phase.get("phase") != "filter":
            continue
        for row in phase["rows"]:
            if "data" in row and "collision_filter" in row["data"]:
                return row["data"]["collision_filter"]
    return None


def block_sim_verdict(v, spec, payload, sim):
    """How many of the harness's checks pass, and exactly which do not.

    Typed, this paragraph went stale the moment a seed range was added: it said
    "nine runs" and "38 of its 41 checks".
    """
    if not sim or "phases" not in sim:
        return ("_run `sim/test_wrist_camera.py` to record the harness's "
                "verdict._\n")
    runs = len(sim["phases"])
    # Which run each check came from: the same check name appears once per run,
    # and without the run it reads as a duplicate line.
    checks = [dict(r, run=p["phase"] + (" --guard" if p.get("guard") else "")
                   + (f", seeds {p['seed_range']}" if p.get("seed_range") else ""))
              for p in sim["phases"] for r in p["rows"] if "ok" in r]
    bad = [r for r in checks if not r["ok"]]
    out = (f"**{len(checks) - len(bad)} of {len(checks)} checks pass**, over "
           f"{runs} runs and the seed ranges "
           f"{', '.join(sim.get('seed_ranges', []))}.")
    if not bad:
        out += " Nothing fails.\n"
        return out
    names = sorted({r["name"] for r in bad})
    out += (f" The {len(bad)} that do not are "
            + (f"{len(names)} distinct checks, some of them failing in both the "
               f"as-shipped and the patched run"
               if len(names) < len(bad) else "these")
            + ":\n\n")
    for row in bad:
        out += (f"- `{row['name']}` (run `{row['run']}`) — measured "
                f"{row['got']}, wanted {row['want']}\n")
    hands = {"right" in r["name"] for r in bad}
    if hands == {True}:
        out += ("\nEvery one of them is the **right**-hand mount. Build the left "
                "hand and the whole harness passes.\n")
    return out


def block_filter_counts(v, spec, payload, sim):
    """What the 1b patch is worth, measured on the shipped geometry.

    These two counts used to be a present-tense sentence carrying a measurement
    taken on a previous revision of the bracket.  The mechanism is a string-prefix
    test and does not move; the counts do.
    """
    counts = _filter_counts(sim)
    if not counts:
        return ("_run `sim/test_wrist_camera.py --phase filter` to measure what "
                "this patch is worth on the current geometry._\n")
    right = counts["right"]
    return (f"Measured on pour seed 0, jaw {right['jaw_half_gap_mm']:g} mm, over "
            f"**{right['configurations']:,} random configurations** of the six arm "
            f"joints (`--phase filter`): "
            + "; ".join(
                f"**{c['mount_against_arm']}** put a mount geom against another "
                f"`arm/` body with the {hand} hand and `Motion.collides` returned "
                f"`None` for **{c['shipped_filter_missed']}** of them"
                for hand, c in sorted(counts.items()))
            + ". With the patch below, the same counts and "
            + f"**{right['patched_filter_missed']}**.\n")


def block_mount_depth(v, spec, payload, sim):
    """How far out the mount's surface sits inside the frustum."""
    occ = v["G3_occlusion"]["recommended_wide"]["self_and_gripper"]
    span = occ.get("mount_depth_in_frustum_mm")
    alone = occ["mount_alone_image_fraction"]
    seen = occ["mount_image_fraction"]
    if not span:
        return "_run verify.py to measure the mount's depth inside the frustum._\n"
    return (f"the mount's surface inside the frustum spans "
            f"**{span[0]:g}-{span[1]:g} mm** along the optical axis "
            f"(`validation.json` → `G3_occlusion.recommended_wide."
            f"self_and_gripper.mount_depth_in_frustum_mm`), so thousands of its "
            f"pixels are past it. They are hidden because **the gripper is in "
            f"front of the strut** on every one of those rays, which is geometry "
            f"and holds on the real camera too. The CAD ray cast agrees once the "
            f"gripper is included as an occluder: {pct(alone)} of the wide frame "
            f"is mount with the gripper removed, {pct(seen)} with it there.\n")


def block_dataset_handedness(v, spec, payload, sim):
    """The recommendation, with the range it was decided on."""
    audit = v["G5_arm_audit"]
    census = _view_census(sim)
    lines = [f"Recommended hand: **left**. {_vote_sentence(sim)}"]
    if census:
        lines.append(
            "The two hands do not see the same task either — glass in frame at "
            "the grasp, per range: "
            + "; ".join(
                f"seeds {span}, right {c.get('right', {}).get('grasp: glass', 0)}"
                f" / left {c.get('left', {}).get('grasp: glass', 0)}"
                for span, c in sorted(census.items()))
            + ". The bottle mouth is in frame on essentially none of them.")
    lines.append(
        f"What the left costs is the full-arm audit: "
        f"{audit['reported_not_gated']['left']} payload collisions in "
        f"{audit['samples']:,} configurations against the right hand's "
        f"{audit['gate_hand_collisions']}, on the wrong side of v4's "
        f"{audit['v4_board_reference']} — a disclosed miss, with about ±8 of "
        f"sampling noise on each count. With the §1b patch the planner refuses "
        f"those configurations; without it, build the right hand and expect the "
        f"glass contact. The README's *Wrist roll* section has every table.")
    return "\n".join(lines) + "\n"


BLOCKS = {
    "sim-verdict": block_sim_verdict,
    "filter-counts": block_filter_counts,
    "mount-depth": block_mount_depth,
    "dataset-handedness": block_dataset_handedness,
    "spread": block_spread,
    "data": block_data,
    "g8": block_g8,
    "g4": block_g4,
    "sim-findings": block_sim_findings,
    "summary": block_summary, "gates": block_gates, "visibility": block_visibility,
    "lens": block_lens, "bottle-image": block_bottle_image,
    "corridor": block_corridor, "rolldip": block_rolldip, "v4": block_v4,
    "sim": block_sim, "handedness": block_handedness, "limits": block_limits,
    "what-if": block_what_if, "overhang": block_overhang,
}


def render(text, data, where):
    unknown = []

    def replace(match):
        name = match.group(2)
        if name not in BLOCKS:
            unknown.append(name)
            return match.group(0)
        return match.group(1) + BLOCKS[name](*data) + match.group(4)

    out = BLOCK.sub(replace, text)
    if unknown:
        raise SystemExit(f"{where}: no renderer for block(s) {unknown}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the generated blocks in place instead of "
                             "checking them")
    args = parser.parse_args(argv)
    data = (load(ROOT / "validation.json"), load(ROOT / "camera_spec.json"),
            load(ROOT / "collision/payload.json"),
            load(ROOT / "sim/validation.json", required=False))
    total, used, bad = 0, set(), 0
    for path in DOCS:
        name = path.relative_to(ROOT)
        text = path.read_text()
        found = {m.group(2) for m in BLOCK.finditer(text)}
        used |= found
        total += len(found)
        fresh = render(text, data, name)
        if args.write:
            path.write_text(fresh)
            continue
        if fresh == text:
            continue
        bad += 1
        import difflib
        diff = list(difflib.unified_diff(text.splitlines(), fresh.splitlines(),
                                         str(name), "rendered", lineterm="", n=1))
        print("\n".join(diff[:80]))
        print(f"\n{name} disagrees with the files it quotes "
              f"({len([d for d in diff if d.startswith('-') and not d.startswith('---')])}"
              f" line(s)).")
    missing = sorted(set(BLOCKS) - used)
    tail = f"; unused renderers: {missing}" if missing else ""
    if args.write:
        print(f"{total} generated block(s) rewritten in "
              f"{len(DOCS)} file(s){tail}")
        return 0
    if bad:
        print("Run: python render_readme.py --write")
        return 1
    print(f"{total} generated block(s) match the files, in "
          f"{len(DOCS)} file(s){tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
