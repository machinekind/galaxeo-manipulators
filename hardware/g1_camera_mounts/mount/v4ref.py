"""v4's swept radius, recomputed from v4's own files instead of remembered.

The README used to quote "v4: 118 mm" and "143 mm" for the webcam variant.
Neither number came out of anything in this directory -- v1..v4 are in git
history, not in the tree -- so a reader had no way to check them and the
comparison was worth nothing.

This reads v4's collision STLs straight out of the commit that added them and
measures them the way ``checks.g5_swept_radius`` measures this revision's
payload: the largest distance of any vertex from the wrist-roll (X) axis, in the
``gripper_link`` frame.  That is the right comparison because v4 attached its
payload to ``gripper_link`` with an identity origin --

    <joint name="camera_board_payload_fixed" type="fixed">
      <parent link="gripper_link" /> <child link="camera_board_payload" />
      <origin xyz="0 0 0" rpy="0 0 0" />

-- and its meshes are millimetres (the URDF scales them by 0.001).

``git show`` needs the repository's history, and a copy of this directory does
not have it.  So the measured radii are also *written down*, in the tracked file
``v4_reference.json``: a run with the history recomputes them and writes that
file, a run without it reads them back.  Either way ``reference()`` returns the
same block, which is what makes ``verify.py --check`` a real comparison outside a
checkout -- it used to exit 1 there, on this one block, and a plain `verify.py`
run outside a checkout silently rewrote the comparison out of the README.

If neither the history nor the cache is there, the numbers are reported as
unavailable with the reason, and the README renderer drops the comparison rather
than printing a number nothing produced.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CACHE_NAME = "v4_reference.json"

# The commit that added v1-v4; `git log` in this repo, or the README's Layout
# section, which says earlier revisions live in history.
V4_COMMIT = "c80b163"
V4_DIR = "hardware/g1_camera_mounts/v4/collision"

VARIANTS = {
    "board": ["board_upper", "board_lower", "board_camera",
              "board_clamp_fastener_-1", "board_clamp_fastener_1",
              "board_pcb_fastener_-1_-1", "board_pcb_fastener_-1_1",
              "board_pcb_fastener_1_-1", "board_pcb_fastener_1_1"],
    "webcam": ["webcam_upper", "webcam_lower", "webcam_camera",
               "webcam_tripod_head", "webcam_clamp_fastener_-1",
               "webcam_clamp_fastener_1"],
}


def _show(path):
    return subprocess.run(["git", "show", f"{V4_COMMIT}:{V4_DIR}/{path}"],
                          cwd=REPO, check=True, capture_output=True).stdout


def swept_radii(commit=V4_COMMIT):
    """{variant: {swept_radius_mm, per_part}} for v4's two payloads."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="v4ref_") as scratch:
        for variant, stems in VARIANTS.items():
            per, worst = {}, 0.0
            for stem in stems:
                path = Path(scratch) / f"{stem}.stl"
                path.write_bytes(_show(f"{stem}.stl"))
                mesh = trimesh.load(path)
                radius = float(np.hypot(mesh.vertices[:, 1],
                                        mesh.vertices[:, 2]).max())
                per[stem] = round(radius, 1)
                worst = max(worst, radius)
            out[variant] = dict(swept_radius_mm=round(worst, 1),
                                per_part_mm=per,
                                at_part=max(per, key=per.get))
    return out


def from_git():
    """The radii out of the history, or None with the reason it could not be read."""
    try:
        return swept_radii(), None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
        return None, str(error)[:200]


def cache_text(variants):
    """Exactly what the tracked cache file holds, for a given measurement."""
    return json.dumps(dict(commit=V4_COMMIT, path=V4_DIR, variants=variants),
                      indent=2) + "\n"


def reference(cache_dir=None):
    """The comparison block validation.json carries, or why it is unavailable.

    Identical whether it came from the history or from the cache, on purpose: a
    block that said which would make ``--check`` fail in a copy of the tree for a
    reason that has nothing to do with the design.
    """
    block = dict(commit=V4_COMMIT, path=V4_DIR, frame="gripper_link",
                 method="max hypot(y, z) over every vertex of v4's collision "
                        "STLs, which its URDF fixes to gripper_link with an "
                        "identity origin and scales by 0.001 -- the same "
                        "quantity checks.g5_swept_radius measures here",
                 source="recomputed from the history where it is available, read "
                        "back from the tracked " + CACHE_NAME + " where it is not; "
                        "a run that has both asserts they agree")
    cache = Path(cache_dir or ROOT) / CACHE_NAME
    cached = None
    if cache.exists():
        try:
            cached = json.loads(cache.read_text()).get("variants") or None
        except (ValueError, OSError):
            cached = None
    measured, error = from_git()
    if measured is not None:
        block.update(variants=measured, available=True,
                     agrees_with_cache=bool(cached is None or cached == measured))
    elif cached is not None:
        block.update(variants=cached, available=True, agrees_with_cache=True)
    else:
        block.update(variants={}, available=False,
                     unavailable_because=f"no git history ({error}) and no "
                                         f"{CACHE_NAME} beside it")
    return block


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2))
