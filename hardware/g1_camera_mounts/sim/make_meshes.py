#!/usr/bin/env python3
"""Bake the payload into the two STLs the simulator loads.

    python -m sim.make_meshes            # from hardware/g1_camera_mounts

Writes ``sim/meshes/wrist_mount_right.stl`` and ``..._left.stl``: the whole
payload (both brackets, the camera module envelope and the fasteners) as one
mesh, in the ``gripper_link`` frame, **in metres**, which is the convention the
vendor meshes in ``ros2_ws`` already use, so ``wrist_camera.py`` can reference
them with no ``scale`` attribute.

This is a build step, not a runtime dependency: it needs cadquery, the
simulator module does not.  The left-hand mesh is the right one mirrored in Y
so the two are exactly each other's mirror rather than two independent
tessellations.  Both are asserted watertight, consistently wound and of positive
volume before export.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mount import checks, design  # noqa: E402

OUT = Path(__file__).resolve().parent / "meshes"
# The fastener envelope is bolt heads and dowels sunk in the brackets; they add
# nothing a renderer can see and a lot of triangles, so they stay out.
SKIP = ("m4_", "dowel_")
# Coarser than the print STLs on purpose: this mesh is only ever drawn, never
# measured, and the gates run on `mount.checks`, not on this file.
TOLERANCE, ANGULAR = 0.25, 0.45


def payload_mesh(cam=None):
    """Whole payload as one trimesh, gripper_link frame, millimetres."""
    parts = design.payload_parts(cam or design.CameraFrame())
    keep = [checks.to_mesh(solid, TOLERANCE, ANGULAR)
            for name, solid in sorted(parts.items()) if not name.startswith(SKIP)]
    return trimesh.util.concatenate(keep)


def main():
    OUT.mkdir(exist_ok=True)
    right = payload_mesh()
    right.apply_scale(0.001)
    left = right.copy()
    # trimesh already re-winds the faces for a negative-determinant transform, so
    # the `left.invert()` that used to be here turned the mesh inside out: volume
    # -79.4 cm3, and MuJoCo carried the inverted winding into the model.
    left.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
    for name, mesh in (("right", right), ("left", left)):
        assert mesh.is_watertight and mesh.is_winding_consistent, f"{name}: bad mesh"
        assert mesh.volume > 0, f"{name}: inside out (volume {mesh.volume})"
        path = OUT / f"wrist_mount_{name}.stl"
        mesh.export(path)
        lo, hi = mesh.bounds
        print(f"{path.name}: {len(mesh.faces)} faces, "
              f"x {lo[0] * 1000:+.1f}..{hi[0] * 1000:+.1f} "
              f"y {lo[1] * 1000:+.1f}..{hi[1] * 1000:+.1f} "
              f"z {lo[2] * 1000:+.1f}..{hi[2] * 1000:+.1f} mm")


if __name__ == "__main__":
    main()
