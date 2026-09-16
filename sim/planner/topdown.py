#!/usr/bin/env python3
"""Redraw a camera frame as a calibrated top-down map of the table plane.

    sim/.venv/bin/python sim/planner/topdown.py [out.png]

The webcam stands somewhere unknown-but-calibrated in front of the arm, so a
policy fed raw frames has to learn 3D localisation from every viewpoint at
once. The calibration is known per session, so instead we warp each frame onto
the table plane: whatever the camera angle, an object standing on the table
lands at its true `(x, y)` in the map, at a fixed scale. The policy gets that
map next to the raw frame.

`topdown(img, K, T_cam2world, centre_xy)` is the warp; `map_of(p, ...)` is
where a world point lands in it. The map covers `side` metres square about
`centre_xy`, world **+y up** and **+x right**, `side / n` metres per pixel --
so the map is the table seen from above in the arm's own frame, not an image.

Only points *on the plane* `z = table_z` are where they belong; anything
standing up (the bottle's body, the arm) smears away from the camera, which is
the usual homography foreshortening and is consistent per camera pose.

`T_cam2world` is OpenCV convention (z forward, y down), which is what
`pour_scene._cam_pose_world` builds and what `info["cam"]["T_cam2world"]`
holds; `K` must be the intrinsics of the render size `img` actually has
(`pour_scene.cam_intrinsics(W, H, fovy)`).
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from pickplace_scene import TABLE_TOP  # noqa: E402

SIDE = 0.7      # metres of table covered by the map, per side
N = 256         # map pixels per side


def map_homography(K, T_cam2world, centre_xy, side=SIDE, n=N, table_z=TABLE_TOP):
    """The 3x3 that takes a map pixel `(u, v, 1)` to an image pixel.

    A map pixel is a point on the plane `z = table_z`:
    `p = [x0 + s u, y0 - s v, table_z]`, with `s = side / n` metres per pixel.
    Writing the camera projection of that point out gives one linear map from
    `(u, v, 1)`, which is the homography."""
    R, t = np.asarray(T_cam2world, float)[:3, :3], np.asarray(T_cam2world, float)[:3, 3]
    Rt = R.T
    s = side / n
    x0, y0 = centre_xy[0] - side / 2, centre_xy[1] + side / 2
    A = np.column_stack([Rt @ [s, 0, 0], Rt @ [0, -s, 0],
                         Rt @ (np.array([x0, y0, table_z]) - t)])
    return np.asarray(K, float) @ A


def topdown(img, K, T_cam2world, centre_xy, side=SIDE, n=N, table_z=TABLE_TOP):
    """`img` redrawn as an (n, n, 3) uint8 top-down map of the table plane."""
    Hm = map_homography(K, T_cam2world, centre_xy, side, n, table_z)
    return cv2.warpPerspective(img, Hm, (n, n),
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def map_of(p_world, centre_xy, side=SIDE, n=N):
    """The `(u, v)` map pixel of a world point (its z is ignored)."""
    s = side / n
    x0, y0 = centre_xy[0] - side / 2, centre_xy[1] + side / 2
    p = np.asarray(p_world, float)
    return np.array([(p[0] - x0) / s, (y0 - p[1]) / s])


def project(p_world, K, T_cam2world):
    """The image pixel of a world point, OpenCV convention."""
    T = np.asarray(T_cam2world, float)
    p_cam = T[:3, :3].T @ (np.asarray(p_world, float) - T[:3, 3])
    uv = np.asarray(K, float) @ p_cam
    return uv[:2] / uv[2]


def _ring(m, uv, r_px, colour, label=None):
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(m, (u, v), max(2, int(round(r_px))), colour, 1, cv2.LINE_AA)
    cv2.drawMarker(m, (u, v), colour, cv2.MARKER_CROSS, 7, 1)
    if label:
        cv2.putText(m, label, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)


def _main(out):
    from PIL import Image

    from pour_scene import build, cam_intrinsics, render, WORKSPACE

    model, data, info = build(2)
    cam = info["cam"]
    W, H = cam["W"], cam["H"]
    img = render(model, data, "laptop_cam", W, H)
    K = cam_intrinsics(W, H, cam["fovy"])
    T = np.asarray(cam["T_cam2world"], float)
    centre = WORKSPACE[:2]

    # The real invariant: the homography must send the map pixel of a world
    # point to the same image pixel the camera projection does. If those agree,
    # whatever the camera saw at that image pixel is what the warp puts at that
    # map pixel -- which is the claim the map makes.
    Hm = map_homography(K, T, centre)
    worst = 0.0
    for name, p in (("bottle base", info["bottle_pos"]), ("glass", info["glass_pos"]),
                    ("arm base", info["arm_base"])):
        m = map_of(p, centre)
        back = Hm @ [m[0], m[1], 1.0]
        back = back[:2] / back[2]
        want = project(np.array([p[0], p[1], TABLE_TOP]), K, T)
        err = float(np.linalg.norm(back - want))
        worst = max(worst, err)
        print(f"  {name:12s} world {np.round(p[:2], 3)}  map px {np.round(m, 1)}  "
              f"image px {np.round(want, 1)}  reprojection err {err:.4f} px")
    assert worst < 1.0, f"map_of / homography disagree with the projection by {worst:.3f} px"
    print(f"  worst homography-vs-projection error {worst:.4f} px (< 1 px)")

    m = topdown(img, K, T, centre)
    s = SIDE / N
    _ring(m, map_of(info["bottle_pos"], centre), info["bottle"]["body_r"] / s, (255, 40, 40), "bottle")
    _ring(m, map_of(info["glass_pos"], centre), info["glass"]["r"] / s, (40, 255, 40), "glass")
    _ring(m, map_of(info["arm_base"], centre), 0.03 / s, (60, 120, 255), "base")
    Image.fromarray(m).save(out)
    raw = os.path.splitext(out)[0] + "_raw.png"
    Image.fromarray(img).save(raw)
    print(f"wrote {out} and {raw}")


if __name__ == "__main__":
    import tempfile
    _main(sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(tempfile.gettempdir(), "topdown_seed2.png"))
