"""README images.

    python -m mount.render

A small painter's-algorithm z-buffer rasteriser (carried over from v4) draws
trimeshes directly, so the renders are of the same geometry the gates were run
on -- not a separate illustration that could drift from it.

Produces:
    render_assembly.png      the two parts on the real wrist, two views
    render_wrist_view.png    what the camera sees, through the declared FoV
    render_bottle.png        a held bottle against the payload, tightest grasp
    render_print.png         both parts as they sit on the bed
"""
import os
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("MPLCONFIGDIR", "/tmp/g1-mpl")
import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.colors                                   # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402

from . import checks, design, params as P, robot as rb     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = (230, 237, 240)
INK = "#173b4c"
COLOURS = {"upper_bracket": "#dc942f", "lower_bracket": "#158d8c",
           "camera_body": "#267650", "camera_holder": "#2f8a5e", "camera_lens": "#1c3747",
           "camera_usb": "#8a8f94"}


def raster(items, eye, rotation, width=900, height=680, hfov=60.0, vfov=45.0):
    """Depth-buffered triangle rasteriser.  `rotation` columns: forward, left, up."""
    rgb = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    depth = np.full((height, width), np.inf)
    fx = width / 2 / np.tan(np.radians(hfov / 2))
    fy = height / 2 / np.tan(np.radians(vfov / 2))
    for mesh, colour in items:
        local = (mesh.vertices - eye) @ rotation
        uv = np.column_stack((width / 2 - local[:, 1] * fx / local[:, 0],
                              height / 2 - local[:, 2] * fy / local[:, 0]))
        for index, face in enumerate(mesh.faces):
            z = local[face, 0]
            if z.min() < 1.0:
                continue
            tri = uv[face]
            x0 = max(0, int(np.floor(tri[:, 0].min())))
            x1 = min(width - 1, int(np.ceil(tri[:, 0].max())))
            y0 = max(0, int(np.floor(tri[:, 1].min())))
            y1 = min(height - 1, int(np.ceil(tri[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            (ax, ay), (bx, by), (cx, cy) = tri
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-8:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            xx = xx + 0.5
            yy = yy + 0.5
            a = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / den
            b = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / den
            c = 1 - a - b
            inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
            zz = 1 / (a / z[0] + b / z[1] + c / z[2])
            window = depth[y0:y1 + 1, x0:x1 + 1]
            mask = inside & (zz < window)
            window[mask] = zz[mask]
            shade = 0.62 + 0.38 * abs(mesh.face_normals[index]
                                      @ np.array([0.3, -0.4, 0.866]))
            rgb[y0:y1 + 1, x0:x1 + 1][mask] = (
                np.array(matplotlib.colors.to_rgb(colour)) * 255 * shade)
    return rgb


def look_at(eye, target):
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    return np.column_stack([forward, left, np.cross(forward, left)])


def gripper_items(jaw=20.0, colour="#8fa0ac"):
    return [(mesh, colour) for mesh in rb.gripper_meshes(jaw).values()]


def payload_items(cam=None, meshes=None):
    meshes = meshes if meshes is not None else checks.payload_meshes(cam)
    return [(mesh, COLOURS.get(name, "#9aa6b0")) for name, mesh in meshes.items()]


def held_bottle_mesh(bottle, z_rel, depth, pitch, sections=48):
    """The tightest-case held bottle as a mesh, in the gripper frame."""
    base, up = checks.bottle_frame(bottle, z_rel, depth, pitch)
    profile = checks.bottle_profile(bottle)
    mesh = trimesh.creation.revolve(profile, sections=sections)
    forward = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
    side = np.cross(up, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([forward, side, up])
    transform[:3, 3] = base
    mesh.apply_transform(transform)
    return mesh


def render_assembly(cam, meshes, path):
    items = gripper_items() + payload_items(meshes=meshes)
    views = [([130.0, -195.0, 150.0], [5.0, 8.0, 14.0], "three-quarter, camera side"),
             ([-30.0, 215.0, 95.0], [-18.0, 0.0, 12.0], "from behind the far jaw")]
    frames = [raster(items, eye, look_at(eye, target), hfov=55, vfov=42)
              for eye, target, _ in views]
    _compose(frames, [v[2] for v in views],
             "THE MOUNT ON THE REAL G1 WRIST",
             "Vendor gripper and finger meshes at a 40 mm jaw opening. "
             "Orange: upper bracket. Teal: lower bracket. Green: camera module.",
             path)
    return frames


def render_wrist_view(cam, meshes, path, fovs=(P.FOV_ALTERNATIVE, P.FOV_RECOMMENDED)):
    """What the camera sees: the gripper, the mount itself and a held bottle."""
    rng = np.random.default_rng(3)
    bottle = checks.sample_bottle(rng)
    bottle.update(body_r=36.0, body_h=170.0, neck_r=15.0, neck_h=60.0,
                  shoulder_h=35.0)
    bottle["height"] = (bottle["body_h"] + bottle["shoulder_h"]
                        + bottle["neck_h"] + 6.0)
    held = held_bottle_mesh(bottle, bottle["body_h"] - 30.0, -20.0, 0.45)
    mount_only = {name: meshes[name] for name in design.PRINTED_PARTS}
    rotation = np.column_stack([cam.axis, -cam.right, cam.up])
    frames, titles = [], []
    for fov in fovs:
        scene = gripper_items(20.0) + payload_items(meshes=mount_only)
        frames.append(raster(scene, cam.pupil, rotation, 640, 480,
                             hfov=fov[0], vfov=fov[1]))
        titles.append(f"{fov[0]:.0f} x {fov[1]:.0f} deg, empty hand")
        frames.append(raster(scene + [(held, "#4f7fa8")], cam.pupil, rotation,
                             640, 480, hfov=fov[0], vfov=fov[1]))
        titles.append(f"{fov[0]:.0f} x {fov[1]:.0f} deg, body-high grasp")
    _compose(frames, titles, "WHAT THE WRIST CAMERA SEES",
             "Cast from the declared entrance pupil through the declared field "
             "of view. Both fingertips, the far jaw and the scene ahead are in "
             "frame; the near blade hides a sliver of the gap behind itself.",
             path, columns=2)
    return frames


def render_bottle_clearance(cam, meshes, path):
    """The binding grasp: body-mid, pitch 0.70, the largest bottle."""
    bottle = dict(body_r=42.0, body_h=200.0, shoulder_h=50.0, neck_r=18.0,
                  neck_h=80.0, lip=2.5)
    bottle["height"] = (bottle["body_h"] + bottle["shoulder_h"]
                        + bottle["neck_h"] + 6.0)
    held = held_bottle_mesh(bottle, bottle["body_h"] / 2.0, -20.0, 0.70)
    points = checks.payload_points(meshes)
    gap = checks.bottle_clearance(points, bottle, bottle["body_h"] / 2.0, -20.0, 0.70)
    items = gripper_items(40.0) + payload_items(meshes=meshes) + [(held, "#4f7fa8")]
    views = [([90.0, -330.0, 110.0], [12.0, 0.0, 70.0], "from -Y: the bottle slab"),
             ([-25.0, -45.0, 330.0], [-18.0, 12.0, 25.0], "from above")]
    frames = [raster(items, eye, look_at(eye, target), hfov=52, vfov=40)
              for eye, target, _ in views]
    _compose(frames, [v[2] for v in views], "HELD BOTTLE AGAINST THE PAYLOAD",
             f"The binding case of G1's whole sweep: body-mid grasp, pitch 0.70, "
             f"the all-maximum bottle of the parameter box "
             f"({bottle['body_r']:.0f} mm x {bottle['body_h']:.0f} mm body). "
             f"Measured clearance {gap:.2f} mm.",
             path)
    return gap


def render_print_layout(path):
    """Both brackets as the slicer will see them, including from bed level.

    Two panels on purpose.  The three-quarter view says what is on the bed; the
    elevation, with the bed drawn as a slab and the eye 8 mm above it, is the one
    that shows *what each part stands on*.  The upper bracket used to stand on a
    single nub -- the zip-tie lug hanging 2.7 mm below the split plane -- with its
    1278 mm2 split face in the air, and a view from above could not tell you.
    """
    items, cursor = [], 0.0
    contact = checks.bed_contact(ROOT / "stl")["per_file"]
    for name, colour in (("upper_bracket_right", "#dc942f"),
                         ("lower_bracket", "#158d8c"),
                         ("bore_gauge", "#6b7f8c"), ("rail_key_gauge", "#6b7f8c")):
        stl = ROOT / "stl" / f"{name}.stl"
        if not stl.exists():
            continue
        mesh = trimesh.load(stl)
        # lay the parts out on the bed the way a slicer would, edge to edge
        mesh.apply_translation([-mesh.bounds[0, 0], cursor - mesh.bounds[0, 1], 0])
        cursor += mesh.extents[1] + 12.0
        items.append((mesh, colour))
    if not items:
        return None
    span = max(m.bounds[1, 1] for m, _ in items)
    reach = max(m.bounds[1, 0] for m, _ in items)
    bed = trimesh.creation.box(extents=(reach + 60.0, span + 60.0, 3.0))
    bed.apply_translation([(reach + 60.0) / 2 - 30.0, span / 2, -1.5])
    eye = np.array([-360.0, span / 2 - 260.0, 330.0])
    top = raster(items + [(bed, "#c2ced6")], eye,
                 look_at(eye, [30.0, span / 2, 15.0]),
                 width=1100, height=620, hfov=52, vfov=32)
    # Elevation: eye 9 mm above the bed, looking across it along +X, so every part
    # is side on and a part that is not touching shows a gap against the slab.
    eye2 = np.array([-520.0, span / 2, 9.0])
    side = raster(items + [(bed, "#c2ced6")], eye2,
                  look_at(eye2, [40.0, span / 2, 13.0]),
                  width=1100, height=330, hfov=48, vfov=15)
    fractions = ", ".join(
        f"{k.replace('_right', '')} {contact[k]['bed_contact_mm2']:.0f} mm2 "
        f"({contact[k]['fraction']:.0%} of its split face)"
        for k in ("upper_bracket_right", "lower_bracket") if k in contact)
    _compose([top, side],
             ["as exported, on the bed: gauges, lower strap, upper bracket",
              "bed level, eye 9 mm up: both brackets stand on their split faces"],
             "PRINT ORIENTATION",
             "Both brackets print split-face down: the clamp hoop, the locating "
             "yoke and the strut all get layers along their load paths. Print "
             "the two gauges first and check them on the real gripper.\n"
             f"Measured bed contact on the exported STLs: {fractions}.",
             path, columns=1)
    return top


def _compose(frames, titles, heading, subtitle, path, columns=2):
    """Grid of frames under a heading, sized from the frames' own aspect.

    The header grows with the subtitle: at 6.6 in of panel width a subtitle wraps
    at roughly 95 characters, and a fixed 1.15 in of header let a three-line one
    run under the first panel's own title.
    """
    rows = int(np.ceil(len(frames) / columns))
    aspect = frames[0].shape[0] / frames[0].shape[1]
    lines = int(np.ceil(len(subtitle) / (14.0 * columns + 81.0))) + subtitle.count("\n")
    panel_w, header = 6.6, 0.95 + 0.20 * max(1, lines)
    panel_h = panel_w * aspect + 0.36
    figure = plt.figure(figsize=(panel_w * columns + 0.5,
                                 header + panel_h * rows), facecolor="#e6edf0")
    height = header + panel_h * rows
    figure.text(0.03, 1 - 0.42 / height, heading, fontsize=20, weight="bold",
                color=INK, va="top")
    figure.text(0.03, 1 - 0.78 / height, subtitle, fontsize=10, color=INK,
                va="top", wrap=True)
    for index, (frame, title) in enumerate(zip(frames, titles)):
        row, column = divmod(index, columns)
        aspect = frame.shape[0] / frame.shape[1]
        x = (0.25 + column * panel_w) / (panel_w * columns + 0.5)
        width = (panel_w - 0.3) / (panel_w * columns + 0.5)
        y = (panel_h * (rows - 1 - row) + 0.12) / height
        axes = figure.add_axes([x, y, width, (panel_h - 0.36) / height])
        axes.imshow(frame)
        axes.axis("off")
        figure.text(x, y + (panel_h - 0.30) / height, title, fontsize=10.5,
                    weight="bold", color=INK)
    figure.savefig(path, dpi=130, facecolor="#e6edf0")
    plt.close(figure)


def main():
    cam = design.CameraFrame()
    meshes = checks.payload_meshes(cam)
    render_assembly(cam, meshes, ROOT / "render_assembly.png")
    render_wrist_view(cam, meshes, ROOT / "render_wrist_view.png")
    render_bottle_clearance(cam, meshes, ROOT / "render_bottle.png")
    render_print_layout(ROOT / "render_print.png")
    print("wrote render_assembly.png render_wrist_view.png render_bottle.png "
          "render_print.png")


if __name__ == "__main__":
    main()
