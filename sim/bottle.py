"""Procedural bottles and a small glass for the pour task.

    from bottle import sample_bottle, bottle_xml, add_bottle_meshes, glass_xml

A bottle is a surface of revolution: a body, a shoulder curve, a neck and a
lip. The visual is one revolved mesh, clear to opaque, plus one or two label
bands of random colour and placement; the collision is a
stack of cylinders (body, three shoulder steps, neck) so the neck stays
graspable -- a convex hull of the whole bottle would fill the shoulder in.
Only the body cylinder carries mass. Every bottle gets a `mouth` site at the
centre of the opening, which is what the pour planner steers.

The glass is a bottom disk and a ring of thin boxes, so it is hollow for real
and can be knocked over. Its `rim` site is the centre of the opening.

Sizes are vodka-like: a 60 to 84 mm body, 220 to 320 mm tall, 26 to 36 mm neck.
"""
import numpy as np

from pickplace_scene import CONTACT

N_SEG = 36                    # facets around the axis
LABEL_RGBA = "0.92 0.92 0.88 1"
COLLISION_GROUP = 3           # not drawn by the default renderer


def sample_bottle(rng, name="bottle", arng=None):
    """One bottle. `rng` draws the shape, the mass and the first label exactly
    as it always has; `arng`, the scene's appearance stream, only adds the
    opaque override and the second band on top. Splitting the two streams is
    what keeps a seed's *task* -- the bottle it has to pick up -- fixed while
    what the camera sees around it is randomised."""
    arng = rng if arng is None else arng
    body_r = float(rng.uniform(0.030, 0.042))
    body_h = float(rng.uniform(0.12, 0.20))
    shoulder_h = float(rng.uniform(0.02, 0.05))
    neck_r = float(rng.uniform(0.013, 0.018))
    neck_h = float(rng.uniform(0.04, 0.08))
    lip = 0.0025
    height = body_h + shoulder_h + neck_h + 0.006
    tint = rng.uniform(0.15, 0.9, 3)
    alpha = float(rng.uniform(0.55, 0.95))         # glass ranges from clear to opaque
    mass = float(rng.uniform(0.3, 1.0))
    labels = [dict(z0=float(body_h * rng.uniform(0.25, 0.4)),
                   z1=float(body_h * rng.uniform(0.6, 0.85)),
                   rgba=np.concatenate([rng.uniform(0.1, 0.95, 3), [1.0]]))]
    if arng.random() < OPAQUE_P:                   # ... and one in three is painted solid
        alpha = 1.0
    if arng.random() < SECOND_BAND_P:
        second = _second_band(arng, labels[0], body_h)
        if second is not None:
            labels = sorted(labels + [second], key=lambda b: b["z0"])
    return dict(name=name, body_r=body_r, body_h=body_h, shoulder_h=shoulder_h,
                neck_r=neck_r, neck_h=neck_h, lip=lip, height=height, mass=mass,
                rgba=np.array([*tint, alpha]), opaque=bool(alpha >= 1.0), labels=labels)


OPAQUE_P = 1 / 3              # chance the bottle is painted solid rather than see-through
SECOND_BAND_P = 0.5           # chance of a narrow band above or below the main label


def _second_band(arng, first, body_h):
    """A narrow band in whatever room the main label leaves on the body."""
    z0, z1 = first["z0"] / body_h, first["z1"] / body_h
    gaps = [(a, b) for a, b in ((0.04, z0 - 0.03), (z1 + 0.03, 0.95)) if b - a > 0.07]
    if not gaps:
        return None
    a, b = gaps[arng.integers(len(gaps))]
    t0 = float(arng.uniform(a, b - 0.05))
    t1 = min(b, t0 + float(arng.uniform(0.04, 0.12)))
    return dict(z0=float(body_h * t0), z1=float(body_h * t1),
                rgba=np.concatenate([arng.uniform(0.1, 0.95, 3), [1.0]]))


def profile(b):
    """(r, z) polyline of the bottle wall, bottom to top, z from the base."""
    pts = [(1e-4, 0.0), (b["body_r"] - 0.004, 0.0), (b["body_r"], 0.005), (b["body_r"], b["body_h"])]
    for t in np.linspace(0, 1, 6)[1:]:                       # quarter ellipse shoulder
        r = b["neck_r"] + (b["body_r"] - b["neck_r"]) * np.cos(t * np.pi / 2)
        pts.append((r, b["body_h"] + b["shoulder_h"] * np.sin(t * np.pi / 2)))
    top = b["body_h"] + b["shoulder_h"] + b["neck_h"]
    pts += [(b["neck_r"], top), (b["neck_r"] + b["lip"], top + 0.001),
            (b["neck_r"] + b["lip"], top + 0.006), (b["neck_r"] - 0.003, top + 0.006),
            (b["neck_r"] - 0.003, top - 0.004), (1e-4, top - 0.004)]
    return np.array(pts)


def revolve(pts, n=N_SEG):
    """Triangle mesh of a profile revolved about z. Returns (verts, faces)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.array([[r * np.cos(a), r * np.sin(a), z] for r, z in pts for a in ang])
    faces = []
    for i in range(len(pts) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [[a, b, d], [a, d, c]]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def add_bottle_meshes(spec, b):
    """Register the revolved visual mesh for bottle `b` on an MjSpec."""
    v, f = revolve(profile(b))
    m = spec.add_mesh()
    m.name = f"{b['name']}_vis"
    m.uservert = v.ravel()
    m.userface = f.ravel()


def bottle_xml(b, pos, yaw):
    """MJCF body for one bottle standing at `pos` (base centre) with yaw."""
    q = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    contact = " ".join(f'{k}="{v}"' for k, v in CONTACT.items())
    top = b["body_h"] + b["shoulder_h"] + b["neck_h"]
    rgba = " ".join(f"{x:.3f}" for x in b["rgba"])
    geoms = [
        f'<geom name="{b["name"]}/vis" type="mesh" mesh="{b["name"]}_vis" rgba="{rgba}" '
        f'contype="0" conaffinity="0" mass="0" group="1"/>',
    ]
    for i, lab in enumerate(b["labels"]):
        geoms.append(
            f'<geom name="{b["name"]}/label{i}" type="cylinder" size="{b["body_r"] + 0.0006:.4f} '
            f'{(lab["z1"] - lab["z0"]) / 2:.4f}" pos="0 0 {(lab["z0"] + lab["z1"]) / 2:.4f}" '
            f'rgba="{" ".join(f"{x:.3f}" for x in lab["rgba"])}" contype="0" conaffinity="0" mass="0" group="1"/>')
    geoms += [
        # collision: body carries all the mass
        f'<geom name="{b["name"]}/body" type="cylinder" size="{b["body_r"]:.4f} {b["body_h"] / 2:.4f}" '
        f'pos="0 0 {b["body_h"] / 2:.4f}" mass="{b["mass"]:.4f}" group="{COLLISION_GROUP}" {contact}/>',
    ]
    # shoulder: three frustum steps, each a cylinder at the radius of its lower edge
    for k in range(3):
        t0, t1 = k / 3, (k + 1) / 3
        r = b["neck_r"] + (b["body_r"] - b["neck_r"]) * np.cos(t0 * np.pi / 2)
        z0 = b["body_h"] + b["shoulder_h"] * np.sin(t0 * np.pi / 2)
        z1 = b["body_h"] + b["shoulder_h"] * np.sin(t1 * np.pi / 2)
        geoms.append(
            f'<geom name="{b["name"]}/shoulder{k}" type="cylinder" size="{r:.4f} {(z1 - z0) / 2:.4f}" '
            f'pos="0 0 {(z0 + z1) / 2:.4f}" mass="0.001" group="{COLLISION_GROUP}" {contact}/>')
    geoms.append(
        f'<geom name="{b["name"]}/neck" type="cylinder" size="{b["neck_r"] + b["lip"]:.4f} '
        f'{(b["neck_h"] + 0.006) / 2:.4f}" pos="0 0 {b["body_h"] + b["shoulder_h"] + (b["neck_h"] + 0.006) / 2:.4f}" '
        f'mass="0.001" group="{COLLISION_GROUP}" {contact}/>')
    return (f'<body name="{b["name"]}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
            f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"><freejoint name="{b["name"]}"/>'
            + "".join(geoms)
            + f'<site name="{b["name"]}/mouth" pos="0 0 {top + 0.006:.4f}" size="0.004" rgba="0 1 0 0.6" group="4"/>'
            + f'<site name="{b["name"]}/base" pos="0 0 0" size="0.004" rgba="0 1 0 0.6" group="4"/>'
            + '</body>')


def sample_glass(rng, name="glass"):
    return dict(name=name, r=float(rng.uniform(0.020, 0.030)), h=float(rng.uniform(0.055, 0.085)),
                mass=float(rng.uniform(0.05, 0.12)), n_wall=12)


def glass_xml(g, pos):
    contact = " ".join(f'{k}="{v}"' for k, v in CONTACT.items())
    rgba = "0.80 0.90 1.0 0.35"
    n, r, h = g["n_wall"], g["r"], g["h"]
    wall_t = 0.0015
    half_w = r * np.tan(np.pi / n) + wall_t          # overlap the segments slightly
    geoms = [f'<geom name="{g["name"]}/bottom" type="cylinder" size="{r + wall_t:.4f} 0.003" pos="0 0 0.003" '
             f'mass="{g["mass"] * 0.5:.4f}" rgba="{rgba}" {contact}/>']
    for k in range(n):
        a = 2 * np.pi * k / n
        c, s = np.cos(a), np.sin(a)
        qz = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
        geoms.append(
            f'<geom name="{g["name"]}/wall{k}" type="box" size="{wall_t:.4f} {half_w:.4f} {h / 2:.4f}" '
            f'pos="{(r + wall_t) * c:.4f} {(r + wall_t) * s:.4f} {h / 2:.4f}" '
            f'quat="{qz[0]:.5f} {qz[1]:.5f} {qz[2]:.5f} {qz[3]:.5f}" '
            f'mass="{g["mass"] * 0.5 / n:.4f}" rgba="{rgba}" {contact}/>')
    return (f'<body name="{g["name"]}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}">'
            f'<freejoint name="{g["name"]}"/>' + "".join(geoms)
            + f'<site name="{g["name"]}/rim" pos="0 0 {h:.4f}" size="0.004" rgba="1 0 1 0.6" group="4"/>'
            + '</body>')
