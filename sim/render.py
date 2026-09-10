#!/usr/bin/env python3
"""Render a still of the scene to a PNG, no window needed.

    sim/.venv/bin/python sim/render.py out.png
"""
import os
import sys

import mujoco
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "a1x_scene.png"
model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "scene.xml"))
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, max(model.key("home").id, 0))
mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera()
cam.lookat[:] = (0.1, 0, 0.9)
cam.distance, cam.azimuth, cam.elevation = 1.8, 140, -20

with mujoco.Renderer(model, height=1000, width=1600) as r:
    r.update_scene(data, camera=cam)
    img = r.render()

try:
    from PIL import Image
    Image.fromarray(img).save(out)
except ImportError:                                  # minimal PPM fallback
    out = os.path.splitext(out)[0] + ".ppm"
    with open(out, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (img.shape[1], img.shape[0])); f.write(img.tobytes())
print("wrote", out)
