#!/usr/bin/env python3
"""Open the A1X scene in the interactive MuJoCo viewer.

    sim/.venv/bin/python sim/view.py              # two arms on a table
    sim/.venv/bin/python sim/view.py sim/a1x.xml  # a single arm, no scene

Drag the actuator sliders in the right-hand "Control" panel to move joints;
double-click a link and ctrl-drag to apply a force. Backspace resets.
"""
import os
import sys

import mujoco
import mujoco.viewer

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "scene.xml")
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, max(model.key("home").id, 0))
mujoco.viewer.launch(model, data)
