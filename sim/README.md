# A1X in MuJoCo

The Galaxea A1X arm as a MuJoCo model, built from the URDF and meshes in
`ros2_ws/src/galaxea_a1xy_description/`. No ROS, no CAN, no hardware.

```bash
uv venv --python 3.12 sim/.venv && uv pip install --python sim/.venv/bin/python mujoco numpy pillow
sim/.venv/bin/python sim/view.py          # two arms on a table, interactive
sim/.venv/bin/python sim/render.py out.png  # headless still
```

| | |
| --- | --- |
| `urdf2mjcf.py` | generates `a1x.xml` from the URDF. Rerun after editing the URDF |
| `a1x.xml` | the arm: 6 hinge joints + 2 finger slides, URDF inertials, position servos, `home` keyframe |
| `scene.xml` | floor, table, two arms (`leader/`, `follower/`) attached from `a1x.xml` |
| `view.py` | interactive viewer. Actuator sliders live in the Control panel |
| `render.py` | offscreen render to PNG |

What the URDF does not say and this adds: position actuators with kp per
joint and force limits from the URDF effort ratings, an equality constraint
that slaves finger 2 to finger 1 (the URDF has no mimic tag), and joint
armature/damping so the integrator is stable at 2 ms steps.

Joint order matches the driver: `arm_joint1..6`, then the gripper.
