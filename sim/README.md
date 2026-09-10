# A1X in MuJoCo

The Galaxea A1X arm as a MuJoCo model, built from the URDF and meshes in
`ros2_ws/src/galaxea_a1xy_description/`. No ROS, no CAN, no hardware.

```bash
uv venv --python 3.12 sim/.venv && uv pip install --python sim/.venv/bin/python mujoco numpy pillow
sim/.venv/bin/python sim/view.py          # two arms on a table, interactive
sim/.venv/bin/python sim/render.py out.png  # headless still
sim/.venv/bin/mjpython sim/handover.py      # ball handover demo, live (macOS needs mjpython)
sim/.venv/bin/python sim/handover.py --gif handover.gif   # same, recorded headless
```

| | |
| --- | --- |
| `urdf2mjcf.py` | generates `a1x.xml` from the URDF. Rerun after editing the URDF |
| `a1x.xml` | the arm: 6 hinge joints + 2 finger slides, URDF inertials, position servos, `home` keyframe |
| `scene.xml` | floor, table, two arms (`leader/`, `follower/`) attached from `a1x.xml` |
| `view.py` | interactive viewer. Actuator sliders live in the Control panel |
| `render.py` | offscreen render to PNG |
| `handover.py` | scripted demo: one arm picks a ball, hands it to the other, which sets it down |

What the URDF does not say and this adds: position actuators with kp per
joint and force limits from the URDF effort ratings, an equality constraint
that slaves finger 2 to finger 1 (the URDF has no mimic tag), and joint
armature/damping so the integrator is stable at 2 ms steps.

The handover is plain position control with real contact physics: each
waypoint is a tool-centre-point pose solved by damped-least-squares IK on the
site Jacobian, joint targets ramp with a smoothstep, the gripper is a position
servo. Three things had to be true for the grasp to hold, and each cost a
debugging round:

* **Finger collision is boxes, not the mesh.** MuJoCo collides meshes as convex
  hulls, and the L-shaped finger (carriage + thin plate) hulls into a wedge
  with a slanted inner face that squeezes anything out. `urdf2mjcf.py` swaps in
  three boxes per finger that trace the carriage, the plate and the curled tip.
* **The TCP sits mid-plate**, 45 mm along the gripper x-axis, where the plates
  run parallel. The tips are 33 mm further out and curl inward, so they cage
  the object. Picking off a table therefore keeps the TCP 8 mm above the
  object centre to keep the tips off the surface.
* **Contacts are stiff.** A 50 g ball with default `solref`/`solimp` sinks
  millimetres into the fingers and creeps down under gravity. The ball sets
  `priority="1"`, a short `solref` and high `solimp`; the scene adds
  `noslip_iterations`.

Joint order matches the driver: `arm_joint1..6`, then the gripper.
