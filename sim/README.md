# A1X in MuJoCo

The Galaxea A1X arm as a MuJoCo model, built from the URDF and meshes in
`ros2_ws/src/galaxea_a1xy_description/`. No ROS, no CAN, no hardware.

```bash
uv venv --python 3.12 sim/.venv && uv pip install --python sim/.venv/bin/python mujoco numpy pillow
sim/.venv/bin/python sim/view.py          # two arms on a table, interactive
sim/.venv/bin/python sim/render.py out.png  # headless still
sim/.venv/bin/mjpython sim/handover.py      # ball handover demo, live (macOS needs mjpython)
sim/.venv/bin/python sim/handover.py --gif handover.gif   # same, recorded headless
sim/.venv/bin/python sim/planner/run_episodes.py --n 20 --seed 0   # pick and place
```

| | |
| --- | --- |
| `urdf2mjcf.py` | generates `a1x.xml` from the URDF. Rerun after editing the URDF |
| `a1x.xml` | the arm: 6 hinge joints + 2 finger slides, URDF inertials, position servos, `home` keyframe |
| `scene.xml` | floor, table, two arms (`leader/`, `follower/`) attached from `a1x.xml` |
| `view.py` | interactive viewer. Actuator sliders live in the Control panel |
| `render.py` | offscreen render to PNG |
| `handover.py` | scripted demo: one arm picks a ball, hands it to the other, which sets it down |
| `a1x_control.py` | `Arm` (DLS IK on the TCP site), `Script` (timed phases), `rot`, `smoothstep`, OPEN/CLOSED. Shared by `handover.py` and the planner |
| `pickplace_scene.py` | `build(seed) -> (model, data, info)`: one arm, a placeholder quadruped, 1-3 random objects. Assembled with `MjSpec` so each episode recompiles |
| `planner/perception.py` | `Perception` protocol; `SimPerception` reads ground truth from `MjData`, with optional noise |
| `planner/grasp.py` | top-down pinch grasps for the parallel jaw, ranked |
| `planner/motion.py` | multi-start IK plus a collision check on the waypoint and on the ramp leading to it |
| `planner/pickplace.py` | the pick-and-place state machine |
| `planner/run_episodes.py` | CLI: seeded episodes, success rate, GIF and `.npz` recording |

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

## Pick and place

One A1X on the table, a placeholder quadruped standing beside it at a random
pose, one to three random primitives on the table. The planner picks one up and
sets it on the dog's back.

```bash
sim/.venv/bin/python sim/planner/run_episodes.py --n 20 --seed 0
sim/.venv/bin/python sim/planner/run_episodes.py --n 1 --seed 4 --gif pickplace.gif
sim/.venv/bin/python sim/planner/run_episodes.py --n 20 --record /tmp/pp   # .npz per episode
sim/.venv/bin/python sim/pickplace_scene.py 4 scene.png   # build one episode, still render
```

**20/20 on `--n 20 --seed 0`, and 100/100 on unseen seeds 100-199.**

Perception sits behind an interface. `SimPerception` reads object and tag poses
out of `MjData`; on the real robot the same two calls come from a detector plus
a grasp network and from an AprilTag on the dog's back. `--noise SIGMA` feeds
the planner Gaussian pose error, which is what the ground-truth number is worth
checking against: 1 cm of noise drops it to 14/20, 2 cm to 5/20, and the
failures land on `verify_grasp`, `verify_lift` and `verify_place` rather than
going unnoticed.

The state machine names every outcome, so a failure says which stage lost it:

    perceive -> plan_pick -> pick -> verify_grasp -> lift -> verify_lift
             -> travel -> place -> verify_place

`verify_grasp` reads the fingertip gap (3 mm closed, 103 mm open) and requires
it to be off the stop and no wider than the object. `verify_lift` checks the
object came up with the gripper. `verify_place` checks it is inside the platform
in the dog's own frame, at the right height, and has stopped moving. The dog
sways a few millimetres at 0.5 Hz (`--no-sway` freezes it), so the tag is re-read
immediately before the final descent.

Four things this cost, on top of the three the handover already paid for:

* **The reachable set for a top-down wrist is small.** Joint 2 is limited to
  `[0, pi]` and joint 3 to `[-3.32, 0]`, so there is only an elbow-down branch,
  and `x = 0` is a shoulder singularity. Single-start DLS lands in a local
  minimum with a 30 cm residual over most of the workspace. `Motion.solve`
  restarts from a fan of seeds -- the previous waypoint first, so consecutive
  poses stay on one branch -- and the carry height and the wrist yaw over the
  dog are both candidate lists rather than constants. The arm cannot hold a
  top-down wrist much above base + 0.2 m, which is what caps the carry height.
* **Waypoints are not enough; the ramp between them has to be checked too.**
  `Motion` samples the smoothstep the servos actually follow and re-attaches the
  carried object to the TCP while doing it, so a swing that sweeps the payload
  through a neighbour is rejected.
* **The retreat backs out in short vertical hops.** A tall object often topples
  against a jaw as it is let go. One long joint-space ramp out of the release
  pose bows sideways and drags the object off the platform -- that alone was
  three failures in twenty.
* **Round objects need rolling friction.** The dog's back is flat, rigid and
  sways; with the handover ball's `condim="4"` a sphere or a toppled capsule has
  nothing to stop it and rolls off every time. Objects here use `condim="6"` and
  `friction="1.5 0.02 0.005"`; everything else about the contact is the ball's
  recipe. That is the only physics parameter changed, and it is a real effect --
  a 0.0001 m rolling coefficient is glass on glass.

Recording: `--record DIR` writes one `.npz` per episode with `time`, `qpos`,
`ctrl` and the TCP pose at 50 Hz, plus `table_cam` frames and their timestamps
at 10 Hz, ready for `ros2_ws/to_lerobot.py`-style conversion.
