# Bottle pour: sim, calibration, planner, data

The task: pick up a bottle of random shape (vodka-like) and pour it into a
small glass, with one laptop webcam placed somewhere unknown, no teleop, and
everything automated from calibration through data collection. This
document is the plan and what exists of it. Everything runs on the MuJoCo
model from [README.md](README.md).

```bash
sim/.venv/bin/python sim/pour_scene.py 3 scene.png                # one episode, laptop-cam still
sim/.venv/bin/python sim/calib/calibrate.py --sim --n 20            # calibration vs ground truth
sim/.venv/bin/python sim/planner/run_pour.py --n 20 --seed 0        # planner success rate
sim/.venv/bin/python sim/planner/run_pour.py --n 1 --seed 1 --gif pour.gif
sim/.venv/bin/mjpython sim/planner/run_pour.py --view --seed 1      # live (macOS needs mjpython)
sim/.venv-lerobot/bin/python sim/planner/run_pour.py --n 100 --lerobot data/pour_sim   # dataset
```

| | |
| --- | --- |
| `bottle.py` | procedural bottles (revolved mesh, cylinder-stack collision) and a hollow glass |
| `pour_scene.py` | `build(seed)`: arm, bottle, glass, random laptop camera, AprilTags on the gripper and base |
| `calib/tags.py` | AprilTag 36h11 detection and single-tag pose (OpenCV aruco) |
| `calib/handeye.py` | camera pose in the arm base frame by reprojection, optional tag-mount refinement |
| `calib/calibrate.py` | the automated session: the arm waves, tags are detected, the fit is scored and written |
| `planner/pour.py` | the pick-and-pour state machine |
| `planner/run_pour.py` | seeded episodes, GIF, `.npz`, or a LeRobot v3 dataset |

## The plan

1. **Sim first.** Bottles and glasses are procedural, the camera pose is
   random, and every episode is a fresh compile. Done; below.
2. **Calibration that runs itself.** A tag cube on the gripper, one upright
   tag by the base, an arm wave, a reprojection fit with a residual gate.
   Done in sim, with ground truth to score against; the real arm needs a
   `Robot` with three methods (`q`, `move`, `image`) on top of the CAN driver.
3. **Scripted planner as the demonstrator.** No teleop, so every
   demonstration comes from the planner: in sim without limit, on the real arm
   with the same code and camera perception. Placing the bottle somewhere new
   at the end of each episode is the reset, so a batch runs unattended until
   the bottle is empty. Done in sim; real perception is the next step.
4. **Data in one schema.** Sim and real episodes both go into LeRobot v3 with
   the features `record_a1x.py` uses, plus the session's camera intrinsics and
   extrinsics per frame, so viewpoint is never lost. Done.
5. **Model.** Train ACT or a diffusion policy with `lerobot` on the sim set,
   evaluate it in this harness, then add real autonomous episodes and
   co-train; a VLA fine-tune once the real count reaches the hundreds. An
   LLM/VLM can guide and judge real episodes where the sim's ground truth is
   not available (the verify stages are the hook). Not started.

## Scene

A bottle is a surface of revolution: body radius 30 to 42 mm, body 120 to
200 mm, a shoulder, a 13 to 18 mm neck 40 to 80 mm long, a lip. The visual is
the revolved mesh plus a label band; collision is a stack of cylinders (body,
three shoulder steps, neck) so the neck stays graspable. Mass 0.3 to 1.0 kg,
all in the body. The glass is a bottom disk and twelve thin wall boxes, 20 to
30 mm radius, free to be knocked over.

The laptop camera is drawn 0.5 to 0.85 m from the workspace at any bearing in
the front half plane, 12 to 38 cm above the table, looking at the workspace
with jitter. `info["cam"]` holds `K` and `T_cam2base` so calibration can be
scored; the planner never reads them.

Two things in the scene are not in the URDF and matter:

* **Gripper force.** The URDF gives no gripper force. A position servo
  commanded to "closed" squeezes with `kp` times the travel left, which on a
  30 mm neck at `kp = 1000` is 13 N, and a full bottle pivots straight out. A
  real gripper closes with a set force whatever the opening, so the closed
  command sits 30 mm past the stop (`GRIP_CMD`) at `kp = 1500`: about 50 N on
  a neck, 90 N on a body, capped by the actuator's force range.
* **Finger pads are grids of boxes.** MuJoCo's convex collider returns one
  contact point per box-cylinder pair, so a plate pressed flat on a bottle is
  a single point and the bottle can pivot freely about the line between the
  fingers. Each plate is split 3 by 2 and each tip in 2 in the pour scene
  only; `a1x.xml` is unchanged.

## Calibration

`calib/calibrate.py` visits random poses over the table, each with a wrist
roll so the tag cube shows a different face, until 14 tag observations from
at least 4 poses are in hand (28 from 8 when tag mounts are being refined).
Forward kinematics comes from `a1x.xml` at the *measured* joint angles, so
servo error is in the fit. The solver minimises reprojection error over every
corner of every tag in every image with Levenberg-Marquardt, and reports the
residual in pixels; a session above 1.5 px is written as untrusted.

Against ground truth over 20 random camera placements, with exact tag mounts:

| | median | max |
| --- | --- | --- |
| translation | 0.4 mm | 4.3 mm |
| rotation | 0.07 deg | 0.39 deg |
| residual | 0.3 px | 0.55 px |

Tag mounts measured with a ruler will be off by a few millimetres. With 4 mm
of noise on every gripper tag mount and no refinement the error is 7 to
22 mm, and the residual (1 to 8 px) flags most of those sessions as
untrusted, which is the point of the gate. With `--refine` the mounts are
solved too and the median comes back to 0.4 mm; the remaining outliers are
sessions where the cap on poses was hit before the gripper had shown enough
orientations, which is why refinement asks for twice the views.

On the real arm: print the four gripper tags and the base tag at 45 mm
plate size (`assets/tags/`), glue the cube on top of the gripper body,
measure the mounts, implement `Robot` on the CAN driver, and run the same
script. Intrinsics come once from a checkerboard.

## Planner

    perceive -> plan_grasp -> pick -> verify_grasp -> lift -> verify_lift
             -> plan_pour -> pour -> verify_pour -> upright -> place -> verify_place

Every waypoint goes through `Motion.solve` with the bottle rigidly attached,
so a swing that drags its base through the table or the glass is rejected
before it is tried. What the episodes taught, in the order it was learnt:

* **A level side grasp is out of reach.** The wrist cannot hold a horizontal
  tool at bottle heights anywhere on the table; an approach pitched 40 to 60
  degrees down reaches everything, and only one closing sign works because of
  the roll limit. So grasps are pitched, closing horizontally.
* **The gripper housing shoves wide bottles.** With a pitched approach the
  housing's top front corner reaches ahead of the TCP into a bottle held by
  its body. `plan_grasp` rejects any grasp where the housing meets the bottle
  (the collision check allows contact with the target, so this is a separate
  test), which leaves a high body grasp at a shallow pitch with the TCP behind
  the axis, or a neck grasp under the lip with the TCP past the axis so the
  tips cage it.
* **Tip the bottle about the approach axis, not the closing axis.** Rotation
  about the line between the fingers is the pendulum motion a pinch barely
  resists; a full bottle held by the neck rotated 35 degrees in the hand and
  pried the tips open. A roll about the approach is resisted by both contacts
  a radius apart. The roll angle for a wanted tilt follows from the approach
  pitch, `cos(tilt) = cos(roll) + sin^2(pitch) (1 - cos(roll))`, which is why
  the shallow pitch matters: at 52 degrees no roll passes horizontal.
* **The grasp planner looks ahead.** A grasp is only accepted if, from the
  lifted pose, a whole pour path solves with the bottle in hand.
* **The pour is closed loop on the bottle.** The path is defined on the
  bottle's frame: tip it about an axis through the mouth while carrying the
  mouth to 3 cm above the rim, keeping the base 5 cm off the table. Every two
  waypoints, and every half second of the hold, the bottle's pose in the hand
  is re-measured and the rest re-solved, so a bottle that settles in the
  fingers still ends up tilted enough with its mouth in the rim. `upright`
  takes the pivot out again before the bottle is set down.
* **Nothing may brush the bottle on the way to the pre-grasp**, and the ramp
  check samples every 0.04 rad of joint travel: a finger through the shoulder
  takes a few centimetres and ten samples over a 2 s move missed it.

`verify_pour` is a geometric proxy for liquid: the mouth inside the rim and
tilted past 89 degrees for at least 1.2 s, the glass standing and within 2 cm
of where it was, the bottle within 3 cm and 45 degrees of where the grasp put
it. A bead fluid can replace it without touching the rest of the machine.

Results on seeds 0 to 19: **18/20**, both failures at `plan_pour` (no pour
path found, so nothing was attempted). With 5 mm of perception noise on seeds
100 to 119: 14/20, five of the failures at `plan_grasp`. Those are mostly an
artefact of the test: the collision check runs against the true bottle pose
while the grasp was planned on the noisy one, so a 5 mm offset reads as the
pre-grasp ramp brushing the bottle and the grasp is refused. On the real arm
the check can only see what perception sees; the honest number there is the
one after real perception exists.

## Data

`run_pour.py --lerobot ROOT` streams successful episodes into a LeRobot v3
dataset at 20 Hz (needs `sim/.venv-lerobot`: `uv venv --python 3.12
sim/.venv-lerobot && uv pip install --python sim/.venv-lerobot/bin/python
'lerobot[dataset]' mujoco`). Features match `record_a1x.py`:

    action                       (7,)  commanded joint targets + gripper, 0 closed .. 0.05 open
    observation.state            (7,)  measured joints + finger position
    observation.images.laptop    640 x 480 from the random webcam
    observation.cam_K, cam_T     (9,) (16,)  the session's camera calibration

Failed episodes are dropped from the dataset but counted in the run's
report. A training run is then the stock lerobot command on that root.

## Known limits

No liquid, so the pour is verified geometrically. Bottles are round; flat
sided ones need a box body and a different closing rule. No wrist camera,
which is the one addition that would most help the pour. The real-arm
`Robot` and camera perception (bottle and glass from one image via the
table plane) are not written.
