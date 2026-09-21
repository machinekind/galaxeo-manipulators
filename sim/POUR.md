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
sim/.venv/bin/python sim/gen_dataset.py --root sim/datasets/pour_sim --episodes 300   # dataset, parallel
sim/.venv-lerobot/bin/python sim/planner/eval_policy.py --ckpt sim/runs/pour_act_0 --n 20  # policy in the loop
sim/.venv/bin/python sim/gen_dagger.py --ckpt sim/runs/pour_act_td2 --root sim/datasets/pour_dagger1 --seeds 400   # DAgger round
```

| | |
| --- | --- |
| `bottle.py` | procedural bottles (revolved mesh, cylinder-stack collision) and a hollow glass |
| `pour_scene.py` | `build(seed)`: arm, bottle, glass, random laptop camera, no markers; `build(seed, calib_card=True)` adds the calibration card |
| `calib/tags.py` | AprilTag 36h11 detection and single-tag pose (OpenCV aruco) |
| `calib/handeye.py` | camera pose in the arm base frame and the card's pose in the gripper, jointly, by reprojection |
| `calib/calibrate.py` | the automated session: the arm waves the card, tags are detected, the fit is scored and written |
| `planner/pour.py` | the pick-and-pour state machine |
| `planner/run_pour.py` | seeded episodes, GIF, `.npz`, or a LeRobot v3 dataset |
| `gen_dataset.py` | N worker processes, disjoint seeds, parts merged into one dataset |
| `planner/eval_policy.py` | a trained lerobot checkpoint driving the scene, judged like the planner |
| `planner/dagger.py` | DAgger worker: policy rollout, ground-truth monitor, rewind, planner takeover, recorded |
| `gen_dagger.py` | N DAgger workers on fresh seeds, corrections merged with the base dataset |
| `jobs/` | GPU payloads: `preflight.sh`, `train_pour.sh`; `pyproject.toml` + `uv.lock` is the training env |

## The plan

1. **Sim first.** Bottles and glasses are procedural, the camera pose is
   random, and every episode is a fresh compile. Done; below.
2. **Calibration that runs itself.** One printed card pinched in the gripper,
   an arm wave, a reprojection fit that solves the camera pose and the card's
   pose in the hand together, with a residual gate and a conditioning gate.
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
5. **Model.** Train ACT with `lerobot` on the sim set, evaluate it in this
   harness, then add real autonomous episodes and co-train; a VLA fine-tune
   once the real count reaches the hundreds. An LLM/VLM can guide and judge
   real episodes where the sim's ground truth is not available (the verify
   stages are the hook). Payloads, training env and the closed-loop
   evaluation exist; see Training below.

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

The arm will be set up on different tables, so nothing about one table may
be learnt. Per seed the scene also draws the table's extents and where the
arm sits on it (10 to 40 cm from each edge, the top always at the mount),
table and floor materials (flat colour or a builtin checker, gradient or flat
texture with random colours and repeat), the skybox and haze, one to three
lights of random kind, position and strength within a budget that keeps a
calibration tag decodable, and zero to two distractor primitives on the table,
kept 12 cm clear of the bottle-to-glass corridor. Bottles get one or two label
bands and are opaque one time in three. Appearance comes from a second random
stream, so a seed's task (bottle, glass, placement, camera pose) is the same
as before the randomisation was added; the planner still scores 18/20 on
seeds 0 to 19.

**The default scene has no markers in it.** `build(seed)` is arm, bottle,
glass, table and clutter, so every recorded frame is marker free. The one
fiducial in the system, the calibration card, appears only for
`build(seed, calib_card=True)`, which is what `calib/calibrate.py` asks for.
There used to be a 40 mm AprilTag cube glued on the gripper and an upright tag
plate beside the mount; both are gone. They had `contype = conaffinity = 0` and
zero mass, so they never touched the task: the planner scores the same 18/20 on
seeds 0 to 19 with the same two failures (seeds 6 and 9, no reachable pour
path) as it did with them in the scene. Checkpoints trained before this change
saw the cube in their frames, which is accepted.

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

The fiducial is **one card, pinched in the gripper, removed afterwards**. It is
a 90 x 130 x 2 mm rectangle held at one end between the finger pads, so its
normal is the closing axis and it sticks out 90 mm past the fingertips along
the approach. The pinched end carries nothing; the protruding end carries one
AprilTag 36h11 per face, 70 mm side, ids 0 and 1 back to back. Two faces
because the laptop can be on either side of the arm and a wrist cannot always
roll a single face round to it; their relative pose is exact printed geometry
(same centre in the card plane, opposite normals, the card thickness apart), so
the pair costs the solver no parameters.

A human puts the card in, so **its pose in the gripper is unknown** and is
solved for. The scene draws a misplacement per seed, up to 12 mm on each card
axis and 10 degrees about each, and never shows it to the solver, which sees
only the nominal ("square to the hand, sticking `CARD['out']` past the tips").
`handeye.solve` estimates the camera pose in the base frame and the card's pose
in the gripper *jointly*, 12 dof, by minimising the reprojection error of every
corner of every tag in every image with Levenberg-Marquardt. It starts from
whichever is better of a closed-form eye-to-hand solution and the nominal, and
restarts from jittered starts if the residual fails the gate. The closed form
is Park and Martin's AX = XB written out in `handeye.solve_axxb`: OpenCV 5
dropped `calibrateHandEye`. Forward kinematics comes from `a1x.xml` at the
*measured* joint angles, so servo error is in the fit.

The wave has two phases. A **search** phase draws poses over the table with the
wrist rolled anywhere in the circle, until a tag is seen at all; that one
detection, with the nominal card, locates the lens well enough to aim. The
**collect** phase then points the card's normal at that estimate, plus or minus
0.75 rad of jitter and optionally a half turn onto the other face, so the tag
stays readable while the hand keeps turning. Diversity is enforced, not hoped
for: `handeye.rotation_spread_R` stacks the rotation vectors of every pair of
accepted orientations and reports the smallest singular value of that (3, N)
matrix, which is the rotation available about the *least* covered axis. Each
step scores a batch of candidates by how much they would raise that number and
solves IK in that order, and the wave stops only once 14 observations from at
least 8 poses are in hand *and* the spread is at least 8 degrees. The gate
repeats all three at the end, so a session that ran out of poses, or that
turned about one axis only, comes back `trusted = False` with a reason instead
of a confident wrong pose. Moves are also capped at 1.2 rad of travel and timed
to a peak joint speed of 0.8 rad/s: the servos trail the commanded ramp by
0.05 s of travel, and a 2 rad swing in 1.5 s trails far enough to put the card
through a bottle the collision check had cleared.

The bottle and glass **are** on the table during calibration, and the wave
plans around them through the ordinary collision check (the card has a padded
collider for clearance); nothing in the fit reads them. `--empty-table` runs
the same sessions with the table cleared, and the numbers are the same, so the
calibration does not depend on what is standing there.

Against ground truth over 20 random camera placements at 640 x 480, each with a
fresh random card misplacement, all 20 trusted:

| | median | worst |
| --- | --- | --- |
| camera translation | 0.21 mm | 0.99 mm |
| camera rotation | 0.06 deg | 0.12 deg |
| residual | 0.11 px | 0.65 px |
| card pose (diagnostic) | 0.05 mm, 0.06 deg | 0.26 mm, 0.40 deg |

Robustness, same 20 seeds: with the card at the limits of the misplacement
(`--card-extreme`) 20/20 and 0.18 mm median, 0.52 mm worst. With 0.5 px of
Gaussian noise on every detected corner (`--corner-noise 0.5`) 20/20 and
1.00 mm median, 2.08 mm worst, rotation 0.25 deg worst, residual 0.68 px
median: pixel noise is what the accuracy is limited by, and the residual moves
with it, which is what makes the gate meaningful. With the table cleared,
20/20 and 0.22 mm median, 0.81 mm worst.

### On the real arm

1. Print one sheet with the two tags at **70 mm marker side** (the images in
   `assets/tags/` already carry a one-cell white quiet zone; keep it).
2. Fold it over a piece of stiff cardboard so the two tags end up back to back,
   about 90 x 130 mm with roughly 90 mm of tagged card past one end.
3. **Measure the printed marker side with a ruler and pass that number in.**
   Printers scale, and the tag size sets the whole solution's scale; this is
   the one thing the sim cannot check for you.
4. Pinch the card in the gripper at the untagged end, anywhere roughly square
   to the hand. Its exact pose does not matter and is not measured.
5. Implement `Robot` (`q`, `move`, `image`) on the CAN driver and run
   `calibrate.py`. Intrinsics come once from a checkerboard.
6. Take the card out. Nothing else in the system has a marker on it.

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
dataset at 20 Hz. `sim/.venv-lerobot` is that environment: `uv venv --python
3.12 sim/.venv-lerobot && uv pip install --python sim/.venv-lerobot/bin/python
'lerobot[dataset,training]' mujoco`, or sync it from `sim/pyproject.toml`.
Features match `record_a1x.py`:

    action                        (7,)  commanded joint targets + gripper, 0 closed .. 0.05 open
    observation.state             (7,)  measured joints + finger position
    observation.images.laptop     320 x 240, downscaled from the 640 x 480 render
    observation.images.topdown    256 x 256, that frame warped onto the table plane
    observation.cam_K, cam_T      (9,) (16,)  the session's camera calibration, for
                                  provenance (`cam_K` is the 640 x 480 intrinsics)

Instead of conditioning the policy on the calibration vector, the calibration
is spent redrawing each frame as a top-down map of the table
(`planner/topdown.py`): a 0.7 m square about the workspace, world +y up, so an
object standing on the table lands at its true (x, y) whatever the camera
angle. Anything above the plane smears away from the camera, consistently per
pose. The first dataset (`marcinwysocki/a1x_pour_sim`) predates this: it
stored the raw 640 x 480 frame and a 25-vector `observation.environment_state`
(K and T concatenated, the one non-image state key ACT reads) in place of the
map. Failed episodes are dropped from the dataset but counted in the run's
report.

Each generated set also lives on the Hub as a private dataset
(`marcinwysocki/a1x_pour_sim`, `marcinwysocki/a1x_pour_sim_td`), which is
where a training box fetches it from (`DATASET_REPO`); shipping the 1.9 GB
archive from a laptop took hours per rental.

`gen_dataset.py` runs N `run_pour.py` processes on disjoint seed ranges, each
into its own part, and merges the parts with `lerobot-edit-dataset`. Eight
workers make about 300 successful episodes in under an hour on a laptop, at
roughly 6.5 MB and 750 frames per episode. The first full run, seeds 1000 to 1374 on eight workers: 330 successful episodes out of 376 seeds, 247,100 frames, 1.9 GB, 57 minutes.

## Training

`sim/jobs/` holds two payloads for an out-of-tree GPU dispatcher (plain bash,
parameters from environment variables, no machine names; the contract is in
`sim/jobs/README.md`). `preflight.sh` proves a node can train: CUDA, imports,
video decoding through PyAV (torchcodec would need a system ffmpeg), the
ResNet18 backbone download, and five real training steps. `train_pour.sh` is
one ACT run: chunk 50, batch 32, lr 1e-5, photometric image augmentation only
(the default set includes an affine jitter, which would break the image's
relation to the calibration the frames carry), checkpoints ten times, resumes
from the last one. `sim/pyproject.toml` with its lock is the environment the
node syncs.

Training does not evaluate. `planner/eval_policy.py` loads a checkpoint, feeds
it exactly the features its config lists at the dataset's rate, writes the
seven actions to the servos with the recorder's gripper mapping inverted, and
judges from ground truth the way the planner is judged: mouth in the rim past
the tilt threshold for 1.2 s cumulative, glass standing, bottle not on the
floor.

### First run

ACT, chunk 50, batch 32, lr 1e-5, 40,000 steps on a rented RTX 4090 (about
2 h 20 min at 5 steps/s, 0.45 USD/h). The L1 action loss fell from 0.73 at
step 100 to 0.117 at 40,000 and was still falling. Closed loop, the
checkpoint fails: **0/20** on unseen seeds 2000 to 2019, 0/20 with temporal
ensembling, and **1/20** on seeds from the training range, where seed 1002
is a clean ten-second pour. The recording shows the failure mode: the arm
reaches the bottle and sweeps it over on the approach, then carries on with
an empty hand.

Two checks separate harness from policy. Replaying a recorded episode's
actions open loop through the same actuator path pours successfully, so the
pathway is right. The policy's own first-step predictions on recorded frames
are off by 0.7 to 3.5 degrees per joint, worse than simply holding the current
state, and a couple of degrees at half a metre of reach is a couple of
centimetres at the gripper, which is enough to knock a bottle over. That is
underfitting, not memorisation. The second run resumed the same checkpoint to
100,000 steps (another 2 h 45 min on a 4090, dataset fetched from the Hub in
minutes). The L1 loss reached 0.101 and was flat from step 80,000 on. The
result is the same regime: **0/20** unseen and **1/20** training-range
(seed 1010, an eight-second pour), with more tilts past horizontal that miss
the glass. More gradient steps do not buy the missing precision, so the next
levers are structural: drop the per-frame calibration input, a larger action
chunk, more episodes, a lower-resolution or cropped image, or a diffusion
policy head.

### The redraw run

The random camera per episode was the leading suspect: with 330 episodes the
network has to learn 3D localisation from every viewpoint at once. The
calibration is known per session, so the third run spends it before the
network instead of beside it. `planner/topdown.py` warps each frame onto the
table plane, a 0.7 m square about the workspace at 2.7 mm per pixel, world +y
up; an object standing on the table lands at its true (x, y) in that map
whatever the camera angle, and anything above the plane smears away from the
camera consistently per pose. The policy sees the map at 256 x 256 next to the
raw frame at 320 x 240; the calibration vector is dropped. The dataset was
regenerated with the same generator (363 episodes, seeds 1000 to 1415, 271,792
frames, 1.4 GB, private Hub copy `marcinwysocki/a1x_pour_sim_td`).

Training was 80,000 steps in two rentals: the first box was stopped by its host
at step 40,000 (the GPU was re-rented under the run), and since only the
weights travel, not the optimizer state, the second rental warm-started from
the step-36,000 weights for the remaining 44,000 steps with a fresh optimizer
(`INIT_FROM`, see `sim/jobs/README.md`). The L1 loss reached 0.099, just under
the first policy's floor of 0.101, and was still drifting down.

Closed loop, the final checkpoint scores **0/20** on unseen seeds 2000 to 2019
and **3/20** on training-range seeds 1000 to 1019 (seeds 1000, 1007 and 1010;
14.9 s, 12.8 s and 4.1 s of pour), against 0/20 and 1/20 before. The failure
mix has moved, though. Of the 20 unseen episodes, five never touch the bottle,
five nudge it, six sweep it flat as the first policy did, and four grasp it,
lift it, tip it and hold it beside or above the glass without the mouth ever
crossing the rim; one of those drops it off the table. On the training seeds
seven of the 17 failures are that lifted-and-tipped kind, and twice the glass
is knocked over. So the map buys the policy the grasp on a good fraction of
scenes, but not the last few centimetres that put the mouth in a 6 cm glass.

That failure is exactly what a teacher can correct: the planner knows the
right joint targets from any state the policy reaches, so the next run should
be a DAgger loop, rolling the policy out, relabelling its visited states with
the planner's actions, and retraining on the union. A wrist camera is the
lever after that; the photorealism step waits until a sim policy pours in sim.

### DAgger

The planner is an open-loop timed script, so relabelling the policy's own
frames is not an option: asked what it would do from any state, its first
command is the start of a smoothstep ramp, which says "stay here", and ACT
learns 50-step chunks, not single steps. What the simulator offers instead is
rewinding. `planner/dagger.py` rolls the policy out on a fresh scene and
snapshots the physics every 0.5 s. A ground-truth monitor ends the rollout at
the first trouble and names it:

| trouble | |
| --- | --- |
| `disturbed` | the bottle moved 10 mm or leaned 0.1 rad with nobody holding it |
| `glass` | the glass moved 10 mm or fell |
| `dropped` | the bottle was carried and is no longer in the fingers (the policy opens the hand early on some scenes) |
| `miss` | held, tipped past 1 rad, mouth outside the rim for 1.5 s: pouring on the table |
| `timeout` | 45 s without a pour |

"Held" is both finger bodies in contact with the bottle, debounced by 0.25 s,
because a bottle rolling in the fingers drops a contact for a tick now and
then. The scene is then restored to 1 s before the trouble (then 2.5 s, then
5 s, if the teacher cannot use the nearer one; a timeout rewinds to a random
snapshot) and `Pour.takeover` finishes the episode from there, recorded in the
dataset's schema as one episode. The policy's own frames are not kept. What
training gains is recoveries from the states the policy's mistakes lead to.

`takeover` has two entries. With the bottle free and upright it is the normal
stage list, replanned from wherever the arm is. With the bottle in the fingers
it freezes the grasp as measured, lifts first if the bottle is still near the
table, and plans a pour from the bottle's current orientation: if it is
already tipped, the roll about the approach axis is continued by an angle found
by scanning (the closed form assumes an upright start), or the mouth is simply
carried over the rim if the tilt is already past the pour angle. Afterwards
the bottle is levelled in steps, raising the hand where turning a tipped bottle
about the TCP would put its base through the table. A segment is kept whole if
the teacher's run succeeded, cut at the end of `verify_pour` if only the
putting-down failed, and dropped otherwise. The nominal planner is unchanged:
seeds 0 to 19 print the same lines as before.

`gen_dagger.py` runs the workers in parallel on seeds from 5000 (2000 to 2019
are refused: they are the evaluation set), writes `ROOT/dagger` (corrections
only), `ROOT/merged` (the base set plus the corrections, which is what training
reads) and `ROOT/dagger_log.jsonl` (seed, trouble, takeover time, entry,
frames per episode). A round is then:

```bash
sim/.venv/bin/python sim/gen_dagger.py --ckpt sim/runs/pour_act_td2 --root sim/datasets/pour_dagger1 --seeds 400
# push sim/datasets/pour_dagger1/merged to a private Hub dataset, then train_pour.sh with
#   DATASET_ROOT=sim/datasets/pour_dagger1/merged DATASET_REPO=<that repo> INIT_FROM=<the rolled-out weights>
sim/.venv-lerobot/bin/python sim/planner/eval_policy.py --ckpt sim/runs/<new run> --n 20 --seed 2000
```

and the next round rolls out the new checkpoint with the previous `merged` as
`--base`.

## Known limits

No liquid, so the pour is verified geometrically. Bottles are round; flat
sided ones need a box body and a different closing rule. No wrist camera,
which is the one addition that would most help the pour. The real-arm
`Robot` and camera perception (bottle and glass from one image via the
table plane) are not written.
