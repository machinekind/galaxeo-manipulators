# Grasp student

A policy trained in simulation (`hvsr-robotics/a1x-wrist-grasp-sim`, `student.pt`) that does the
last centimetres of a top-down grasp on the A1X from its wrist camera: from a hover 6 cm above a
believed grasp point until the object is lifted. It runs as a strategy of the digital twin,
`galaxeo.grasp.strategy:Grasp` (entry point `grasp` in `lerobot_mp.strategies`), started from the
panel's **Strategie** tab; whether the arm is simulated or real is the twin's backend choice. Two
scripts at the root of this repository prepare a rig: `grasp_model.py` (the weights) and
`grasp_bench.py` (the measured values). The A1X in the twin: [TWIN.md](TWIN.md).

**Nothing has moved the real arm with this code yet.** It is tested in simulation, on a fake CAN
bus and against fixtures recorded from training.

## What it is, and its limits

- **Objects on the arm's left only.** Training put the object at joint 1 between 40 and 139 deg and
  0.22 to 0.44 m from the joint 1 axis, hover z 0.05 to 0.14 m; the strategy refuses a hover outside.
  Tape the cube spot near x = 0, y = 0.33 m.
- **The camera frame is re-projected, not cropped.** The real wrist camera gives 1920x1080 through
  a strongly distorting lens, mounted upside down; it is remapped to training's ideal pinhole
  (160x120, f = 142.5 px) with the 180 degree rotation folded in. The model card's centre crop is
  5 to 27 px off at 160x120 and is not used.
- **The wrist K and dist describe the rotated (upright) frame.** The default pair (2026-09-26, 15
  views, 0.85 px RMS) is not trusted: the dot check decides.
- **IK is training's**: one warm 6-D damped least squares step per policy tick, accepted at 1 mm and
  5 mrad, refused when a joint would jump more than 6 deg or the arm would hit the table or the
  object.
- **The command floor**: training's is TCP z 20 mm above the table, 17 mm lower than the original
  plan said. On a rig it is the taught table touch + 4 mm + the measured sag.
- **Latency**: camera + inference p95 + arm must stay under 150 ms; the policy is memoryless and
  was trained with an instant response.
- **Joint offsets must be zero.** The twin's `a1x.joint_offsets_deg` must stay all zero: the wrist
  camera was calibrated on raw joint angles, and the strategy cannot read the offsets.

## What the twin must have

The A1X package (galaxeo#21, twin#36) and the robot cameras and workspace `look` (twin#37). Into
the twin's environment (Python 3.10+, numpy 2, torch, OpenCV contrib, PyYAML):

```bash
pip install "galaxeo[grasp] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
```

Add the link's extra (`xcan` on macOS, `pcan` on Windows) as in [TWIN.md](TWIN.md). The bench
scripts run from a clone with the same environment: `pip install -e ".[arm,grasp,xcan]"`.

## The weights

```bash
python3 grasp_model.py fetch ~/models/student.pt   # HTTPS, size and sha256 checked before the file appears
python3 grasp_model.py check ~/models/student.pt   # sha256, weights-only strict load, schema, statistics
python3 grasp_model.py time ~/models/student.pt    # exit 1 over 30 ms median or 45 ms p95 on this CPU
```

sha256 `05e5aaf6e7d47126d494a90957238d3a4275f7cccdc36ccc00679f71cdcb8536`, 45 736 059 bytes: the
same file as `sim/grasp/student_v4.pt` on branch `grasp-rl`. `fetch` has no default destination
(keep the weights out of the repository) and never overwrites another file. Every load checks the
sha256 before anything is unpickled; another checkpoint needs its own `--sha256` here and
`student_sha256` in the strategy.

## The rig file

One JSON file per rig (`galaxeo.grasp.rig`, format 1) at a path you choose: the bench writes it with
`--rig PATH`, the strategy reads it through `rig_json`. Each tool re-reads the file and changes only
its own keys; unknown keys or another `version` are errors. Nothing is written without `--rig`, so
values measured on the simulated arm reach a file only when you name one.
`python3 grasp_bench.py show --rig PATH` prints the file, the teaching ages and what shadow and real
still miss (`nic - moze ruszac` when nothing).

| key | written by | needed for |
|---|---|---|
| `table.touch_tcp_z`, `.tcp`, `.joints`, `.taught_at` | `table` | shadow, real |
| `grasp.tcp`, `.yaw` (rad), `.joints`, `.taught_at`, `.object_height`, `.joint_offsets_deg` (zeros) | `point` | shadow, real |
| `latency.camera_s`, `.actuation_s`, `.infer_p95_s`, `.total_s` | `latency` | shadow, real (`total_s` at most 0.15); `camera_s` pairs frames with joints |
| `wrist.dot_error_px` | `dot --error-px` | shadow, real (at most 8 px) |
| `gripper.stroke_close_s_effort`, `.stroke_close_s_operator` | `gripper` | real; the shadow jaw's speed when present |
| `gripper.freeze_threshold` | `gripper --set-freeze` (100 = off) | real |
| `gripper.stroke_open_s_*`, `.free_effort_peak`, `.loaded_effort_peak`, `.separable`, `.suggested_threshold`, `.gap_mm` | `gripper` | nothing; they inform the freeze decision |
| `wrist.K`, `wrist.dist`, `wrist.upside_down` | you, K and dist from one calibration of the upright 1920x1080 frame | nothing (defaults above) |
| `overrides.shadow_done` | you, a note after ten clean shadow episodes | real |
| `overrides.table`, `.dot`, `.latency`, `.taught` | you, a note saying who and why | each lifts its limit |

Teaching expires after 12 hours (`teaching old`): repeat `table` and `point` every day.

## The bench, in order

Before anything moves:

- **One CAN sender.** Close the twin panel, `jog_a1x.py`, `so101_bridge.py` and any ROS driver; the
  bench listens 0.3 s and refuses with `busy` on another sender's command frame. On macOS the XCAN
  adapter admits one process only, and the camera needs a process started from Terminal.app.
- **The A1X has no brakes and does not yield.** Every tool that moves the real arm needs `--i-know`
  and an Enter; from then on a bare Enter is STOP, and so is Ctrl-C. A STOP holds the measured pose
  and never releases the motors. Pre-positioning runs at 20 deg/s: straight up first when the tool
  is low near the target, then the strategy's staged plan (elbow, wrist, shoulder and base), every
  stage collision-checked; `collision` means: bring the arm closer first (panel or `jog_a1x.py`).
- **Same config as the twin.** `--config configs/local.yaml` reads the twin's `a1x:` section, so the
  jaw runs as it will under the strategy.
- **Rehearse on the simulated arm.** `--sim` instead of `--iface` runs `gripper`, `table` and `point`
  on galaxeo.arm's MuJoCo model; `latency` and the dot capture need the real camera.

The commands use Linux's `--iface can0` (macOS: `--iface xcan`) and
`C="--config configs/local.yaml --rig rig.json"`.

**1. Gripper.** The jaw moves 99, 100, 0, 100, 0, 100 with nothing between the plates, 2 s per move,
sampled at 150 Hz or more; type `s` + Enter when you see the jaws stop. At the prompt put a 3 cm cube
between the plates and type `d` + Enter for a loaded close and open. The bench turns the contact
freeze off for this run.

```bash
python3 grasp_bench.py gripper --iface can0 --i-know $C
python3 grasp_bench.py gripper --rig rig.json --set-freeze 100          # or the suggested threshold
python3 grasp_bench.py gripper --rig rig.json --gap-mm 0:3 50:53 100:103 # ruler readings, mm
```

Gate: every move valid, sampling at least 150 Hz (else nothing is written). The close stroke is the
last moment the jaw's effort leaves its noise band (and your `s`); the backend's default stream
covers 0..100 in 1.0 s. Put the freeze value into the twin's `a1x.gripper_force` too. `UWAGA:
rozwarcie` means the jaw's zero or span is off: check `a1x.gripper_closed_pdes` and
`a1x.gripper_open_pdes`. The raw traces go next to the rig file (`<rig>_gripper_<stamp>.npz`).

**2. Table touch**, at the cube spot, before the cube is there:

```bash
python3 grasp_bench.py table --iface can0 --i-know $C --xy 0 0.33
```

The arm goes to TCP z 0.12 m above the point (`--start-z`), tool down, and opens the jaw there. Jog
with lines ended by Enter: `d MM` / `u MM` (down, up; at most 5 mm, 1 mm below TCP z 0.06 m),
`x MM` / `y MM` (at most 5 mm), `r DEG` (at most 5), `g N` (jaw 0..100), `ok` (store), `q` (quit
without saving); an empty line is STOP. Each step is a guarded move of galaxeo.arm. The table is
called by the measured height (two 1 mm steps that dropped less than 0.4 mm), the joint lag, or the
effort guard; the arm then holds the measured pose and backs off 2 mm. On `Kontakt prawdopodobny`:
`d 1` twice, `ok`. Gate: the touch within 15 mm of 33.9 mm, else `Nie zapisano` (`--force` stores it
with `overrides.table`).

**3. Grasp point.** Cube on the spot, one face across the closing axis:

```bash
python3 grasp_bench.py point --iface can0 --i-know $C --xy 0 0.33 --yaw-deg 0 --object-height 0.030
```

Jog until the open plates straddle the cube and the fingertips are just above the table; `r DEG`
squares the plates to the cube. `ok` prints the believed grasp (taught xy, z = touch + 6 mm) and
whether its hover lies inside the training envelope (the strategy refuses it outside). The table
touch must be within 10 cm (xy) of this point (`table touch far`).

**4. Latency.** The wrist camera through OpenCV (an index on macOS, `/dev/v4l/by-id/...` on Linux;
`--camera`, or `wrist.source` in the rig file), delivering 1920x1080. The model is timed on this
machine with the camera running.

```bash
python3 grasp_bench.py latency --iface can0 --i-know $C --student ~/models/student.pt --camera 2
```

The arm goes to the strategy's hover over the taught point, opens the jaw and steps joint 6 by 3
degrees five times through the `a1x` backend, the path the strategy's commands take. Gate:
`total_s` at most 150 ms (else `latency high`), at least three good steps (else nothing is written).

**5. Dot capture.** The arm goes to the hover, opens the jaw and takes one wrist frame; the bench
re-projects it with the rig's K and dist and draws it as the strategy's dumps do:

```bash
python3 grasp_bench.py dot --iface can0 --i-know $C --camera 2 --out dot.png
```

In `dot.png` (the model input, four times enlarged) the **small square** is where the camera model
puts the centre of the cube's top face; the cross with its 8 px circle is the believed grasp, about
1 cm above it, so it sits a few pixels off even with a perfect camera. Measure the square against
the real top-face centre in pixels of the 160x120 frame (PNG pixels divided by 4).

**6. Dot record.**

```bash
python3 grasp_bench.py dot --rig rig.json --error-px 5.5
```

Gate: at most 8 px (else `dot off`: recalibrate the wrist camera). After this `show` lists nothing
for shadow. The strategy's dumps (`dump_dir`) show the same drawing on every run: check it again
there.

## Running in the twin

Panel, tab **Strategie**, strategy `grasp`. The form is filled from the config section `grasp:`; an
unknown key or a wrong type is refused before the arm moves.

| setting | default | meaning |
|---|---|---|
| `student_pt` | required | path of the checkpoint |
| `student_sha256` | the published hash | the sha256 the file must have |
| `rig_json` | `""` | the rig file; empty = simulation defaults, allowed only on a simulated arm |
| `camera` | `wrist` | the frame the policy sees, `Observation.frames[camera]` |
| `object_name` | `cube` | the believed grasp: `Observation.objects[object_name]` when the twin reports it, else the rig file's taught point |
| `shadow` | `False` | the real jaw stays open, the floor stays above the object, never a success |
| `dump_dir` | `""` | where `ticks.jsonl`, `summary.json` and the image dumps go; empty = nowhere |
| `dump_every` | `0` | an image dump every N policy ticks; 0 = none; more than 0 needs `dump_dir` |

```yaml
grasp:
  student_pt: /home/op/models/student.pt
  rig_json: /home/op/rig/rig.json
  shadow: true
  dump_dir: /home/op/rig/run1
  dump_every: 20
```

In this order: **simulation** (arm `sim`, the training look below; the panel does not pass objects
to strategies yet, so give a sim rig file with a taught point, `grasp_bench.py point --sim --rig
sim.json ...`, and put the cube at that point in the twin's scene; the 50-episode evaluation with
objects from the scene is OpenProject #163), **shadow** (ten episodes on the real arm with the rig file and a camera recording;
no automatic stop, the fingertips at least 5 mm above the cube's top, the tool descending and yawing
towards it; then write `"shadow_done": "10 clean episodes <date>, <initials>"` in `overrides`),
**real** (three attempts, the freeze decision equal to the running `a1x.gripper_force`, teaching
younger than 12 hours, a hand on STOP).

**Refused at the start**, before anything moves (Polish message in the panel): an arm that is not
the A1X or has no gripper; a supervisor limiting an arm joint below 90 deg/s; a galaxeo.arm model
whose joint ranges are not training's; a real arm without `rig_json`; a rig file that is malformed
or misses what the mode needs (the `show` list); a checkpoint that is missing, has another sha256 or
another schema; no frame from `camera`, or one that is not RGB uint8, not 160x120 on a simulated arm
or not 1920x1080 on a real one; a wrist calibration that does not fit; no table touch; no believed
grasp (no object and no taught point); on a real arm an object reported more than 1 s ago or further
than 10 cm from the table touch point (the touch height was measured there); no hover, a hover
outside the training envelope, or a hover where the open jaw hits the object box.

**A run**: straight up when the tool is low near the hover, the staged plan to the hover at no more than 20 deg/s
(each stage collision-checked with the object box), the jaw opened there (99, then 100), until the
arm stands still at the hover, 0.3 s of sag measurement, then the policy on a 50 ms grid for at most
120 ticks or 12 s. The strategy is called at 100 Hz; the ticks between send nothing.

**Endings.** `success`, `lifted`, `tick cap` and `time cap` keep the last episode target, jaw included
(the jaw may hold the object); every other ending holds the arm and the jaw where they measure.

| ending | when |
|---|---|
| `success` | the object, reported by the twin at the start, risen more than 4 cm with the jaw commanded closed; the only ending that counts as a success |
| `lifted` | when the twin does not report the object: the closed jaw risen 4 cm above its lowest point; whether the object came along is unknown |
| `tick cap`, `time cap` | 120 policy ticks, 12 s; the normal ending in shadow |
| `impact lag <joint>` | a joint more than 4 deg (pre-positioning) or 3 deg (episode) behind the supervisor's command for 0.15 s |
| `table` | the measured tool (the lower of the TCP and the lowest fingertip) under the table touch + 0.5 mm on two ticks |
| `camera` | a frame older than 0.5 s, a frame that cannot be re-projected, or five ticks in a row without a fresh frame |
| `rate limited` | ten policy ticks in a row where the supervisor's command trails the previous target: a speed limit below the policy's 90 deg/s (panel slider, `velocity_scale`) |
| `pairing`, `lag` | five ticks in a row without a joint sample within 60 ms of the exposure; ten ticks with the arm more than 8 deg behind |
| `ik`, `collision`, `not arrived`, `j5 offset` | pre-positioning: a step up refused near the object, no clear staged plan, the hover not reached in 3 s, joint 5 more than 1 deg off zero at the hover (real arm) |
| `non-finite`, `error` | a non-finite observation, action or target; an exception in a tick |
| `stopped` | the twin ended the run: STOP, Home, preemption, its time limit |

**Shadow** keeps the real jaw at 100, raises the command floor to the highest of the bench floor, the
fingertips 10 mm above the object's top and the TCP 20 mm above the believed point, and never reports
a success. The jaw gap the policy reads
comes from a virtual jaw that follows the policy's own command at the close stroke measured by
`grasp_bench.py gripper`, else at the contract's 76.9 units/s (the old 1.3 s stream).

**`dump_dir`** gets `ticks.jsonl` (one line per policy tick: action, believed pixel, commanded TCP,
yaw and jaw, IK result, floor, sag, measured TCP, gap, lag, rate limiting, frame age, pairing gap,
inference time), `summary.json` at the end (ending, counts of ticks, skips, attempts, IK and
collision refusals, box and floor hits, lowest TCP, floors, sag, inference median and p95) and,
every `dump_every` policy ticks, `NNNN.png` with the dot drawing and `NNNN_heat.png`.

### The wrist camera

- **Simulation**: the robot camera `wrist` in `galaxeo/robot/a1x/robot.yaml`, on `gripper_link` at
  training's pose, 45.67 deg vertical field of view, rendered at 160x120: used as it is.
- **Real arm**: a workspace camera named `wrist` (the panel names new cameras `kameraN`; rename it in
  `twin.json`) delivering the raw 1920x1080 frame, re-projected with the rig file's K and dist.

### The scene look

The simulated policy is sensitive to how the scene looks, not only to its geometry: with the twin's
own table colour and lights it grasped 34 of 50 cubes, with training's 46 of 50. Give the workspace
training's look in `twin.json` (keep your other `table` keys; a light's `pos` counts from the table
top's centre):

```json
"table": {"rgba": [0.55, 0.45, 0.35, 1.0]},
"look": {
  "lights": [
    {"pos": [0.0, 0.0, 1.8], "dir": [0.0, 0.0, -1.0], "directional": true},
    {"pos": [1.0, -1.0, 1.3], "dir": [-0.4, 0.4, -1.0], "diffuse": 0.5}
  ],
  "headlight": {"ambient": 0.3, "diffuse": 0.6, "specular": 0.0}
}
```

## Known open issues

- **The real jaw is too slow for this policy.** Training set the finger target directly each tick
  and its jaw closed in about 0.23 s; the A1X stream needs 1.0 s (current default) to 1.3 s. In
  simulation with the jaw slowed to 1.3 s the policy grasped 0 of 50 (46 of 50 at the simulated
  speed). Retraining with a slewed jaw and a lagging gap observation is due before the first real
  grasp.
- **Corner walk**: in about one simulated run in five a fingertip dips 1 to 2 mm at a corner of the
  command box.
- **IK refusals**: about 10 % of simulated ticks are refused by the 6 deg joint jump rule, steps
  training took.
- **No success detection on the real arm** (`lifted` only), and a table moved since teaching goes
  unnoticed.

## Tests

```bash
python -m pytest tests/test_grasp_*.py tests/test_robot_a1x_gripper_state.py
```

Fakes only: the A1X on a fake CAN bus with a virtual clock, galaxeo.arm's MuJoCo model, scripted
keys and a fake wrist camera; no CAN link, camera or network. Tests that need the published weights
take their path from `GALAXEO_GRASP_STUDENT` and skip without it.
