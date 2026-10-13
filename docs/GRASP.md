# Grasp student

A policy trained in simulation (`hvsr-robotics/a1x-wrist-grasp-sim`, `student.pt`) that does the
last centimetres of a top-down grasp on the A1X from its wrist camera: from a hover 6 cm above a
believed grasp point until the object is lifted, 20 ticks a second. It runs as a strategy of the
digital twin: `galaxeo.grasp.strategy:Grasp`, entry point `grasp` in `lerobot_mp.strategies`,
started from the panel's **Strategie** tab. Whether the arm is simulated or real is the twin's
backend choice. Two scripts at the root of this repository prepare a rig: `grasp_model.py` (the
weights) and `grasp_bench.py` (the measured values). For the A1X in the twin see [TWIN.md](TWIN.md).

**Nothing has moved the real arm with this code yet.** It is tested in simulation, on a fake CAN
bus and against fixtures recorded from training.

## What it is, and its limits

- **Objects on the arm's left only.** Training put the object at joint 1 between 40 and 139 deg and
  0.22 to 0.44 m from the joint 1 axis; a hover outside that is out of distribution. Tape the cube
  spot near x = 0, y = 0.33 m.
- **The camera frame is re-projected, not cropped.** The real wrist camera gives a 1920x1080 frame
  through a strongly distorting lens, mounted upside down. It is remapped to training's ideal
  pinhole (160x120, f = 142.5 px) with the 180 degree rotation folded in. The model card's centre
  crop is 5 to 27 px off at 160x120 and is not used.
- **The wrist K and dist describe the rotated (upright) frame.** The default pair was measured
  2026-09-26 (15 views, 0.85 px RMS) and is not trusted: the dot check decides.
- **IK is training's**: one warm 6-D damped least squares step per tick, accepted at 1 mm and
  5 mrad, refused when a joint would jump more than 6 deg.
- **The command floor** in training is TCP z 20 mm above the table top, 17 mm lower than the
  original plan said. On a rig the floor is the taught table touch + 4 mm.
- **Latency**: camera + inference p95 + arm must stay under 150 ms; the policy is memoryless and
  was trained with an instant response.

## Install next to the twin

Into the twin's environment (Python 3.10+, numpy 2, torch, OpenCV contrib):

```bash
pip install "galaxeo[grasp] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
```

Add the extra the link needs (`xcan` on macOS, `pcan` on Windows) as in [TWIN.md](TWIN.md). The
twin finds the strategy through the entry point; its settings come from the panel form, filled from
the section `grasp:` of the twin's config file. The bench scripts run from a clone of this
repository with the same environment: `pip install -e ".[arm,grasp,xcan]"`.

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

One JSON file per rig (`galaxeo.grasp.rig`, format 1) at a path you choose: the bench writes it
with `--rig PATH`, the strategy reads it through `rig_json`. Each tool re-reads the file and changes
only its own keys, so your edits survive; unknown keys or another `version` are errors. Nothing is
written without `--rig`, so values measured on the simulated arm reach a file only when you name
one. `python3 grasp_bench.py show --rig PATH` prints the file, the teaching ages and what each mode
still misses (`nic - moze ruszac` when nothing).

| key | written by | refused without it in |
|---|---|---|
| `table.touch_tcp_z`, `.tcp`, `.joints`, `.taught_at` | `table` | shadow, real |
| `grasp.tcp`, `.yaw` (rad), `.joints`, `.taught_at`, `.object_height`, `.joint_offsets_deg` | `point` | shadow, real |
| `latency.camera_s`, `.actuation_s`, `.infer_p95_s`, `.total_s` | `latency` | shadow, real (`total_s` at most 0.15) |
| `wrist.dot_error_px` | `dot --error-px` | shadow, real (at most 8 px) |
| `gripper.stroke_close_s_effort`, `.stroke_close_s_operator` | `gripper` | real |
| `gripper.freeze_threshold` | `gripper --set-freeze` (100 = off) | real |
| `gripper.stroke_open_s_*`, `.free_effort_peak`, `.loaded_effort_peak`, `.separable`, `.suggested_threshold`, `.gap_mm` | `gripper` | never; they inform the freeze decision |
| `wrist.K`, `wrist.dist`, `wrist.upside_down` | you, K and dist from one calibration of the upright 1920x1080 frame | never (defaults above) |
| `guard.effort_abs` | you, about 1.5 x the largest effort peak in motion (default 12.0) | never |
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
  and never releases the motors. Pre-positioning runs at 20 deg/s.
- **The arm starts near the work.** The bench drives one collision-checked joint-space line to its
  start pose, after 5 mm steps straight up when the tool is low near it; if the line collides it
  refuses with `collision`: bring the arm closer first (panel or `jog_a1x.py`).
- **Same config as the twin.** `--config configs/local.yaml` reads the twin's `a1x:` section, so the
  jaw runs as it will under the strategy. All `a1x.joint_offsets_deg` must be zero: the bench and
  the camera calibration work in raw wire angles.
- **Rehearse on the simulated arm.** `--sim` instead of `--iface` runs `gripper`, `table` and
  `point` on galaxeo.arm's MuJoCo model, nothing needed and nothing written without `--rig`.

The commands below use Linux's `--iface can0`; on macOS it is `--iface xcan`.

**1. Gripper.** The jaw moves 99, 100, 0, 100, 0, 100 with nothing between the plates, 2 s per move,
sampled at 150 Hz or more; type `s` + Enter when you see the jaws stop. At the prompt put a 3 cm cube
between the plates and type `d` + Enter for a loaded close and open. The bench turns the contact
freeze off for this run, so no config change is needed.

```bash
python3 grasp_bench.py gripper --iface can0 --i-know --config configs/local.yaml --rig rig.json
python3 grasp_bench.py gripper --rig rig.json --set-freeze 100          # or the suggested threshold
python3 grasp_bench.py gripper --rig rig.json --gap-mm 0:3 50:53 100:103 # ruler readings, mm
```

Gate: every move valid, the close stroke about 1.0 to 1.3 s, sampling at least 150 Hz (else nothing
is written). Put the freeze value you record into the twin's `a1x.gripper_force` as well. A
`UWAGA: rozwarcie` line means the jaw's zero or span is off: check `a1x.gripper_closed_pdes` and
`a1x.gripper_open_pdes`. The raw traces go next to the rig file (`<rig>_gripper_<stamp>.npz`).

**2. Table touch**, at the cube spot, before the cube is there:

```bash
python3 grasp_bench.py table --iface can0 --i-know --config configs/local.yaml --rig rig.json --xy 0 0.33
```

The arm goes to 0.12 m above the point (`--start-z`), tool down, and opens the jaw there. Jog with
lines ended by Enter: `d MM` / `u MM` (down, up; at most 5 mm, 1 mm below TCP z 0.06 m), `x MM` /
`y MM` (at most 5 mm), `r DEG` (at most 5), `g N` (jaw 0..100), `ok` (store), `q` (quit without
saving); an empty line is STOP. Each step is a guarded move; the table is called by the measured
height (two 1 mm steps that dropped less than 0.4 mm), the joint lag, or the effort guard, and the
arm then holds the measured pose and backs off 2 mm. On `Kontakt prawdopodobny`: `d 1` twice, `ok`.
Gate: the touch within 15 mm of 33.9 mm, else `Nie zapisano` (`--force` stores it with
`overrides.table`).

**3. Grasp point.** Cube on the spot, one face across the closing axis:

```bash
python3 grasp_bench.py point --iface can0 --i-know --config configs/local.yaml --rig rig.json \
    --xy 0 0.33 --yaw-deg 0 --object-height 0.030
```

Jog until the open plates straddle the cube and the fingertips are just above the table; `r DEG`
squares the plates to the cube. `ok` prints the believed grasp (taught xy, z = touch + 6 mm) and
whether its hover lies inside the training envelope. The table touch must be within 10 cm (xy) of
this point (`table touch far`).

**4. Latency.** The wrist camera through OpenCV (an index on macOS, `/dev/v4l/by-id/...` on Linux;
`--camera`, or `wrist.source` in the rig file); it must deliver 1920x1080. The model is timed on this
machine with the camera running.

```bash
python3 grasp_bench.py latency --iface can0 --i-know --config configs/local.yaml --rig rig.json \
    --student ~/models/student.pt --camera 2
```

The arm goes to the hover over the taught point, opens the jaw, and steps joint 6 by 3 degrees five
times through the `a1x` backend, the path the strategy's commands take. Gate: `total_s` at most
150 ms (else `latency high`); at least three good steps (else nothing is written). It also prints
the joints' effort peaks, the input for `guard.effort_abs`.

**5. Dot.** With the arm above the cube (`point --xy <taught x y> --yaw-deg <taught yaw> --start-z
0.10`, then `q`: the arm holds), get a model-input dump from the strategy (`dump_dir`, below). The **small square** is where the
camera model puts the centre of the cube's top face; the cross with its 8 px circle is the believed
grasp, about 1 cm above it, so it sits a few pixels off even with a perfect camera. Measure the
square against the real top-face centre in pixels of the 160x120 frame (the PNG divided by its
scale), repeat at `--start-z 0.08` and 2 cm off, and record the largest:

```bash
python3 grasp_bench.py dot --rig rig.json --error-px 5.5
```

Gate: at most 8 px (else `dot off`: recalibrate the wrist camera). After this `show` lists nothing
for shadow.

## Running in the twin

Panel, tab **Strategie**, strategy `grasp`; the form is filled from the config section `grasp:`
and refuses an unknown key or a wrong type before the arm moves.

| setting | default | meaning |
|---|---|---|
| `student_pt` | required | path of the checkpoint |
| `student_sha256` | the published hash | the sha256 the file must have |
| `rig_json` | empty | the rig file; empty = simulation defaults, refused on a real arm |
| `camera` | `wrist` | the frame the policy sees, from `Observation.frames` |
| `object_name` | `cube` | the believed grasp comes from `Observation.objects[object_name]` when the twin provides it, else from the rig file's taught point |
| `shadow` | | jaw held open, floor raised above the object; never a success |
| `dump_dir`, `dump_every` | | model-input dumps every N ticks with the believed pixel and the projected cube top: the dot check reads them |

```yaml
grasp:
  student_pt: /home/op/models/student.pt
  rig_json: /home/op/rig/rig.json
  shadow: true
  dump_dir: /home/op/rig/dumps
```

In this order:

1. **Simulation.** Arm `sim`, `rig_json` empty, the training look (below). The cube comes from the
   scene. A clean run descends, closes once and lifts.
2. **Shadow**, ten episodes on the real arm with a camera recording it: `shadow: true` and the rig
   file. No automatic stop, the fingertips at least 5 mm above the cube's top all the way, the tool
   descending and yawing towards the cube. Then write `"shadow_done": "10 clean episodes <date>,
   <initials>"` in `overrides`.
3. **Real**, three attempts: `shadow: false`, the freeze decision equal to the running
   `a1x.gripper_force`, teaching younger than 12 hours, a hand on STOP. A stop or a strike sends you
   back to the dot check and shadow, with the logs.

### The wrist camera

- **Simulation**: the robot camera `wrist` declared in `galaxeo/robot/a1x/robot.yaml`, mounted on
  `gripper_link` at training's pose, 45.67 deg vertical field of view, rendered at 160x120: the
  policy sees it as it is.
- **Real arm**: a workspace camera named `wrist` (the panel names new cameras `kameraN`; rename it
  in `twin.json`) delivering 1920x1080. The strategy re-projects it with the rig file's K and dist.
  Keep the camera out of other processes while the strategy runs.

### The scene look

The simulated policy is sensitive to how the scene looks, not only to its geometry: with the twin's
own table colour and lights it grasped 34 of 50 cubes, with training's 46 of 50. Give the
workspace training's look in `twin.json` (table colour in `table.rgba`):

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

Keep the other `table` keys of your workspace; `pos` of a light counts from the table top's centre.

## Known open issues

- **The real jaw is too slow for this policy.** Training's jaw closes almost at once; the A1X
  stream needs about 1.3 s for a full stroke. In simulation with the jaw slowed to 1.3 s the policy
  grasped 0 of 50 (46 of 50 at the simulated speed). Retraining with a slewed jaw and a lagging gap
  observation is due before the first real grasp.
- **Corner walk**: in about one simulated run in five a fingertip dips 1 to 2 mm at a corner of the
  command box.
- **IK refusals**: about 10 % of simulated ticks are refused by the 6 deg joint jump rule, steps
  training took.
- **The first dot check needs a dump from the strategy, and shadow refuses without a recorded dot.**
  The twin's planned observe-only mode will give that dump without moving.
- **No jaw position on the A1X**, no success detection on the real arm, and a table moved since
  teaching goes unnoticed.

## Tests

```bash
python -m pytest tests/test_grasp_*.py tests/test_robot_a1x_gripper_state.py
```

Fakes only: the A1X on a fake CAN bus with a virtual clock, galaxeo.arm's MuJoCo model, scripted
keys and a fake wrist camera; no CAN link, camera or network. Tests that need the published weights
take their path from `GALAXEO_GRASP_STUDENT` and skip without it.
