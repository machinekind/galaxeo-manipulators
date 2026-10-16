# galaxeo.robot

The Galaxea A1X as a robot package for the digital twin
([digital_twin_training](https://github.com/machinekind/digital_twin_training), package `lerobot_mp`).
It holds the robot folder `a1x/` and the hardware backend `a1x`.
The twin finds both through the entry points `lerobot_mp.robots` and `lerobot_mp.backends`.
galaxeo never imports the twin.
[a1x/README.md](a1x/README.md) describes the model.

Read [docs/SAFETY.md](../../docs/SAFETY.md) before the first power-on.

## Contents

| Path | Holds |
|---|---|
| `a1x/robot.yaml` | joints, limits, home, labels, connections, wave ranges and jaw thresholds |
| `a1x/a1x.xml` | the MuJoCo model, generated from the vendor URDF by `import_twin_model.py` |
| `a1x/demo.json` | the example workspace: a 1.2 x 0.8 m table and two simulated cameras |
| `a1x/README.md` | the model: its source, the post-processing rules and the checks |
| `__init__.py` | `ROBOT_DIR`, the path of the robot folder |
| `backend.py` | backend `a1x`: the twin's commands as CAN-FD frames, through `galaxeo.protocol` and `galaxeo.bus` |

`create(ctx)` in `backend.py` builds an `A1XArm` from the twin's `BackendContext`.
The backend reads that context by duck typing.
`A1XConfig` is its `a1x:` config section.

## Entry points

```toml
[project.entry-points."lerobot_mp.robots"]
a1x = "galaxeo.robot:ROBOT_DIR"

[project.entry-points."lerobot_mp.backends"]
a1x = "galaxeo.robot.backend:create"
```

## Install next to the twin

Install galaxeo into the twin's environment, from a clone of this repository:

```bash
pip install -e ../galaxeo-manipulators            # Linux: standard library only
pip install -e "../galaxeo-manipulators[xcan]"    # macOS: + pyusb (and brew install libusb)
pip install -e "../galaxeo-manipulators[pcan]"    # Windows: + python-can
```

Or install it straight from git:

```bash
pip install "galaxeo[xcan] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
```

When an adapter's module is missing, *Connect* names the extra to install, for example
`pip install "galaxeo[xcan]"`.

Get a workspace from the twin:

```bash
lerobot-twin demo --robot a1x    # copies a1x/demo.json to workspace/twin.json
lerobot-twin workspace           # prints the workspace: robot a1x, backend sim
```

The workspace's `robot` field picks the arm. The panel does not change it.
For the real station, delete both simulated cameras in the Cameras tab and add the USB ones.
A workspace file of just `{"robot": "a1x"}` also works. It has an empty table and no cameras.
*Connect* stores the backend and the port in the workspace.

Bring the CAN link up first.
[docs/HARDWARE.md](../../docs/HARDWARE.md) has the wiring and the timings.
On Linux, section 0 of [docs/STEERING.md](../../docs/STEERING.md#0-bus-first) brings the link up.
On macOS, [the XCAN driver](../../docs/STEERING.md#macos-the-adapter-without-a-kernel) talks to the adapter.
`connections` in `robot.yaml` gives the panel one *Galaxea A1X* entry per system, with these ports:

| System | Port field |
|---|---|
| Linux | `can0` |
| macOS | `xcan`. With two adapters, `xcan:<USB address>` |
| Windows | `PCAN_USBBUS1` |

An empty Port field means `xcan` on macOS and `can0` everywhere else.
On Windows, type `PCAN_USBBUS1`.

## The `a1x:` configuration section

The backend reads the `a1x:` section of the twin's config.
The twin's `configs/default.yaml` does not have it.
Write only that section into the twin's `configs/local.yaml`. Unset keys take the defaults below.
Start the panel with `lerobot-twin --config configs/local.yaml ui`.

```yaml
a1x:
  joint_offsets_deg: [0.0, 0.0, 1.49, 0.0, 0.0, 0.0]   # measure yours (below)
```

The section is strict.
An unknown key, or a value of the wrong type, fails *Connect* and names the key.
The twin ignores `robot.backend` and `robot.port` from the file.
The panel's Arm and Port fields pick them, and the workspace remembers them.

| Key | Default | Meaning |
|---|---|---|
| `kp` | 20.0 | gain sent in `0x050`. The arm ignores it, because only `p_des` acts. At `kp` 0 the arm stops listening, so `kp <= 0` is refused |
| `kd` | 1.0 | damping sent in `0x050` and `0x051`. It has no effect on motion |
| `gripper_kp` | 25.0 | `kp` of the gripper frame `0x051`. It must also be > 0 |
| `gripper_open_pdes` | −1.95 | gripper `p_des` [rad] at 100 (open). More negative is wider. It stays 0.04 rad before the open stop (−1.992) |
| `gripper_closed_pdes` | 0.08 | gripper `p_des` [rad] at 0 (closed): 0.04 rad of pressure past the closed stop (+0.039). Never +0.6, see [The gripper](../../docs/HARDWARE.md#the-gripper) |
| `gripper_fb_closed` / `gripper_fb_open` | 0.039 / −1.992 | group 7 position of `0x052` [rad] at 0 and at 100, the stops measured on 2026-09-27. Set both or neither, and not equal. `null` in both means open loop: the panel shows the gripper command and warns "sprzezenie niezmierzone" |
| `gripper_start` | 0.0 | the gripper value the backend reports before the first command, while the gripper is not measured. 0 is closed, as in the station's home pose |
| `gripper_rate_pdes_s` | 2.0 | highest gripper `p_des` speed [rad/s]. The full 2.03 rad stroke takes about 1.0 s |
| `gripper_force` | 1.2 | how far group 7's \|effort\| may rise above rest while closing before the gripper target freezes. Empirical unit. Crushing was seen from about 6 |
| `joint_offsets_deg` | 6 × 0.0 | arm joint zero [deg], six values. Subtracted on read, added on send |
| `stream_hz` | 200.0 | rate of the `0x050` / `0x051` stream. The arm answers `0x052` every 5 ms |
| `max_step_deg_s` | 90.0 | the backend's own `p_des` speed limit [deg/s], a second layer under the supervisor |
| `stale_ms` | 150.0 | no `0x052` for longer than this means the link is silent |
| `frozen_frames` | 50 | this many identical `0x052` frames in a row mean frozen telemetry: the arm was released (FF 2) |
| `deaf_err_deg` / `deaf_s` | 3.0 / 1.0 | a command this far from the measurement, for this long, on a standing joint (< 0.3 deg/s) means the arm is deaf to `0x050` or blocked. The backend reports a fault and the twin stops |
| `enable_on_connect` | `false` | runs FF 1 → 5 → 6 right after *Connect*. Keep it `false` and use the panel button. FF 5 briefly releases the motors |
| `gripper_over_xcan` | `true` | drives the gripper over XCAN (macOS). That works, measured on 2026-09-27. `false` refuses the gripper there. `A1X_GRIPPER_XCAN=1` forces `true` |

`stream_hz`, `max_step_deg_s` and `gripper_rate_pdes_s` must be > 0.

### Joint zero (`joint_offsets_deg`)

On the galaxeo arm J3 reads about +1.49° at rest, past its own 0° limit.
That is a zero offset. [docs/PROTOCOL.md](../../docs/PROTOCOL.md) confirms the decoding.
Measure yours:

1. Fold the arm onto its stops and power it.
2. Set `joint_offsets_deg` to all zeros and *Connect* without going home.
3. Read J2 and J3, which rest on their stops, from the sliders or from the log line `A1X na …: poza startowa …`.
4. Put the J3 value (e.g. `1.49`) third. Leave the rest at 0 unless you have a sure reference.

The backend subtracts the offset from the reading, so after a panel restart J3 reads about 0.

## What the backend does in the twin

- Nothing goes on the bus until the first move. *Connect* never drives home.
- The `p_des` stream starts from the measured pose and moves at most `max_step_deg_s`.
  It runs at `stream_hz` and repeats the last target.
- The backend clamps to the URDF limits, widened to the pose at *Connect*.
  The folded arm stands outside the twin's own limits.
  The supervisor leaves such a joint where it stands. The joint's first move leads it into range at 15°/s.
- On link loss the backend stops sending at once. Link loss is no `0x052` for `stale_ms`, or frozen telemetry.
  Arm and gripper targets are dropped and the twin stops.
  When frames return, nothing moves until the next command, and that command starts from a fresh measurement.
- It never sends FF 2, never `kp` 0 and never the `0x023` heartbeat.
  *Disconnect* ends the stream and releases nothing.
- FF 1 → 5 → 6 goes out only from the panel button **Wlacz silniki A1X (FF 1-5-6)**, or with `enable_on_connect`.
  The button shows only with a connected `a1x`, and it needs its confirmation box.
  It refuses while the panel is armed, or while a policy, the wave or an identification owns the arm.
  During the sequence the backend streams `p_des` equal to the measured pose.
  A freshly powered arm listens right away and does not need it.

The gripper:

- Group 7 of `0x052` measures the jaws only while `0x051` flows.
  The backend trusts it from 4 `0x052` frames after the first `0x051` of the connection, and again after a link loss.
  Before that it reports the last command, or `gripper_start` before any command.
  *Connect* logs the raw rest value ("A1X chwytak: grupa 7 w spoczynku …").
- The `0x051` stream starts only on a command more than 0.5 away from the reported value.
  So *Connect* closes nothing.
- *Set home = here* stores gripper 0 (closed).
  If the gripper has not been driven in this connection, the twin then assumes the jaws are closed.
  If it has, the button stores only the arm pose, and the log warns "chwytak juz sterowany".
  Close the jaws with the slider and click again.
  In `sim` the scene's gripper closes at once.
- Each close measures rest as the median \|effort\| of its first 0.1 s. The target does not freeze in that window.
  After it, the target freezes once \|effort\| exceeds rest by `gripper_force`.
  Any opening command, or one equal to the target, lifts the freeze.
  The next close measures rest again only after the jaws opened 0.3 rad of `p_des` from the tightest point of the close.
  After a smaller opening the jaws still press the object, and that pressure must not become the new rest.
- While the jaw state is only a guess (`gripper_start` or *Set home = here*), the first opening goes straight to the target.
  Closing always ramps, so a wrong guess never snaps the jaws shut.

## Units

| | App | Twin (MJCF) | Arm (CAN) |
|---|---|---|---|
| arm joints | degrees | radians, the same angles | radians + `joint_offsets_deg` |
| gripper | 0..100 (0 = closed) | jaw travel 0..0.05 m (0 = plates together, tips 3 mm apart). The second jaw mirrors the first | `p_des` [rad], linear from +0.08 (0) to −1.95 (100). Group 7 of `0x052` [rad]: +0.039 (0) to −1.992 (100) |

The A1X gripper has no servo ticks, so the SO-101's tick settings do not apply.
The jaw thresholds (`jaw` in `robot.yaml`) are in metres.
20 mm of travel is 43 mm between the tips, more than the 42 mm diagonal of a 3 cm cube.

## First session through the twin

Before you start:

- The arm is folded and resting, and the space under it is clear.
- The supply's emergency switch and the panel's STOP are at hand.
- Nothing else sends on the bus.
- The link is checked. Linux: `ip -det -s link show can0`.
  macOS: `python -m galaxeo.xcan_usb --normal`, then close it. One process at a time holds the adapter.

### 0. Connect

Run `lerobot-twin --config configs/local.yaml ui`.
In the Arm tab, pick the *Galaxea A1X* entry for your system and press *Connect*.
The sliders show the measured pose.
*Connect* fails when:

- no `0x052` frame arrives within 3 s. The link is down or the arm has no power.
- the telemetry is frozen. The arm was released (FF 2). Support it and power-cycle it.
  FF 1-5-6 does not fix this, because a released arm's pose cannot be read.

The folded arm usually stands outside the twin's limits.
J2 reads about −0.2° and J3 about +1.5°, against limits of 3…177° and −187…−3°.
That is expected. The first move of the J2 or J3 slider leads the joint into range.

### 1. A small move

Press ARM, then move one slider (e.g. J1 Podstawa) by a few degrees.
Check the direction, and check that the sliders follow the arm.
If the arm reports but does not move, the panel shows the error "… ramie gluche na 0x050 … albo zablokowane" after about 1 s and stops.
Only then, or after an FF 2 release and a power cycle, use **Wlacz silniki A1X (FF 1-5-6)**.

### 2. Home and power-off

The A1X's *Home* is the model's raised pose: J2 57°, J3 −92°, J4 34°, jaws closed.
So the first Home is a big move up.
*Set home = here* stores the measured arm pose and always gripper 0. Close the empty jaws for real first.
Before you cut power, fold the arm with the sliders (J2 → 3°, J3 → −3°, the rest → 0) and support it.
The A1X has no brakes. The last 3° to the stop are the twin's limit margin.

### Then

Cameras, intrinsics, camera poses, sim-real and dynamics follow the twin's own procedure,
in the twin repository's `docs/TWIN.md`.
The card in the jaws sticks out about 70 mm.
The calibration wave samples poses from `wave_ranges` in `robot.yaml`.

### Open questions

| Question | Why in doubt | How to check |
|---|---|---|
| 0.04 rad of pressure when closed, the margin when open | the endpoints come from the measured stops. Nobody tried them on the table yet | close empty: group 7 about +0.039, \|effort\| near rest, jaws touch. Open: group 7 about −1.95, effort near rest |
| effort unit and the `gripper_force` threshold of 1.2 | the `/600` scale has no known unit. The threshold is empirical, crushing from about 6 | grip a 3 cm cube: the target must freeze on contact. Check that the gripper effort is not stuck at 0.00. That means a deaf gripper, and the backend does not detect it |
| does `kp` 20 keep up at 90°/s? | `kp` is ignored, measured on small steps. Fast moves were not checked with the twin | Home and wave: no lag error (25° for 0.5 s), and the sliders follow the arm |
| the "deaf or blocked" threshold | 3° for 1 s on a standing joint. A joint resting on an obstacle trips it too | watch whether it fires during calm work. Tune `deaf_err_deg` / `deaf_s` if needed |
| J3 and gripper zero | +1.49° comes from the galaxeo arm. The closed jaw stop is +0.039 rad (2026-09-27), but nobody checked that the plates touch there | measure `joint_offsets_deg` as above. At gripper 0 the jaws must touch |

## In simulation

```bash
lerobot-twin demo --robot a1x                          # the workspace with two sim cameras. Panel: Arm sim, Connect
lerobot-twin calib-sim --robot a1x --n 3 --cameras 2   # camera calibration in simulation
```

[a1x/README.md](a1x/README.md) describes the model and what was checked with it.
