# galaxeo.robot

The Galaxea A1X as a plugin for the digital twin (package `lerobot_mp`).
It holds the robot folder `a1x/` and the hardware backend `a1x`, which implement the twin's plugin contracts.
The twin finds them through the entry points `lerobot_mp.robots` and `lerobot_mp.backends`. galaxeo never imports the twin.

The digital twin is machinekind's tool that measures a robot station, its cameras and its dynamics.
It runs any control strategy on the station through a safety supervisor, in simulation and on hardware.
This package plugs the A1X into it.
The twin lives in [digital_twin_training](https://github.com/machinekind/digital_twin_training), a private repository at the moment.

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

`create(ctx)` in `backend.py` builds an `A1XArm` from the twin's `BackendContext`, which it reads by duck typing.

## Entry points

```toml
[project.entry-points."lerobot_mp.robots"]
a1x = "galaxeo.robot:ROBOT_DIR"

[project.entry-points."lerobot_mp.backends"]
a1x = "galaxeo.robot.backend:create"
```

## Install

Install galaxeo into the twin's environment, from a clone of this repository:

```bash
pip install -e ../galaxeo-manipulators            # Linux: standard library only
pip install -e "../galaxeo-manipulators[xcan]"    # macOS: + pyusb (and brew install libusb)
pip install -e "../galaxeo-manipulators[pcan]"    # Windows: + python-can
```

Or straight from git:

```bash
pip install "galaxeo[xcan] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
```

The twin then offers the A1X. `lerobot-twin demo --robot a1x` copies `a1x/demo.json`.
When an adapter's module is missing, `connect()` names the extra to install.

## Port

The backend opens the CAN link on the port the twin passes. Bring the link up first.
[docs/HARDWARE.md](../../docs/HARDWARE.md) has the wiring and the timings.
On Linux, section 0 of [docs/STEERING.md](../../docs/STEERING.md#0-bus-first) brings it up.
On macOS, [the XCAN driver](../../docs/STEERING.md#macos-the-adapter-without-a-kernel) talks to the adapter.
`connections` in `robot.yaml` names one port per system:

| System | Port |
|---|---|
| Linux | `can0` |
| macOS | `xcan`. With two adapters, `xcan:<USB address>` |
| Windows | `PCAN_USBBUS1` |

An empty port means `xcan` on macOS and `can0` everywhere else, so Windows needs `PCAN_USBBUS1` set.

## The `a1x:` configuration section

`A1XConfig` reads the `a1x:` section of the twin's config file. Unset keys take the defaults below.
The section is strict. An unknown key, or a value of the wrong type, fails `create(ctx)` and names the key.

| Key | Default | Meaning |
|---|---|---|
| `kp` | 20.0 | gain sent in `0x050`. The arm ignores it, because only `p_des` acts. At `kp` 0 the arm stops listening, so `kp <= 0` is refused |
| `kd` | 1.0 | damping sent in `0x050` and `0x051`. It has no effect on motion |
| `gripper_kp` | 25.0 | `kp` of the gripper frame `0x051`. It must also be > 0 |
| `gripper_open_pdes` | −1.95 | gripper `p_des` [rad] at 100 (open). More negative is wider. It stays 0.04 rad before the open stop (−1.992) |
| `gripper_closed_pdes` | 0.08 | gripper `p_des` [rad] at 0 (closed): 0.04 rad of pressure past the closed stop (+0.039). Never +0.6, see [The gripper](../../docs/HARDWARE.md#the-gripper) |
| `gripper_fb_closed` / `gripper_fb_open` | 0.039 / −1.992 | group 7 position of `0x052` [rad] at 0 and at 100, the stops measured on 2026-09-27. Set both or neither, and not equal. `null` in both means open loop: the backend reports the last command, and `calibration_warnings()` says so |
| `gripper_start` | 0.0 | the gripper value the backend reports before the first command, while the gripper is not measured. 0 is closed, as in the station's home pose |
| `gripper_rate_pdes_s` | 2.0 | highest gripper `p_des` speed [rad/s], > 0. The full 2.03 rad stroke takes about 1.0 s |
| `gripper_force` | 1.2 | how far group 7's \|effort\| may rise above rest while closing before the gripper target freezes. Empirical unit. Crushing was seen from about 6 |
| `joint_offsets_deg` | 6 × 0.0 | arm joint zero [deg], six values. Subtracted on read, added on send |
| `stream_hz` | 200.0 | rate of the `0x050` / `0x051` stream, > 0. The arm answers `0x052` every 5 ms |
| `max_step_deg_s` | 90.0 | the backend's own `p_des` speed limit [deg/s], > 0. A second layer under the supervisor |
| `stale_ms` | 150.0 | no `0x052` for longer than this means the link is silent |
| `frozen_frames` | 50 | this many identical `0x052` frames in a row mean frozen telemetry: the arm was released (FF 2) |
| `deaf_err_deg` / `deaf_s` | 3.0 / 1.0 | a command this far from the measurement, for this long, on a standing joint (< 0.3 deg/s) means the arm is deaf to `0x050` or blocked. The backend reports a fault and the twin stops |
| `enable_on_connect` | `false` | runs `enable()` right after `connect()`. Keep it `false`. FF 5 briefly releases the motors |
| `gripper_over_xcan` | `true` | drives the gripper over XCAN (macOS). That works, measured on 2026-09-27. `false` makes `gripper_drivable()` false there. `A1X_GRIPPER_XCAN=1` forces `true` |

### Joint zero (`joint_offsets_deg`)

On the galaxeo arm J3 reads about +1.49° at rest, past its own 0° limit.
That is a zero offset. [docs/PROTOCOL.md](../../docs/PROTOCOL.md) confirms the decoding.
Measure yours:

1. Fold the arm onto its stops and power it.
2. Set `joint_offsets_deg` to all zeros and connect. Send no command.
3. Read J2 and J3, which rest on their stops. `connect()` logs them in the line `A1X on …: start pose …`.
4. Put the J3 value third, for example `joint_offsets_deg: [0.0, 0.0, 1.49, 0.0, 0.0, 0.0]`.
   Leave the rest at 0 unless you have a sure reference. J3 then reads about 0.

## What the backend does

- Nothing goes on the bus until the first `send_joints()`, so `connect()` never drives home.
- `connect()` fails when no `0x052` arrives within 3 s, or when the telemetry is frozen.
- The `p_des` stream starts from the measured pose and moves at most `max_step_deg_s`. It runs at `stream_hz` and repeats the last target.
- It clamps to the URDF limits, widened to the pose at `connect()`. The folded arm stands outside the twin's own limits.
- Link loss is no `0x052` for `stale_ms`, or frozen telemetry. The backend then stops sending at once.
  It drops the arm and gripper targets and reports a fault in `faults()`, and the twin stops.
  When frames return, nothing moves until the next command, and that command starts from a fresh measurement.
- It never sends FF 2, never `kp` 0 and never the `0x023` heartbeat. `disconnect()` ends the stream and releases nothing.
- Its one action is `enable()`. It sends FF 1 → 5 → 6, only on request or with `enable_on_connect`.
  During the sequence it streams `p_des` equal to the measured pose. A freshly powered arm does not need it.

### The gripper

- Group 7 of `0x052` measures the jaws only while `0x051` flows.
  The backend trusts it from 4 `0x052` frames after the first `0x051` of the connection, and again after a link loss.
  Before that it reports the last command, or `gripper_start` before any command.
- The `0x051` stream starts only on a command more than 0.5 away from the reported value, so connecting closes nothing.
- `assume_gripper(0)` confirms a closed gripper without a measurement.
  It returns `False` on a measured gripper, or once the `0x051` stream has started.
- Each close measures rest as the median \|effort\| of its first 0.1 s.
  After that window the target freezes once \|effort\| exceeds rest by `gripper_force`.
  Any opening command, or one equal to the target, lifts the freeze.
- The next close measures rest again only after the jaws opened 0.3 rad of `p_des` from the tightest point.
  After a smaller opening the jaws still press the object, and that pressure must not become the new rest.
- While the jaw state is only a guess (`gripper_start` or `assume_gripper`), the first opening goes straight to the target.
  Closing always ramps, so a wrong guess never snaps the jaws shut.

## Units

| | App | Twin (MJCF) | Arm (CAN) |
|---|---|---|---|
| arm joints | degrees | radians, the same angles | radians + `joint_offsets_deg` |
| gripper | 0..100 (0 = closed) | jaw travel 0..0.05 m (0 = plates together, tips 3 mm apart). The second jaw mirrors the first | `p_des` [rad], linear from +0.08 (0) to −1.95 (100). Group 7 of `0x052` [rad]: +0.039 (0) to −1.992 (100) |

The A1X gripper has no servo ticks, so the SO-101's tick settings do not apply.
The jaw thresholds (`jaw` in `robot.yaml`) are in metres.
20 mm of travel is 43 mm between the tips, more than the 42 mm diagonal of a 3 cm cube.
