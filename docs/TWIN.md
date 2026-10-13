# The A1X in the digital twin

The digital twin ([digital_twin_training](https://github.com/machinekind/digital_twin_training),
package `lerobot_mp`) drives the A1X with the same panel, supervisor and procedure as the SO-101.
It holds no robot of its own. This package brings the A1X to it:

* `galaxeo/robot/a1x/` — the robot folder: `robot.yaml` (joints, limits, home, labels,
  connections), the MuJoCo model `a1x.xml` and the example workspace `demo.json`. See its
  [README](../galaxeo/robot/a1x/README.md).
* `galaxeo/robot/backend.py` — the `a1x` backend: the twin's commands as CAN-FD frames, with
  `galaxeo.protocol` and `galaxeo.bus`.

The twin finds both through entry points (`lerobot_mp.robots`, `lerobot_mp.backends`) once galaxeo
is installed in its environment. galaxeo never imports the twin.

Read [SAFETY.md](SAFETY.md) before the first power-on. Protocol and measurements:
[PROTOCOL.md](PROTOCOL.md), [STEERING.md](STEERING.md), [HARDWARE.md](HARDWARE.md).

## Install

Into the twin's environment, from a clone of this repository:

```bash
pip install -e ../galaxeo-manipulators            # Linux: standard library only
pip install -e "../galaxeo-manipulators[xcan]"    # macOS: + pyusb (and brew install libusb)
pip install -e "../galaxeo-manipulators[pcan]"    # Windows: + python-can
```

Or straight from git: `pip install "galaxeo[xcan] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"`.
An installed galaxeo takes a missing extra with `pip install "galaxeo[xcan]"`; *Connect* names the
extra when its module is missing.

## Hardware and the CAN link

- A **24 V** supply (not 48 V). Termination switches R1 and R2 in the CAN box **up**. An XCAN /
  PEAK USB-CAN FD adapter (USB `0c72:0012`).
- **One sender on the bus.** While the panel drives an `a1x` arm, do not run `ARM_APP`,
  `so101_bridge.py`, `jog_a1x.py`, `teleop2.py` or a transmitting ROS 2 driver. Two streams on
  `0x050` interleave and the arm follows neither. The backend sees a `0x050` / `0x051` / FF frame
  it did not send and reports a fault for 0.5 s after the last one, so the twin STOPs.

| system | link | Port field in the panel |
|---|---|---|
| Linux | SocketCAN, kernel `peak_usb`; nothing more in Python | `can0` (empty = `can0`) |
| macOS | userspace driver `galaxeo.xcan_usb` (libusb): `brew install libusb` + extra `xcan` | `xcan`; with two adapters `xcan:<USB address>` (empty = `xcan`) |
| Windows | PEAK PCAN-Basic (PEAK driver) through python-can, extra `pcan` | `PCAN_USBBUS1` — **type it**, empty means `can0` |

**Linux.** `./can_up.sh` brings the link up from this repository's container (no `sudo` on the
host) and picks the adapter by bus traffic. Without the container, by hand (vendor timings):

```bash
sudo ip link set can0 type can \
    bitrate 1000000 sample-point 0.875 \
    dbitrate 5000000 dsample-point 0.875 \
    fd on restart-ms 100
sudo ip link set can0 txqueuelen 65535
sudo ip link set can0 up
ip -det -s link show can0     # link state, bus-off, error counters
```

Do not add `berr-reporting on`: `pcan_usb_fd` does not support it and the whole `ip link set`
fails. **The link comes back down after every replug**; bring it up again. Both adapters carry the
same serial number; `python3 which_arm.py can0 can1` shows which arm hangs on which link
(read-only).

**macOS.** There is no SocketCAN, and PCBUSB opens the adapter but receives nothing. So
`galaxeo.xcan_usb` talks to it through libusb. A check that sends nothing:

```bash
python -m galaxeo.xcan_usb --normal     # firmware version, then 0x052 at about 200 Hz
```

Without `--normal` the adapter listens without acknowledging. The arm then repeats every frame and
`0x052` arrives at about 6 kHz with repeated content; that is not a fault. **One process at a
time** can hold the adapter: close the check before the panel connects.

## Workspace and configuration

**Workspace.** The `robot` field of the workspace file picks the arm; the panel does not change
it. Start from the example:

```bash
lerobot-twin demo --robot a1x          # galaxeo/robot/a1x/demo.json -> workspace/twin.json (table 1.2 x 0.8 m, 2 sim cameras)
lerobot-twin workspace                 # "ramie: a1x, backend: sim, ..."
```

For the real station, delete both simulated cameras in the Cameras tab and add the USB ones. A
workspace file of just `{"robot": "a1x"}` also works (empty table, no cameras). *Connect* is what
stores backend and port in the workspace.

**`configs/local.yaml`** (in the twin). The `a1x` backend reads the `a1x:` section, which
`configs/default.yaml` does not have. Write just that section (unset keys take the defaults
below) and start the panel with `lerobot-twin --config configs/local.yaml ui`:

```yaml
a1x:
  joint_offsets_deg: [0.0, 0.0, 1.49, 0.0, 0.0, 0.0]   # measure yours (below)
```

The section is strict: an unknown key, or a value of the wrong type, fails *Connect* with the
key's name. The twin **ignores** `robot.backend` and `robot.port` from the file: the panel picks
them (Arm and Port) and the workspace remembers them.

| key | default | meaning |
|---|---|---|
| `kp` | 20.0 | gain sent in `0x050`. The arm ignores it on this link (only `p_des` acts), but at `kp` 0 it stops listening. `kp <= 0` is refused |
| `kd` | 1.0 | damping in `0x050` and `0x051`; no effect on motion |
| `gripper_kp` | 25.0 | `kp` of the gripper frame `0x051`; also > 0 |
| `gripper_open_pdes` | −1.95 | gripper `p_des` [rad] at 100 (open); more negative = wider. A hair before the open stop (−1.992), so it does not lean on it |
| `gripper_closed_pdes` | 0.08 | `p_des` [rad] at 0 (closed): 0.04 rad of pressure past the closed stop (+0.039). **Not +0.6**, see the gripper measurement below |
| `gripper_fb_closed` / `gripper_fb_open` | 0.039 / −1.992 | group 7 position of `0x052` [rad] at 0 and at 100 (stops measured 2026-09-27). Both or neither, and not equal. `null` in both = open loop: the panel shows the gripper **command**, not a measurement (warning "sprzezenie niezmierzone") |
| `gripper_start` | 0.0 | what the backend reports as the gripper before the first command without feedback. 0 = closed, the station convention "home = jaws closed" (*Set home = here*) |
| `gripper_rate_pdes_s` | 2.0 | highest gripper `p_des` speed [rad/s]; the full 2.03 stroke in about 1.0 s (`gripper_travel_s` adds 0.2 s) |
| `gripper_force` | 1.2 | how far group 7's \|effort\| may rise above rest while closing before the gripper target freezes (grip, do not crush). Rest is the median \|effort\| of the first 0.1 s of each close (the target does not freeze in that window), measured again after each opening. Empirical unit; crushing seen from about 6 |
| `joint_offsets_deg` | 6 × 0.0 | arm joint zero [deg]: subtracted on read, added on send |
| `stream_hz` | 200.0 | rate of the `0x050` / `0x051` stream; the arm answers `0x052` every 5 ms |
| `max_step_deg_s` | 90.0 | the backend's own `p_des` speed limit [deg/s], a second layer under the supervisor |
| `stale_ms` | 150.0 | no `0x052` for longer = the link is silent: sending stops, targets are dropped, the twin stops |
| `frozen_frames` | 50 | this many identical `0x052` frames in a row = frozen telemetry (arm released, FF 2) |
| `deaf_err_deg` / `deaf_s` | 3.0 / 1.0 | command further than `deaf_err_deg` from the measurement for `deaf_s` on a standing joint (< 0.3 deg/s) = arm deaf to `0x050` or blocked: error and STOP |
| `enable_on_connect` | `false` | the FF 1 → 5 → 6 sequence right after *Connect*. Keep `false` and use the panel button: FF 5 briefly releases the motors |
| `gripper_over_xcan` | `true` | gripper over XCAN (macOS) works (measured 2026-09-27); `false` = refuse ("no gripper"). `A1X_GRIPPER_XCAN=1` forces `true` |

**Joint zero (`joint_offsets_deg`).** On the galaxeo arm J3 reads about +1.49° at rest, past its
own 0° limit. That is a zero offset, not a decoding error. Measure yours: arm folded onto its stops,
powered, *Connect* without going home, `joint_offsets_deg` all zero. Read J2 and J3 (resting on
their stops) from the sliders or the log (`A1X na …: poza startowa …`). Put the J3 value (e.g.
`1.49`) third and leave the rest 0 unless you have a sure reference. The value is subtracted from
the reading, so after a panel restart J3 reads about 0.

### Gripper: the 2026-09-27 measurement

On a real A1X over XCAN (`python -m galaxeo.xcan_usb --grip-test`: `0x051` at 200 Hz, kp 20,
kd 1, 3 s per step):

| `p_des` [rad] | group 7 of `0x052` | peak \|effort\| | |
|---|---|---|---|
| −1.50 | −85.60° = −1.494 rad | 0.25 | free motion |
| 0.00 | −0.40° = −0.007 rad | 0.21 | free motion |
| −2.00 | −114.14° = −1.992 rad | 3.69 | on or just past the open stop |
| +0.60 | +2.21° = +0.039 rad | 7.80 | **closed stop, 0.56 rad past it** |

- **Group 7 of `0x052` is the jaw position**, in the units of `p_des` (rad, `raw / 4700`). In free
  motion measurement = command. The backend reads it as `pos[6]` and maps it to 0..100 through
  `gripper_fb_closed/open`. While the gripper is driven it **is measured** (`gripper_measured()`):
  the panel, the 3D jaws and grip detection ("in hand") see the jaws, not the command.
- **At rest after power-on group 7 shows an arbitrary value**, e.g. −164.9° (−2.88 rad) with the
  jaws where the previous run left them (about −85°). It follows `p_des` only while `0x051`
  flows. So the backend trusts group 7 only from 4 `0x052` frames after the first `0x051` of the
  connection (and again after a link loss: the arm may have been off). Before that it reports its
  belief as without feedback: `gripper_start` (0), *Set home = here* works (`assume_gripper`), and
  the first opening goes straight to the target. The rule is "since the stream", not "a value in
  the measured range ± 0.15 rad": an arbitrary rest value can land in the range, while the `0x051`
  stream is the condition the measurement showed. *Connect* logs the raw rest value
  ("A1X chwytak: grupa 7 w spoczynku …").
- **Stops:** closed about +0.039 rad (+2.2°), open about −1.992 rad (−114°): the defaults of
  `gripper_fb_closed` / `gripper_fb_open`.
- **Why not +0.6.** The gripper is position-controlled: at `p_des` +0.6 the jaws stop at +0.039
  and the controller keeps pushing toward a target 0.56 rad further. At kp 20 that is a stall
  current: \|effort\| 7.8 instead of about 0.2 in free motion. It flowed for as long as "closed"
  was held and **overheated the gripper motor** (2026-09-27). The contact stop (`gripper_force`)
  does not always prevent it: a close that starts with the jaws already on the stop measures the
  stall current as rest, and `--grip-test` has no contact stop at all. Closed is now +0.08:
  0.04 rad past the stop, about 7 % of the old error. The contact stop catches objects anyway.
- **Open:** −2.0 leaned on the stop (effort 3.7). Now −1.95, 0.04 rad before it. The measurement
  at 100 then reads about 98, not 100, on purpose.

Not checked yet: whether 0.04 rad of pressure closes empty jaws fully, and whether −1.95 is clear
of the stop. One try on the table: close, read \|effort\| and group 7 (about +0.039, effort near
rest), then open and the same at about −1.95.

**Measuring again** (another arm, or after gripper service). Easiest: `python -m galaxeo.xcan_usb
--grip-test` (macOS, XCAN): a few `p_des` steps of 3 s with group 7 and peak \|effort\| read. Close
the panel first (one process holds the adapter). The stops are where the position stops following
`p_des` and the effort rises: put them into `gripper_fb_closed` / `gripper_fb_open`. Set
`gripper_closed_pdes` a few hundredths of a radian past the closed stop and `gripper_open_pdes` a
few hundredths before the open one. **Do not stay far past a stop for long**: the stall current
heats the motor; 3 s steps are enough.

Without `--grip-test` (Linux, SocketCAN) a script that sends nothing reads group 7, the median of
50 `0x052` frames:

```python
from galaxeo import bus, protocol
b = bus.open_bus("can0")          # macOS: "xcan", Windows: "PCAN_USBBUS1"
try:
    pos = []
    while len(pos) < 50:          # 50 frames of 0x052 = 0.25 s
        f = b.recv(0.1)
        if f is not None and f.can_id == protocol.FB_ID and len(f.data) == protocol.FB_LEN:
            pos.append(protocol.decode_feedback(bytes(f.data)).pos[6])
    print(f"group 7: {sorted(pos)[25]:.4f} rad")
finally:
    b.close()
```

Empty gripper, ARM, gripper slider at 0, wait for the jaws to stop, run the script; the same at
100. Read only while `0x051` flows. On Linux the script can read next to the panel (many readers,
one sender). On macOS the panel holds the adapter and *Disconnect* stops the stream, so such a
reading is not reliable there (group 7 at rest can be anything): use `--grip-test`.

## First session, step by step

Before: arm folded and resting, the space under it clear, the supply's emergency switch and the
panel's STOP at hand, nothing else sending on the bus. Link checked (Linux:
`ip -det -s link show can0`; macOS: `python -m galaxeo.xcan_usb --normal`, then close it).

**0. Connect.** `lerobot-twin --config configs/local.yaml ui`, Arm tab: the *Galaxea A1X* entry
for your system (Port as in the table above), *Connect*. It never drives home. Nothing goes on the
bus until you move something. The sliders show the measured pose. *Connect* fails when:

- no `0x052` frame arrives within 3 s: the link is not up or the arm has no power;
- the telemetry is frozen: the arm was released (FF 2). Support it, power it off and on. FF 1-5-6
  does not fix it: a released arm's pose cannot be read.

The folded arm usually stands **outside the twin's limits** (J2 about −0.2°, J3 about +1.5°,
limits 3…177° and −187…−3°). That is normal: the supervisor leaves a joint where it stands (soft
limits). The first move of the J2 or J3 slider leads the joint into range at 15°/s.

**1. A small move.** ARM, then one slider (e.g. J1 Podstawa) by a few degrees. The `p_des` stream
starts from the **measured** pose, at most 90°/s (backend) and no faster than the supervisor
allows. Check the directions and that the sliders follow the arm. The arm reports but does not
move? After about 1 s the error "… st. od rozkazu przez … s i stoi — ramie gluche na 0x050 … albo
zablokowane" and a STOP appear. Only then (or after an FF 2 release and a power cycle) use
**Wlacz silniki A1X (FF 1-5-6)**. The button shows only with a connected `a1x`, needs its
confirmation box, and refuses while the panel is armed or a policy, wave or identification owns
the arm. During the sequence the backend streams `p_des` = the measured pose. A freshly powered
arm listens right away and does not need it.

**2. Home and power-off.** The A1X's *Home* is the raised pose of the model (J2 57°, J3 −92°,
J4 34°, jaws closed), so the first Home is a big move up. *Set home = here* stores the measured
arm pose and **always gripper 0 (closed)**: the station convention, home = jaws closed. Close the
empty jaws for real first. While the gripper has not been driven in this connection (group 7 at
rest is uncertain, see above), or without feedback (`gripper_fb_*` = `null`), the twin assumes from
then on that they are closed (panel, 3D jaws, the next "open" ramps from 0). If the gripper was
already driven in this session, the button only stores home and the gripper state (measured by
then) stays (log warning "chwytak juz sterowany"): close it with the slider and click again. In
`sim` the scene's gripper closes at once. **Fold the arm before cutting power** with the sliders
(J2 → 3°, J3 → −3°, the rest → 0) and support it: the A1X has no brakes. The last 3° to the stop
are the twin's limit margin. *Disconnect* releases nothing: the arm holds the pose it stands in.

**Then: cameras, intrinsics, camera poses, Sim-Real, dynamics, policies** — as steps 1–8 of the
SO-101 procedure in the twin's `docs/TWIN.md`, with the same confirmations. The card in the jaws
sticks out the same (about 70 mm); the wave samples poses from `wave_ranges` in `robot.yaml`. The
base policies (`reach-v3`, `lift-v3`) are trained on the SO-101: runner and evaluation refuse them
on an A1X station ("polityka uczona na so101, stanowisko ma a1x"). An A1X policy has to be trained
(`lerobot-twin train --task reach` with an A1X workspace, on an NVIDIA card). MuJoCo Warp has not
been tried with the A1X model yet (jaw equality, mesh colliders).

## How the A1X differs from the SO-101 (safety)

- **No brakes.** Cutting power (emergency switch, unplugged cable) drops the arm where it is.
  Fold or support it before power-off.
- **STOP holds the measured pose by streaming**; it does not release the motors. An arm that is
  sent nothing also holds its pose and does not drift. Stopping is safe; starting with a target
  other than the arm's pose is not, and the stream starting from the measurement prevents that.
- **The twin never sends FF 2** (release: the arm goes limp, drops and stops reporting). The `a1x:`
  section has no option to drop torque on disconnect; the `0x023` heartbeat is never sent.
- **FF 5 releases the motors for a fraction of a second.** Use the enable button only with the arm
  supported and the panel disarmed; never "just in case".
- **`t_ff`, `kp`, `kd` do nothing** (measured, see STEERING.md). No compliance, no gravity
  compensation, no yielding on collision. **Limits and the speed limit are the only protection**:
  the twin's limits (URDF minus 3–5°), the URDF limits in the backend (widened to the start pose),
  the supervisor and the backend's 90°/s. A collision is caught only by the lag checks (25° for
  0.5 s in the supervisor, 3° for 1 s on a standing joint in the backend).
- **Link loss.** No `0x052` for 150 ms, or frozen telemetry: sending stops at once, arm and gripper
  targets are dropped, the twin stops. When frames return nothing moves until the next command,
  and that one starts from a fresh measurement.
- **Gripper.** Its target cannot be read back from the arm; the jaw position can (group 7 of
  `0x052`), but only while the gripper is driven, from the first `0x051` of the connection. The
  closed `p_des` is +0.08, just past the stop: +0.6 kept the motor stalled and overheated it.
  Before the first `0x051`, and without feedback (`gripper_fb_*` = `null`), the backend runs open
  loop and the panel shows the command. The `0x051` stream starts only on a command differing from
  the reported value by more than 0.5, so *Connect* closes nothing. While closing, the target
  freezes once \|effort\| exceeds the rest of the first 0.1 s of the close by `gripper_force`
  (1.2); any opening command (or one equal to the target) lifts the freeze. The rest is measured
  again only after opening 0.3 `p_des` from the tightest point of the close: opened by a notch, the
  jaws still press the cube, and that pressure must not become the new rest (the threshold would
  grow every cycle). Without feedback the jaw state after *Connect* is `gripper_start` (closed), an
  assumption. Close open jaws with the slider before *Disconnect* and before *Set home = here*.
  While the state is only guessed (`gripper_start` or *Set home = here*), the first opening goes
  straight to the target (an empty closed gripper loses nothing, an open one does not twitch, and a
  wrong guess does not snap the jaws shut in the first frame); closing always ramps.
- **The gripper moves over XCAN (macOS).** The caveat on `galaxeo.bus.XcanBus` says the gripper
  ignores `0x051` from XCAN (10 bytes go out as 12, DLC 9). CAN FD has no 10-byte frame, so
  SocketCAN on Linux sends 12 bytes too, and on 2026-09-27 `--grip-test` moved the jaws over XCAN.
  So `a1x.gripper_over_xcan` defaults to `true`; `false` restores the refusal ("no gripper"). The
  same test checks an adapter with the panel closed.

## Units

| | app | twin (MJCF) | arm (CAN) |
|---|---|---|---|
| arm joints | degrees | radians, the same angles | radians + `joint_offsets_deg` |
| gripper | 0..100 (0 = closed) | jaw travel 0..0.05 m (0 = plates together, tips 3 mm apart); the second jaw mirrored | `p_des` [rad] linear +0.08 (0) … −1.95 (100); group 7 of `0x052` [rad]: +0.039 (0) … −1.992 (100) |

The A1X gripper has no ticks: `center_ticks` and `gripper_*_ticks` of the twin's `feetech:`
section are SO-101 only. Jaw thresholds are in metres (`jaw` in `robot.yaml`): 20 mm of travel is
43 mm between the tips, more than the 42 mm diagonal of a 3 cm cube.

## Open questions for the first session

| question | why in doubt | how to check |
|---|---|---|
| 0.04 rad of pressure when closed, the margin when open | endpoints set from the measured stops, not tried on the table yet | close empty: group 7 about +0.039, \|effort\| near rest, jaws touch; open: about −1.95, effort near rest |
| effort unit and the `gripper_force` 1.2 threshold | scale `/600` with no known unit; empirical threshold, crushing from about 6 | grip a 3 cm cube: the target must freeze on contact. Also check the gripper effort is not stuck at 0.00 (a deaf gripper; the backend does not detect it) |
| does `kp` 20 keep up at 90°/s? | `kp` is ignored (measured on small steps); fast moves not checked with the twin | Home and wave: no lag error (25° / 0.5 s), sliders follow the arm |
| the "deaf or blocked" threshold | 3° for 1 s on a standing joint; a joint resting on an obstacle trips it too | does it fire during calm work; tune `deaf_err_deg` / `deaf_s` if needed |
| J3 and gripper zero | +1.49° from the galaxeo arm; closed jaw stop +0.039 rad (2026-09-27), but whether the plates touch there is unchecked | `joint_offsets_deg` above; at 0 the jaws must touch |

## The A1X in simulation

Without hardware:

```bash
lerobot-twin demo --robot a1x               # the workspace with two sim cameras; panel: Arm sim, Connect
lerobot-twin calib-sim --robot a1x --n 3 --cameras 2
```

`import_twin_model.py` generates the model from the vendor URDF. The post-processing rules are in
the robot folder's [README](../galaxeo/robot/a1x/README.md), each with a test in
`tests/test_robot_a1x_model.py`. Jaw 2 follows jaw 1 through an equality in the MJCF, stiffened by
rule 6 (`solref`, `solimp`). Measured in the twin:

| | result |
|---|---|
| camera calibration, `calib-sim --robot a1x --n 3 --cameras 2` | 6/6 trusted; median 0.24 mm / 0.017°, worst 0.82 mm / 0.041°, residual 0.21 px |
| calibration session, seed 0, 1 camera | 16 poses, 0.20 mm, 0.017° |
| camera from the side (~180° to the base), seed 1, 2 cameras | 2/2 trusted (0/2 before the rotation-median fix in `calib/handeye.py`) |
| IK for `reach` goals at A1X scale | 20/20, worst 0.33 mm |
| calibration wave: poses with the card over the table | 85 % (170/200) |
| physics at the 5 ms RL step, random actions near the cube (2400 ticks) | no NaN, cube at most 2.0 m/s and 23 mm above the table |
| jaw symmetry under violent actions | worst 5.8 mm (36 mm with the default equality) |
| `lift` environment on the A1X | passes the Gymnasium check |

There are no A1X policies yet. GPU training (`lerobot-twin train --workspace <A1X workspace>`)
waits for the model to be checked in MuJoCo Warp.

**`pickplace`** runs on both arms: grasp the cube seen by the cameras, carry it and put it down on
the goal. On the A1X the observation has 42 numbers (39 on the SO-101). The IK expert solves it on
CPU (5 seeds, A1X in 79–86 of 300 ticks). There is no `pickplace` policy yet; training it on a GPU
(`lerobot-twin train --task pickplace`) is the next step.
