# Steering the arm

Four ways to move a Galaxea A1X, cheapest first. All of them are position
control — read [Why there is no float mode](#why-there-is-no-float-mode) before
you plan anything that assumes torque control, because on this transport there
isn't any.

> **The A1X has no brakes on its joint motors.** Cutting power drops the arm.
> Read [SAFETY.md](SAFETY.md) once before the first power-on. It is short.

| | You need | Gives you |
| --- | --- | --- |
| [1. Look before you touch](#1-look-before-you-touch) | the arm, a CAN adapter | live joint angles, nothing transmitted |
| [2. Hand-guide with an SO-101](#2-hand-guide-with-an-so-101) | + an SO-101 leader | a person driving the arm by hand |
| [3. ROS 2 driver](#3-ros-2-driver) | + Docker | `/hdas/*` topics, RViz, your own nodes |
| [4. Script it over raw CAN](#4-script-it-over-raw-can) | Python 3 only | ~40 lines to a moving arm |
| [5. On-screen jog](#5-on-screen-jog) | Python 3, macOS (libusb) or Linux | hold a key, a joint moves |

---

## 0. Bus first

Nothing works until the arm's adapter is up as `can0` at the right CAN-FD
timings. `can_up.sh` does this from inside the container (it has `NET_ADMIN`
and the host network namespace, so no host `sudo`):

```bash
./can_up.sh
```

It configures every CAN interface it finds, listens on each for 2 s, and picks
whichever one carries traffic — both XCAN adapters report the same serial
(`XCAN_XCAN-USB_FD`) and enumeration order has been observed to change across
replugs, so traffic is the only stable discriminator. The arm's adapter is then
renamed to `can0`, because the vendor's `ARM_APP` hardcodes that name.

Expected tail:

```
--- can0 ---
frames in 2s: 400  (~200 Hz)
OK: arm is streaming on can0.
```

If you have two arms, `which_arm.py` tells you which bus is which without
transmitting anything — push each arm by hand and watch which line reacts. See
[HARDWARE.md](HARDWARE.md) for timings and wiring.

---

## 1. Look before you touch

Read-only. Decodes `0x052` and prints joint state. Transmits nothing, so it
cannot disturb the arm whatever state it is in.

```bash
python3 which_arm.py can0                            # is it alive, is it moving
./ros2.sh shell
  ros2 run galaxea_a1xy_driver can_probe can0 3      # 3 s decoded dump
```

You should see 200 Hz, six joint angles inside the limits in
[HARDWARE.md](HARDWARE.md), velocities near zero and small efforts. If the
payload never changes, the arm is in the released state — see
[FF 2 froze my telemetry](#ff-2-froze-my-telemetry).

---

## 2. Hand-guide with an SO-101

**This is the recommended way to drive the arm**, and the one that gets used.

The A1X cannot be hand-guided (see [below](#why-there-is-no-float-mode)), so
instead of fighting that, a [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
— an arm built to backdrive, gripper included — is the leader and the A1X
follows. `so101_bridge.py` is that bridge.

### Wire-up

* SO-101 on its USB serial adapter: `/dev/cu.usbmodem*` on macOS,
  `/dev/ttyACM*` on Linux. The bridge finds it when there is exactly one;
  otherwise `--port`. Servo torque off, so it moves by hand.
* A1X on the XCAN dongle (macOS, `--follower xcan`, the default there) or on
  `can0` (Linux, bus up via `./can_up.sh`).
* Python with numpy, pyserial, and pyusb (macOS) or python-can (Linux). On
  the Mac that is `.venv-mac`. **No LeRobot needed**: `so101_feetech.py`
  reads the servos directly. If you have LeRobot and its calibration,
  `--leader lerobot` uses that instead.

### Calibrate the gripper once

Joint mode works with no calibration: the mapping is relative and a servo
tick is a known angle. The gripper needs one, because only a calibration says
which end of its travel is closed. Six seconds:

```bash
python so101_feetech.py --calibrate-gripper   # hold it closed, then fully open
python so101_feetech.py                       # live readout: 0% closed, 100% open
```

`--mode ik` needs absolute angles too: `--calibrate` does every joint end to
end, then the gripper.

Writes `so101/calib_my_leader.json` (`--cal-id` for another name).

### Dry run first, always

```bash
python so101_bridge.py --dry-run
```

Transmits nothing. Move each SO-101 joint and check the printed A1X targets
move the way you want. Any joint that goes the wrong way gets flipped with
`--signs`, one character per mapped joint in the order
`pan, lift, elbow, wrist_flex, wrist_roll`:

```bash
python so101_bridge.py --dry-run --signs '+-+++'
```

### Then for real

```bash
python so101_bridge.py --grip          # streams until Ctrl-C
```

The A1X starts from wherever it is; that pose and the leader's pose at
start are the zero of the relative mapping. If the arm is folded at its
rest pose, J2 and J3 sit on a limit and can only move one way from there;
either jog it to an unfolded pose first or pass `--home "0,60,-90,0,0,0"`
to have the bridge unfold it (slowly, `--home-rate` deg/s, keep clear).

| flag | default | what it does |
| --- | --- | --- |
| `--mode joint` | joint | 1:1 joint mapping. Preferred — exact and singularity-free |
| `--mode ik` | | FK the SO-101 tip, scale into the A1X workspace, solve A1X IK |
| `--follow-rate` | 90 | deg/s slew cap on the follower |
| `--gain` | 1.0 | motion gain in joint mode; < 1 for fine work |
| `--smooth` | 0.35 | low-pass on the leader |
| `--grip` | off | map the gripper too |
| `--grip-force` | 1.2 | freeze the grip target above this effort — grips, doesn't crush |
| `--kp` / `--kd` | 20 / 1 | follower gains |
| `--home` | | drive to an explicit pose (degrees) before engaging |
| `--leader` | raw | `raw` = pyserial reader, `lerobot` = LeRobot's bus + calibration |
| `--follower` | xcan / can0 | A1X bus, by platform |

The mapping is **relative to the pose both arms start in**, so nothing jumps on
start and no absolute calibration is needed. Put both arms roughly where you
want them, then start.

`arm_joint5` (wrist yaw) has no SO-101 counterpart and is held.

### Why joint mode over IK

The topologies line up almost exactly:

```
shoulder_pan   -> arm_joint1 (yaw)
shoulder_lift  -> arm_joint2 (pitch)
elbow_flex     -> arm_joint3 (pitch)
wrist_flex     -> arm_joint4 (pitch)
wrist_roll     -> arm_joint6 (roll)
```

`--mode ik` exists for when Cartesian correspondence matters more than posture.
Measured workspace scale (ratio of 95th-percentile reach) is 1.657. Keep
`--w-rot` low: the SO-101 has 5 DOF and cannot specify a full 6-DOF
orientation, and demanding exact orientation costs about 10 mm of position
error on targets that are otherwise exactly reachable.

### Recording demonstrations

`record_a1x.py` is the same teleop loop wrapped in a LeRobot v3 dataset writer
— parquet plus encoded video, so it trains with lerobot policies and uploads to
the Hub unchanged.

```bash
python record_a1x.py --repo-id you/a1x_candy --task "pick up the candy" \
    --cameras "top:0,wrist:2" --episodes 5 --episode-secs 20
```

```
action             (7,)  commanded target: 6 joints + gripper
observation.state  (7,)  measured:         6 joints + gripper
observation.images.<name>   one video stream per camera
```

Ctrl-C ends the current episode cleanly and stops. `--no-record` gives you the
live rerun display without writing a dataset.

---

## 3. ROS 2 driver

`ros2_ws/src/galaxea_a1xy_driver` speaks the protocol in
[PROTOCOL.md](PROTOCOL.md) over SocketCAN and publishes vendor-compatible topic
names, so code written against it still works if you later switch to Galaxea's
own stack.

```bash
./ros2.sh build                                       # first time, pulls ~4 GB
./can_up.sh
./ros2.sh colcon
./ros2.sh shell
  ros2 launch galaxea_a1xy_driver driver.launch.py    # publish
  ros2 launch galaxea_a1xy_driver display.launch.py   # live arm in RViz
```

| topic | type | notes |
| --- | --- | --- |
| `/hdas/feedback_arm` | `sensor_msgs/JointState` | 6 joints, position/velocity/effort |
| `/hdas/feedback_gripper` | `sensor_msgs/JointState` | gripper only |
| `/joint_states` | `sensor_msgs/JointState` | arm + URDF fingers, for RViz |

Measured 200.1 Hz, 0.28 ms standard deviation.

### Transmit is off by default

`command_can_id` defaults to `-1`, which makes the driver read-only. The
command id **is** known now — `0x050`, confirmed on the wire against the
vendor's own HDAS — so to command the arm:

```bash
ros2 launch galaxea_a1xy_driver driver.launch.py command_can_id:=80   # 0x050
```

It stays opt-in on purpose. This arm has no brakes, and a control payload
landing on the right id with the wrong contents commands full-speed motion.

`config/driver.yaml` holds `can_interface`, `joint_offsets` (radians,
subtracted from each raw reading) and the rest.

### Arm-to-arm teleop as ROS nodes

`galaxea_a1xy_teleop` does leader→follower inside ROS, tuned in
`config/teleop.yaml`. Gentle defaults; raise `kp` only once the mapping is
confirmed to behave. Note the shape of the topics matters — a wrong-length
array is dropped silently rather than erroring.

---

## 4. Script it over raw CAN

No ROS, no vendor binaries, no dependencies beyond Python 3 on Linux. This is
what `teleop2.py`, `kp0_test.py` and `phase_compare.py` do, and the whole
recipe is ~40 lines. Full field layout in [PROTOCOL.md](PROTOCOL.md).

```python
import socket, struct, time

FIELDS = ((-6.5, 6.5, 4700.0),    # p_des  rad
          (-40.0, 40.0, 750.0),   # v_des  rad/s
          (0.0, 500.0, 60.0),     # kp
          (0.0, 200.0, 150.0),    # kd
          (-50.0, 50.0, 600.0))   # t_ff

def encode(p, v, kp, kd, t):
    """60 bytes: 6 joints x 5 big-endian int16."""
    out = bytearray(60)
    for j in range(6):
        for k, vals in enumerate((p, v, kp, kd, t)):
            lo, hi, sc = FIELDS[k]
            raw = max(-32768, min(32767, int(max(lo, min(hi, vals[j])) * sc)))
            off = j * 10 + k * 2
            out[off], out[off + 1] = (raw >> 8) & 0xFF, raw & 0xFF
    return bytes(out)

s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
s.bind(("can0",))

# read q from 0x052 first, then stream p_des = q and move it from there
frame = struct.pack("=IBBBB60s", 0x050, 60, 0, 0, 0, encode(q, [0]*6, kp, kd, [0]*6))
s.send(frame)
```

Three things that are not optional:

1. **Drain the RX socket every cycle.** Skip it and your readings go seconds
   stale while the kernel buffer fills, and every safety check you built on
   them is checking history.
2. **Stream continuously at 100–200 Hz.** A single frame is not a move command;
   the arm tracks the setpoint you keep sending.
3. **Start from `p_des = q`**, always. Sending a setpoint that isn't where the
   arm currently is makes it snap there at whatever speed it can manage.

An uncommanded arm holds where it is and does not drift, so stopping your
sender is safe — it does not fall and does not snap back to an old pose.

---

## 5. On-screen jog

`jog_a1x.py` is a small Tk window: hold a key or a button and one joint moves
at the speed on the slider. It is the only path here that runs on **macOS**.

```bash
brew install libusb && pip install pyusb        # macOS
pip install python-can                          # Linux (SocketCAN backend)
python jog_a1x.py --check        # read-only: Hz, pose, released or not
python jog_a1x.py --dry-run      # the window, transmits nothing
python jog_a1x.py                # ARM starts streaming; space disarms
```

Keys: `q/a` J1, `w/s` J2, `e/d` J3, `r/f` J4, `t/g` J5, `y/h` J6, `o/c`
gripper, `space` stop. The setpoint is seeded from the measured pose on ARM
and only moves at the slider's deg/s, so nothing jumps; stale feedback disarms
it; targets are clamped to the limits in [HARDWARE.md](HARDWARE.md). The
Enable button sends FF 1 → 5 → 6 with the setpoint pinned throughout and is
only for an arm that reports but ignores the jog.

The gripper's setpoint cannot be read back, so its first key press starts it
from the open position (`--grip-start`). Closing stops above `--grip-force`.

**Go home** slews every joint to `--home` (six degrees, default all zero: the
URDF zero, shoulder and elbow at their limits, the arm folded) at
`--home-speed`, 9 deg/s by default. Any jog key cancels it. It reports "safe
to disarm" on arrival; the arm holds there uncommanded, and a folded arm is
the one to cut power on, since there are no brakes.

Set home to the pose the arm *rests* in, not the exact zeros: at rest the
shoulder reads −0.2° and the elbow +0.2° (plus its known +1.5° offset), and
a joint commanded into its mechanical stop pushes there forever, which shows
up as steady effort at home. **Set home = here** captures the measured pose
and prints the `--home` value for next time.

### macOS: the adapter without a kernel

macOS has no SocketCAN. The freeware PCBUSB library (mac-can.com) opens the
XCAN dongles and even reads their firmware version, but **never receives a
frame from them** — tested on two units, every timing, listen-only and active.
So `xcan_usb.py` is a userspace driver: it speaks the uCAN protocol from the
Linux kernel's `peak_usb` driver directly over libusb. Command and record
layouts are transcribed from `drivers/net/can/usb/peak_usb/pcan_usb_fd.c` and
`include/linux/can/dev/peak_canfd.h`.

```bash
python xcan_usb.py --normal      # standalone: fw info, then 0x052 at ~200 Hz
```

`jog_a1x.py` uses it by default on macOS (`--iface xcan`, or `xcan:<usb
address>` to pin one of two dongles; with two plugged in it picks the one that
hears traffic). The driver refuses to run anywhere else: Linux already has
the adapter as SocketCAN, and the panel goes through python-can there
(`--iface can0`, or `PCAN_USBBUS1` on Windows with PCAN-Basic). Two things learned on the way:

* In **listen-only** mode the adapter does not ACK, so the arm retransmits
  every frame back-to-back and 0x052 appears at ~6 kHz with a repeating
  payload. Normal mode (ACK only, nothing sent) gives the true 200 Hz.
* The dongle is a **full-speed** USB device: command packets go out in
  64-byte pieces, as the kernel driver does.

---

## Why there is no float mode

Measured, 2026-08-23, on the A1X's CAN-FD path. Of the five command fields
**only `p_des` has any effect**:

| field | test | result |
| --- | --- | --- |
| `t_ff` | ±12 Nm, predicted ±34 deg offset | measured −0.02 deg — inert |
| `kp` | 0.5 vs 20 against a 3 deg step | identical tracking — ignored |
| `kd`, `v_des` | | no observable effect |
| `mode` | 0/1/2/3/10 | never reaches the wire; identical bytes |

An acceptance control in the same run passed, so this is the transport, not the
test. Consequences:

* **Gravity compensation is impossible here.** Galaxea's A1Z docs put `tau_g`
  from Pinocchio RNEA into `t_ff`; that field does nothing on this path.
* **So is any compliant or float mode.** J2/J3 will not backdrive by hand.
* **Position mirroring works at 98–100%**, which is why [the SO-101
  bridge](#2-hand-guide-with-an-so-101) is the answer rather than a workaround.

Do not spend time trying to make this arm compliant over CAN. Full evidence in
[diag/REPORT.md](../diag/REPORT.md).

---

## When it doesn't move

### The arm reports but ignores 0x050

It needs enabling, and enable is a **sequence**: function frames (`0x053`)
**1 → 5 → 6**. Code 1 alone leaves an arm holding and reporting but deaf.

A freshly power-cycled arm obeys immediately — you only need this to recover
from a release.

> **Stream `p_des = q` throughout the enable sequence.** Enabling without it
> caused a **76 deg swing at saturated torque** (effort 50.0). FF 5 briefly
> disengages the motors, so there is a moment where the arm is holding nothing.

### FF 2 froze my telemetry

Function frame 2 is release. The arm goes limp **and stops reporting** —
`0x052` repeats one identical payload forever. That is the released state, not
a dead bus. Re-enable with 1 → 5 → 6, streaming `p_des = q`.

### Two writers on one bus

Exactly one process may transmit per bus. `teleop2.py` refuses to start if
`ARM_APP` or a driver node is already running; do the same in anything you
write. Two senders on `0x050` means the arm sees them interleaved.

### Readings are stale / laggy

You are not draining the RX socket. See point 1 above.

### `ip link set` fails

Drop `berr-reporting on`. It is in Galaxea's own R1 script but `pcan_usb_fd`
does not support it and the whole command fails. `can_up.sh` already omits it.

### arm_joint3 reads ~1.5 deg outside its limit

Known, systematic, and a zero-offset rather than a decode error — the
group→joint mapping was confirmed by a push test. Compensate with
`joint_offsets` in `config/driver.yaml`. Measure your own before trusting it.
