# Hardware

The setup this repo was developed against: a **Galaxea A1X** 6-DOF arm with a
gripper, wired directly to an x86_64 Ubuntu laptop over CAN-FD. Some work was
done with two identical arms on two adapters.

## Power and wiring

* **24 V** supply — not 48 V, despite what some notes elsewhere say.
* The CAN box has termination switches **R1 and R2, both in the top position**.
* Adapters: XCAN / PEAK USB-CAN-FD, USB id `0c72:0012`, kernel driver
  `peak_usb`.

> **No brakes.** Cutting power drops the arm. Read [SAFETY.md](SAFETY.md).

## Bus settings

These are the vendor's own timings, from `start_hdas_r1.sh`:

```bash
ip link set can0 type can \
    bitrate 1000000 sample-point 0.875 \
    dbitrate 5000000 dsample-point 0.875 \
    fd on restart-ms 100
ip link set can0 txqueuelen 65535
ip link set can0 up
```

**Omit `berr-reporting on`.** It appears in Galaxea's R1 script, but
`pcan_usb_fd` does not support it and its presence makes the whole `ip link
set` fail. `can_up.sh` leaves it out.

Links come up **down** after every replug. Re-run `./can_up.sh`.

## Telling two adapters apart

Both XCAN adapters report the identical `ID_SERIAL` (`XCAN_XCAN-USB_FD`), so
the serial is useless as a handle. Two options, both imperfect:

* **By traffic** — what `can_up.sh` does. It configures every interface,
  listens 2 s on each, and takes whichever carries frames. Reliable with one
  arm; a coin flip with two, since **both** arms transmit unprompted at 200 Hz.
* **By USB port path** — stable within a session, but enumeration order has
  been observed to move across replugs (`1-1.4`, `1-2.4`, `1-1` on different
  days).

With two arms, settle it empirically instead:

```bash
python3 which_arm.py can0 can1
```

Read-only. Push each arm by hand in turn and watch which line reacts:

| output | meaning |
| --- | --- |
| `MOVING` | joints changing — compliant *and* reporting |
| `still` | reporting but not moving: nobody is pushing, or it is holding against you |
| `FROZEN` | identical payloads — transmitting but not reporting. This is the released state (function frame 2); such an arm is useless as a leader |

The vendor's `ARM_APP` **hardcodes `can0`** and has no parameter to change it,
so whichever arm the vendor stack drives has to be `can0`. Our own code takes
an interface argument and doesn't care.

## Joint limits

From the A1X URDF in `ros2_ws/src/galaxea_a1xy_description/`.

| joint | axis | limit (rad) | limit (deg) |
| --- | --- | --- | --- |
| `arm_joint1` | base yaw | −2.880 … 2.880 | −165 … 165 |
| `arm_joint2` | shoulder pitch | 0.0 … 3.142 | 0 … 180 |
| `arm_joint3` | elbow pitch | −3.316 … 0.0 | −190 … 0 |
| `arm_joint4` | wrist pitch | −1.571 … 1.571 | −90 … 90 |
| `arm_joint5` | wrist yaw | −1.571 … 1.571 | −90 … 90 |
| `arm_joint6` | wrist roll | −2.880 … 2.880 | −165 … 165 |

`arm_joint3` reads about **+1.5 deg** at rest, i.e. just outside its own upper
limit. This is systematic across every capture and is a zero-offset, not a
decode error — the group→joint mapping was confirmed independently by a push
test. Compensate with `joint_offsets` in
`ros2_ws/src/galaxea_a1xy_driver/config/driver.yaml`, after measuring your own.

## Resting behaviour

Worth knowing, because it looks like a fault and isn't:

* An arm with **zero frames transmitted to it** still holds its position and
  still reports `0x052` at 200 Hz. That is the hardware's resting state.
* It is a position servo with finite stiffness, not a brake. You can deflect it
  by hand and the motors put it back. Measured excursion under hand force with
  nothing transmitting: 0.05–0.07 deg.
* An arm that has been released (function frame 2) goes limp **and stops
  reporting** — the same payload repeats forever.

## The gripper

Measured 2026-09-27 on a real A1X over XCAN, with
`python -m galaxeo.xcan_usb --grip-test`. It streams `0x051` at 200 Hz with
kp 20 and kd 1, and holds each step for 3 s.

| `p_des` [rad] | group 7 of `0x052` | peak \|effort\| | |
| --- | --- | --- | --- |
| −1.50 | −85.60 deg = −1.494 rad | 0.25 | free motion |
| 0.00 | −0.40 deg = −0.007 rad | 0.21 | free motion |
| −2.00 | −114.14 deg = −1.992 rad | 3.69 | on or just past the open stop |
| +0.60 | +2.21 deg = +0.039 rad | 7.80 | **closed stop, 0.56 rad past it** |

Group 7 is the jaw position in the units of `p_des`, but only while `0x051`
flows (see [PROTOCOL.md](PROTOCOL.md#group-7-is-the-gripper-motor-angle-not-the-vendors-stroke-value)).
In free motion the measurement equals the command.

**Stops.** The closed stop is at about +0.039 rad (+2.2 deg). The open stop
is at about −1.992 rad (−114 deg).

**+0.6 overheated the gripper motor.** The gripper is position-controlled.
At `p_des` +0.6 the jaws stop at +0.039, and the controller keeps pushing
toward a target 0.56 rad further. At kp 20 that is a stall current. The
\|effort\| was 7.8, against about 0.2 in free motion. It flowed for as long
as "closed" was held. The twin's contact stop (`gripper_force`) does not always
prevent this. A close that starts with the jaws already on the stop measures
the stall current as rest. `--grip-test` has no contact stop at all.

**Chosen endpoints.** Closed is +0.08, 0.04 rad past the stop. That is about
7 % of the old error. The contact stop catches objects anyway. Open is −1.95,
0.04 rad before the stop. At −2.0 the jaws leaned on the stop (effort 3.7).
With −1.95 the measurement at gripper 100 reads about 98, on purpose. These
are the defaults of the twin backend, see
[galaxeo/robot/README.md](../galaxeo/robot/README.md).

**Not checked yet.** Whether 0.04 rad of pressure closes the empty jaws
fully, and whether −1.95 is clear of the open stop. One try on the table
answers both. Close the jaws: group 7 should read about +0.039, with the
effort near rest. Then open: group 7 should read about −1.95, with the effort
near rest.

### Measuring again

Measure again on another arm, or after gripper service.

The easiest way is `python -m galaxeo.xcan_usb --grip-test` (macOS, XCAN).
It holds a few `p_des` steps for 3 s each and prints group 7 and the peak
\|effort\| of each step. Close the twin's panel first. One process at a time
holds the XCAN adapter.

* The stops are where the position stops following `p_des` and the effort
  rises. Put them into `gripper_fb_closed` and `gripper_fb_open`.
* Set `gripper_closed_pdes` a few hundredths of a radian past the closed
  stop, and `gripper_open_pdes` a few hundredths before the open stop.
* **Do not stay far past a stop for long.** The stall current heats the
  motor. Steps of 3 s are enough.

On Linux with SocketCAN, this script reads group 7 and sends nothing. It
prints the median of 50 `0x052` frames:

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

Something else must stream `0x051` meanwhile, for example the twin's panel.
Empty the gripper, press ARM, set the gripper slider to 0 and wait for the
jaws to stop. Run the script. Then do the same at 100.

* Read only while `0x051` flows. At rest group 7 can be anything.
* On Linux the script can read next to the panel. SocketCAN takes many
  readers next to one sender.
* On macOS the panel holds the adapter, and *Disconnect* stops the stream.
  So the script cannot read reliably there. Use `--grip-test`.

## The wrist camera

A 32 mm UVC board camera with a wide M12 lens on a printed strut behind the
G1 housing, looking down the jaw gap at the fingertips. Fitted 2026-09-25.

* macOS reports it as `USB Camera`, UVC vendor 4283 / product 11016. It
  streams **1920 x 1080**, a 16:9 sensor.
* It is mounted **upside down**: rotate every frame 180 deg. `wrist_cam.py`
  does; the recorder must do the same.
* Focus is the M12 barrel, by hand. `uv run wrist_cam.py` shows the live feed
  in a browser with a focus score; turn until it peaks.
* Position is a compromise for pick-and-place, not for the bottle pour: it sits
  over the housing where a body-grasped bottle would hit it. The outboard
  bracket on PR #2 was designed around that bottle and is not needed for this
  task.
* Aim: the fingertips should sit about a third of the way up from the bottom
  edge of the frame, so the object beyond the tips stays in view.

### Where it is (measured 2026-09-26)

Eye-in-hand calibration with `calib_wrist.py`: a sheet of six AprilTag 36h11
(ids 3-8, 37 mm) flat on the table, the arm carried the camera around it over
two sessions (59 poses, 152 tag views). The fit lives in
`hardware/g1_camera_mounts/camera_spec_lashup.json`, which
`sim/wrist_camera.attach_wrist_camera(..., design=load_spec(that file))` reads.

| quantity | value |
| --- | --- |
| lens in `gripper_link` | (-51, 43, 94) mm |
| optical axis | 31 deg toward -y, 26 deg below the tool axis |
| field of view, full frame | 74.7 x 45.7 deg (K from a 6x4, 39 mm checkerboard, 15 views, 0.85 px) |
| reprojection residual | 28 px at 12-24 cm, 10.5 px at 15-32 cm |

The residual is far above the 1.5 px the fixed-camera calibration reaches.
It behaves like a 3-4 mm position error per pose, not a lens problem: fitting
six constant joint offsets barely moves it (26 px), and the camera pose is
stable to about a centimetre between the near and the far session
((-51, 40, 92) against (-60, 51, 105) mm). So the number above is good to
roughly 1 cm and 2 deg, and the sim should randomise the wrist camera by at
least that much. Two open leads: the intrinsics (few views, strong distortion
terms), and the arm's own forward kinematics under the wrist load, which the
same tooling can measure against the tags.

The old `camera_spec.json` next to it is the rejected PR #2 outboard strut
(CAD nominal, never built); `payload.json` is that strut's mass and collision
shape too, and describes nothing on the arm today.

## The SO-101 leader

The arm cannot be hand-guided (see
[STEERING.md](STEERING.md#why-there-is-no-float-mode)), so hand teleoperation
uses a [SO-101](https://github.com/TheRobotStudio/SO-ARM100) as the leader.

* USB serial, `/dev/ttyACM0`.
* Calibrated through LeRobot under a calibration id — `my_leader` by default in
  these scripts.
* Its URDF is at `so101/so101_new_calib.urdf`; `kinematics.py` parses it
  directly, no ROS and no pinocchio involved.
* Measured workspace ratio, A1X / SO-101 95th-percentile reach: **1.657**.

One gotcha that costs an afternoon: `bus.sync_read("Present_Position")` returns
**bare motor names**, while lerobot's `get_action()` returns `<name>.pos`.
Mixing the two yields a silent 0.0 everywhere through `dict.get` defaults
rather than an error.
