# GALAXEO — an open driver for the Galaxea A1X arm

A working SDK for the **Galaxea A1X** 6-DOF arm, plus the documentation needed
to actually drive one.

Galaxea does not publish the A1X's CAN protocol, and the SDK their docs point
you at has been unobtainable since the download link died. So this is the
protocol recovered from their binaries and verified on real hardware, a ROS 2
driver built on it, and a teleoperation stack that lets a person pick the arm
up and move it.

Nothing here needs a vendor binary. If you have an A1X, a CAN-FD adapter and
Linux, you can drive it.

> **The arm has no brakes.** Cutting power drops it. Read
> [docs/SAFETY.md](docs/SAFETY.md) once before the first power-on — it is one
> page and it is the difference between a good afternoon and a bent arm.

## Start here

**[docs/STEERING.md](docs/STEERING.md)** — how to move the arm, four ways,
cheapest first.

The 60-second version, read-only, transmits nothing:

```bash
./can_up.sh                 # find the arm's adapter, bring it up as can0
python3 which_arm.py can0   # is it alive? is it moving?
```

Then the way this actually gets driven — an SO-101 as a hand-held leader, the
A1X following:

```bash
python so101_bridge.py --dry-run                           # prints targets, sends nothing
python so101_feetech.py --calibrate-gripper                # once: hold closed, then open
python so101_bridge.py --grip                              # for real, until Ctrl-C
```

Runs on macOS too (XCAN dongle through `xcan_usb.py`, leader over pyserial,
no LeRobot needed).

## Documentation

| | |
| --- | --- |
| [**STEERING.md**](docs/STEERING.md) | how to move the arm. Start here |
| [**SAFETY.md**](docs/SAFETY.md) | no brakes, and the two ways to get hurt by that |
| [**PROTOCOL.md**](docs/PROTOCOL.md) | the CAN-FD protocol, every field, and how it was recovered |
| [**HARDWARE.md**](docs/HARDWARE.md) | wiring, bus timings, joint limits, telling two adapters apart |
| [**VENDOR_SDK.md**](docs/VENDOR_SDK.md) | which Galaxea SDK is which, and where the useful one lives |
| [docs/notes/](docs/notes/) | working notes, kept for provenance |

## What's in here

### The driver — `ros2_ws/src/`

```
galaxea_a1xy_driver        SocketCAN driver, protocol module, probe tool
galaxea_a1xy_msgs          MotorControl.msg (mirrors hdas_msg/MotorControl)
galaxea_a1xy_description   A1X URDF + meshes
galaxea_a1xy_teleop        leader -> follower teleoperation as ROS 2 nodes
```

Publishes vendor-compatible topic names — `/hdas/feedback_arm`,
`/hdas/feedback_gripper`, `/joint_states` — so code written against this still
works if you later switch to Galaxea's own stack. Measured 200.1 Hz,
0.28 ms standard deviation.

Transmit is **off by default** (`command_can_id: -1`). Turn it on deliberately:
`command_can_id:=80`.

### The standalone tools — top level

No ROS, no vendor binaries, Python 3. Linux with SocketCAN, or macOS with the
XCAN dongle through `xcan_usb.py`.

| | |
| --- | --- |
| `so101_bridge.py` | SO-101 leader → A1X follower. The main teleop path; macOS or Linux |
| `so101_feetech.py` | reads the SO-101's Feetech servos over pyserial, and calibrates them. No LeRobot |
| `record_a1x.py` | the same loop, recording a LeRobot v3 dataset with cameras |
| `kinematics.py` | URDF serial-chain FK / Jacobian / damped-least-squares IK in numpy |
| `which_arm.py` | read-only: which bus has which arm, and can you move it |
| `jog_a1x.py` | on-screen jog, one joint per key. Runs on macOS |
| `wrist_cam.py` | live wrist-camera view in a browser with a focus score; runs on macOS |
| `galaxeo/` | the Python package: protocol, CAN transports, the XCAN driver. See [Python package](#python-package) |
| `galaxeo/xcan_usb.py` | userspace libusb driver for the XCAN / PCAN-USB FD dongle: macOS without SocketCAN |
| `teleop2.py` | arm-to-arm teleop over raw CAN, two A1X arms |
| `can_up.sh` | bring the arm's adapter up as `can0` at the right timings |
| `ros2.sh` | build / shell / run against the container |
| `diag/` | the probe scripts behind [`diag/REPORT.md`](diag/REPORT.md) |
| `ros2_ws/*.py` | one-shot probes from the reverse-engineering work, kept as worked examples |

### Python package — `galaxeo/`

The protocol and the CAN transports as an installable package, so other
projects drive the arm with the same encoders as the tools here:

```bash
pip install -e .                 # stdlib only
pip install -e ".[xcan]"         # + pyusb, for the XCAN dongle on macOS (brew install libusb)
pip install -e ".[pcan]"         # + python-can: PCAN on Windows, virtual:x loopback
```

```python
from galaxeo import protocol, bus

b = bus.open_bus("can0")         # or "xcan", "xcan:6", "PCAN_USBBUS1", "virtual:x"
fb = protocol.decode_feedback(b.recv(0.1).data)         # when it is a 0x052 frame
b.send(protocol.CMD_ID, protocol.encode_arm(fb.pos[:6], kp=20.0, kd=1.0))
```

* `galaxeo.protocol` - CAN ids (`0x050`-`0x055`; the vendor heartbeat `0x023`
  is documented and never sent), field scales and clamps, `encode_arm` /
  `encode_gripper` / `encode_ff`, `decode_feedback`, URDF limits,
  `ENABLE_SEQUENCE`. `encode_arm` refuses `kp <= 0`.
* `galaxeo.bus` - `open_bus(iface)`: raw SocketCAN (stdlib), the XCAN dongle
  (macOS), python-can. Frames go out with their true length (10 bytes on
  `0x051`); the 1-byte `0x053` goes as classic CAN with `fd=False`.
* `python -m galaxeo.xcan_usb` - read-only check of the XCAN dongle.

`jog_a1x.py` and `so101_bridge.py` import it from the repo root; the ROS 2
driver keeps its own copy, pinned byte for byte by `tests/test_protocol.py`
(`python -m pytest tests`).

### The container — `docker/`

ROS 2 Humble on Ubuntu 22.04, for a host that isn't. `ros2_ws/` is
bind-mounted, so you edit on the host and build inside. See
[Container details](#container-details) below.

## What was found

Three things worth knowing before you plan work on this arm.

**The protocol.** Feedback on `0x052` at 200 Hz, commands on `0x050`, gripper
on `0x051`, function frames on `0x053` — every field, scale and clamp read out
of the vendor's own `arm_encode_impl` / `arm_decode_impl` and cross-checked
against the wire. [PROTOCOL.md](docs/PROTOCOL.md).

**The arm is position-only.** `t_ff` is inert to ±12 Nm, `kp` is ignored,
`kd`/`v_des`/`mode` do nothing. Only `p_des` has any effect. So gravity
compensation is impossible on this transport, and so is any compliant or float
mode — J2/J3 will not backdrive by hand. Position mirroring, on the other hand,
works at 98–100%.
[The measurements](docs/STEERING.md#why-there-is-no-float-mode).

**Which is why the leader is a different arm.** Rather than fight for
compliance the A1X cannot give, an SO-101 — built to be hand-guided, gripper
included — leads and the A1X follows. The topologies map 1:1, relative to
whatever pose both arms start in, so there is no calibration step and nothing
jumps.

## Vendor SDKs

Not redistributed here; they are Galaxea's to distribute. Nothing in this repo
needs them.

```bash
./tools/fetch_vendor_sdk.sh atc     # the x86_64 ATC host build (HDAS/ARM_APP)
```

Three different packages get called "the Galaxea SDK" and two of them are for
other products — [VENDOR_SDK.md](docs/VENDOR_SDK.md) sorts out which is which,
where the live download is, and how it was tracked down.

---

## Container details

Ubuntu 24.04 host, ROS 2 Humble (Ubuntu 22.04) inside. Install Docker from the
Ubuntu archive:

```bash
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
sudo usermod -aG docker "$USER"      # applies to NEW login sessions only
```

Group membership only takes effect in a new login session. Until you log out
and back in, `docker` is permission-denied; prefix a shell with
`sg docker -c '...'` (or `newgrp docker`) to get it working in the session you
already have.

Then:

```bash
./ros2.sh build      # first build pulls ~4 GB (desktop-full)
./ros2.sh shell      # ROS 2 and the workspace overlay already sourced
```

| command | what it does |
| --- | --- |
| `./ros2.sh shell` | open a shell, starting the container if it's down |
| `./ros2.sh colcon` | `colcon build --symlink-install` on `ros2_ws` |
| `./ros2.sh run "ros2 topic hz /joint_states"` | one command in the container |
| `./ros2.sh down` | stop and remove the container |
| `./ros2.sh clean` | delete `build/`, `install/`, `log/` |
| `./ros2.sh --gpu up` | start with the NVIDIA runtime overlay |
| `./ros2.sh --noetic up` | the ROS 1 Noetic container, for the vendor's ROS 1 SDK |

Aliases inside: `cb` (build), `cbp <pkg>` (build one package), `ct` (test),
`sws` (re-source the overlay).

### Configuration notes

* **Non-root user.** The container user `ros` is created with your host UID/GID
  from `.env`, so build artifacts in `ros2_ws/` stay owned by you. It has
  passwordless `sudo` inside the container, and is in `dialout`, `video` and
  `plugdev`.
* **`network_mode: host` + `ipc: host`.** DDS discovery needs multicast and
  shared memory. Host networking also means CAN can be configured from inside
  the container (it has `NET_ADMIN`) with no host `sudo`. Isolate with
  `ROS_DOMAIN_ID` in `.env` instead — 0–101 are safe.
* **GUI.** `ros2.sh` runs `xhost +local:root` and mounts `/tmp/.X11-unix` and
  `/dev/dri`. RViz2 reports OpenGL 4.6 through the Mesa path. For CUDA or
  hardware GL, install `nvidia-container-toolkit` and use `./ros2.sh --gpu up`.
* **Simulator.** Humble's `desktop-full` ships **Ignition Gazebo Fortress**
  (`ign gazebo`), *not* Gazebo Classic — there is no `gazebo`/`gzserver`
  binary. Add `ros-humble-gazebo-ros-pkgs` to the Dockerfile if you need
  Classic.
* **Smaller image.** `BASE_IMAGE=osrf/ros:humble-desktop` in `.env` drops
  Gazebo; `ros:humble-ros-base` is headless.
* **Middleware.** Fast DDS by default; Cyclone is installed —
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in `.env` switches.
* **Adding packages.** Source packages go in `ros2_ws/src/`, then
  `./ros2.sh run "rosdep install --from-paths src --ignore-src -r -y"` and
  `./ros2.sh colcon`. Apt packages belong in the Dockerfile, not a live
  container, so the environment stays reproducible.

### Where ROS gets sourced

`/etc/ros2_env.sh` inside the image, pulled in from three places because each
kind of shell reads a different file:

| shell | reads |
| --- | --- |
| `./ros2.sh shell` (interactive) | `~/.bashrc` |
| `./ros2.sh run "..."` (`bash -lc`) | `/etc/profile.d/10-ros2.sh` |
| compose `command:` / `docker run` | `docker/entrypoint.sh` |

Ubuntu's stock `.bashrc` returns early when non-interactive, so it does not
cover `bash -lc`; and `docker exec` bypasses the entrypoint. All three hooks
are needed. Put environment changes in the `/etc/ros2_env.sh` heredoc in
`docker/Dockerfile`.
