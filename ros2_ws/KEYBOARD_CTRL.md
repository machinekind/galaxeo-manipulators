# Keyboard control for the Galaxea A1X

Drive the arm one joint at a time from your keyboard, using
[`keyboard_a1x.py`](keyboard_a1x.py) and GALAXEO's own ROS 2 driver.

> **The arm has no brakes.** Cutting power drops it — it falls under its own
> weight from wherever it is. Clear the space under it before you start, and
> keep a hand near the **power switch**, not the USB cable (unplugging USB
> leaves it powered and holding a stale setpoint). Full detail in
> [../docs/SAFETY.md](../docs/SAFETY.md).

Stopping is the safe direction. An uncommanded arm holds where it is: it does
not fall and it does not snap back. The risky moment is always the *start*.

---

## What you need

* Power cable (Probably already plugged in).
* The USB-CAN-FD adapter. It should be on the table or in the manipulator's box. It has an USB-A type end on one side with small encased electronic board and two visible diodes and a small unknown type end on the other side. 
* Either ROS2 installed or docker installed.

You will need **two terminals**. Steps 2 and 3 each keep a terminal busy.

---

## Before you start

* Make sure the power cable is plugged in, but the safety switch is turned off (for your safety). The safety switch is located on the black box near the floor.
* Connect the USB-CAN-FD adapter to the computer and manipulator. The USB-A type goes to the computer and the other end goes to the manipulator next to the USB input.
* If you connected the adapter turn on the safety switch on the power cable. The manipulator should make noises indicating it's working and it should constantly make this swoosh sound.
* If you dont have ros2 installed locally, install docker and run: 
```
docker compose up -d ros2
```

---

## Step 1 — Bring up the CAN bus

From the repo root:

```bash
./can_up.sh
```

This finds the arm's adapter, configures it at the right CAN-FD timings and
renames it to `can0`. Expected tail:

```
--- can0 ---
frames in 2s: 400  (~200 Hz)
OK: arm is streaming on can0.
```

If it says **no CAN traffic on any interface**, the arm is not powered or the
CAN cable is loose. Fix that before going on.

> Links come up **down** after every replug, so re-run `./can_up.sh` any time
> you unplug the adapter.

---

## Step 2 — Start the driver (terminal 1)

```bash
./ros2.sh shell
```

Then **inside the container**:

```bash
ros2 launch galaxea_a1xy_driver driver.launch.py command_can_id:=80
```

Wait for **both** of these lines, then leave the terminal alone:

```
[WARN] TRANSMIT ENABLED on CAN id 0x050 - arm may MOVE.
[INFO] feedback 200 Hz
```

`command_can_id:=80` is what allows the driver to transmit. Without it the
driver is read-only and the keyboard script will appear to do nothing.

---

## Step 3 — Run the keyboard script (terminal 2)

Open a second terminal:

```bash
./ros2.sh shell
```

Then inside the container, **dry run first** — this publishes nothing:

```bash
cd ~/ros2_ws && python3 keyboard_a1x.py --dry-run
```

Hold `y` and `h`. The `J6 wroll` numbers will not change (nothing is being
sent), but a `*` appears beside that joint showing your keys are registering.
Press `ESC` to quit.

Now for real — **make sure the space around the arm is clear**:

```bash
cd ~/ros2_ws && python3 keyboard_a1x.py
```

Start with **`y` / `h`**. That is J6, wrist roll: the lightest joint, lowest
inertia, and it cannot drop anything. Tap once for a single 0.5° step, or hold
the key to jog continuously.

> The script must run in a real terminal. It refuses to start otherwise, which
> is why you run it from `./ros2.sh shell` and not `./ros2.sh run "..."`.

---

## The keys

| key | joint | key | joint |
| --- | --- | --- | --- |
| `q` / `a` | J1 base yaw | `r` / `f` | J4 wrist pitch |
| `w` / `s` | J2 shoulder pitch | `t` / `g` | J5 wrist yaw |
| `e` / `d` | J3 elbow pitch | `y` / `h` | J6 wrist roll |

| key | what it does |
| --- | --- |
| `SPACE` | stop here — freeze at the arm's current position |
| `0` | go back to the pose it was in when you started |
| `[` / `]` | smaller / bigger step per keypress (0.1° to 2°) |
| `ESC` or `Ctrl-C` | quit, releasing the arm |

Each keypress adds one step to the target, and your terminal's key-repeat does
the rest when you hold a key. So the jog speed is *step size × repeat rate* —
raise it with `]` if the arm feels slow.

---

## Reading the display

One line, updated ten times a second:

```
J1 yaw   -9.07  J2 shldr  -0.05  J3 elbow  -5.89  J4 wpitch 15.95  J5 wyaw  1.58  J6 wroll* -3.72 | 0.50d/key
```

* The numbers are the **measured** joint angles in degrees.
* A `*` marks whichever joint is currently being driven.
* The right-hand value is the current step size.

**If a joint stops responding while you are still holding the key**, that is the
safety clamp: the target has run 5° ahead of where the arm actually is, which
means it is stalled — against its own limit, or against something solid. Let go
and look before pushing further.

---

## Stopping

* `SPACE` — freeze where it is.
* `ESC` or `Ctrl-C` in terminal 2 — quit and release the arm.
* `Ctrl-C` in terminal 1 — shut the driver down.

---

## When you are done

`Ctrl-C` does not reliably take the child process with it, and a leftover
transmit-enabled driver keeps the CAN bus claimed while being invisible to the
obvious checks. Always finish with:

```bash
./ros2.sh run "ps -eo pid,args | grep -E 'keyboard_a1x|driver_node' | grep -v grep"
```

Anything listed:

```bash
./ros2.sh run "kill -9 <pid>"
```

---

## If something goes wrong

| what you see | what it means |
| --- | --- |
| `no feedback on /hdas/feedback_arm` | the driver in terminal 1 is not running |
| `stdin is not a TTY` | use `./ros2.sh shell`, not `./ros2.sh run "..."` |
| keys register (`*` appears) but the arm never moves | the driver was started without `command_can_id:=80`, or you left `--dry-run` on |
| `feedback stale -- holding` | frames stopped arriving; check the arm is still powered |
| a joint will not move in one direction | it is at its limit — check the display against the limits in [../docs/HARDWARE.md](../docs/HARDWARE.md) |
| nothing works after replugging USB | re-run `./can_up.sh`; links come up down after a replug |
| the arm ignores commands entirely | it needs re-enabling — function frames `1 → 5 → 6`. A freshly power-cycled arm obeys immediately. See [../docs/STEERING.md](../docs/STEERING.md) |

**Only one process may transmit on the bus at a time.** Two writers on `0x050`
means the arm receives them interleaved and tracks neither. Do not run
`keyboard_a1x.py` and the web UI (`webui_a1x.py`) at the same time.

---

## Options

```bash
python3 keyboard_a1x.py --help
```

| flag | default | what it does |
| --- | --- | --- |
| `--step` | `0.5` | degrees added per keypress (0.1 – 2.0) |
| `--dry-run` | off | show the targets, publish nothing |

---

## Why it behaves the way it does

Three things about this arm shape the script; they are not arbitrary choices.

**It streams continuously.** The driver puts one CAN frame on the wire per
message it receives, so the script publishes at 100 Hz the whole time it runs,
not only when you press a key. An arm that stops receiving just latches where
it is — safe, but it will not track.

**Keys add to a target instead of setting a speed.** The usual
velocity-with-a-deadman approach has to outlive your terminal's ~500 ms
auto-repeat delay to feel smooth, which means it also keeps driving for that
long after you let go. Accumulating steps instead means releasing a key stops
the motion within one control cycle, with no overshoot.

**The target can never lead the arm by more than 5°.** If the arm stalls, the
target stops running away rather than banking up a snap for when it comes free.
On an arm with no compliance — `t_ff`, `kp` and `kd` are all inert on this
transport, so it will not yield if it drives into something — this is the guard
that matters most.
