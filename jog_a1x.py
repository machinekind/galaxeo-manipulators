#!/usr/bin/env python3
"""On-screen jog for the Galaxea A1X: hold a key or a button, the joint moves.

Works on macOS (the XCAN / PEAK USB-CAN FD adapter through xcan_usb.py, our
own libusb driver) and on Linux (SocketCAN via python-can). No ROS, no SO-101,
no vendor binaries.

    python jog_a1x.py --check              # read-only: is the arm streaming?
    python jog_a1x.py --dry-run            # the GUI, transmits nothing
    python jog_a1x.py                      # the GUI, ARM button turns TX on

Keys (hold to move, release to stop):

    J1 base yaw       q / a        J4 wrist pitch   r / f
    J2 shoulder       w / s        J5 wrist yaw     t / g
    J3 elbow          e / d        J6 wrist roll    y / h
    gripper           o = open     c = close (force-limited)
    space / Esc       STOP: drop every jog and disarm

The on-screen [-] [+] buttons do the same thing with the mouse. "Go home"
slews every joint to --home (default all zeros: the folded pose) at
--home-speed, 9 deg/s by default; any jog key cancels it, and it reports
"safe to disarm" once there. "Set home = here" takes the measured pose as
home and prints the --home value to reuse. Prefer that over the zeros: a
joint commanded even 0.2 deg into its mechanical stop pushes there forever,
visible as steady effort at rest.

How it moves the arm, and why it is safe to start:

  * The setpoint is seeded from the MEASURED pose the moment you arm, and only
    ever moves at the speed slider's rate (deg/s). Nothing jumps.
  * While armed it streams p_des at 200 Hz. Releasing a key holds the target
    where it is; the arm holds. Disarming stops the stream; an uncommanded A1X
    re-latches where it is (see docs/SAFETY.md).
  * Feedback older than 150 ms while armed disarms immediately.
  * Targets are clamped to the URDF limits (widened to include wherever the
    arm actually started, since J3 reads ~1.5 deg past its limit at rest).
  * The gripper's p_des cannot be read back from the arm, so the first
    gripper key press starts it from --grip-start (open) and jogs from there.
    Closing freezes when |effort| exceeds --grip-force: grips, doesn't crush.

Enable (function frames 1 -> 5 -> 6) is a button, off by default: a
power-cycled arm obeys 0x050 immediately and FF 5 briefly disengages the
motors. Use it only if the arm reports but ignores the jog. The setpoint is
pinned to the measured pose throughout the sequence, as docs/SAFETY.md demands.

Requirements: python-can (Linux), pyusb + libusb (macOS: brew install libusb).
"""
import argparse
import math
import platform
import sys
import threading
import time

try:
    import can
except ImportError:                      # only needed for the python-can backends
    can = None

N = 6
S_POS, S_VEL, S_EFF = 4700.0, 750.0, 600.0
FIELDS = ((-6.5, 6.5, 4700.0), (-40.0, 40.0, 750.0), (0.0, 500.0, 60.0),
          (0.0, 200.0, 150.0), (-50.0, 50.0, 600.0))
LIMITS = [(-2.880, 2.880), (0.0, 3.142), (-3.316, 0.0),
          (-1.571, 1.571), (-1.571, 1.571), (-2.880, 2.880)]
NAMES = ["J1 base yaw", "J2 shoulder", "J3 elbow",
         "J4 wrist pitch", "J5 wrist yaw", "J6 wrist roll"]
CMD_ID, GRIP_ID, FB_ID, FF_ID = 0x050, 0x051, 0x052, 0x053
KEYS = {"q": (0, +1), "a": (0, -1), "w": (1, +1), "s": (1, -1),
        "e": (2, +1), "d": (2, -1), "r": (3, +1), "f": (3, -1),
        "t": (4, +1), "g": (4, -1), "y": (5, +1), "h": (5, -1),
        "o": (6, -1), "c": (6, +1)}          # gripper: p_des more negative = open

# PEAK-USB FD, 80 MHz clock: 1 Mbit/s and 5 Mbit/s, both at sample point 0.875
# (the vendor's own timings, see docs/HARDWARE.md).
PCAN_FD_TIMING = dict(f_clock_mhz=80, nom_brp=1, nom_tseg1=69, nom_tseg2=10, nom_sjw=10,
                      data_brp=1, data_tseg1=13, data_tseg2=2, data_sjw=2)


def encode_group(p, v, kp, kd, tff):
    out = bytearray(10)
    for k, val in enumerate((p, v, kp, kd, tff)):
        lo, hi, sc = FIELDS[k]
        x = lo if val < lo else (hi if val > hi else val)
        raw = max(-32768, min(32767, int(x * sc)))
        out[k * 2] = (raw >> 8) & 0xFF
        out[k * 2 + 1] = raw & 0xFF
    return bytes(out)


def open_bus(iface):
    """Backends, by --iface:

        xcan[:<usb addr>]   macOS only: our libusb driver, xcan_usb.py
        can0, can1, ...     Linux: SocketCAN through python-can (bring it up
                            with ./can_up.sh first)
        PCAN_USBBUSn        Windows: PEAK's PCAN-Basic through python-can
        <interface>:<chan>  any other python-can interface, e.g. virtual:x
    """
    if iface.startswith("xcan"):
        if platform.system() != "Darwin":
            raise SystemExit("xcan_usb.py is the macOS driver. On Linux use SocketCAN "
                             "(./can_up.sh, then --iface can0).")
        from xcan_usb import XcanBus
        addr = int(iface.split(":", 1)[1]) if ":" in iface else None
        return XcanBus(addr)
    if can is None:
        raise SystemExit("python-can is missing:  pip install python-can")
    if iface.upper().startswith("PCAN"):
        return can.Bus(interface="pcan", channel=iface, fd=True, **PCAN_FD_TIMING)
    if ":" in iface:
        interface, channel = iface.split(":", 1)
        return can.Bus(interface=interface, channel=channel, fd=True)
    return can.Bus(interface="socketcan", channel=iface, fd=True)


class A1X:
    """Feedback decode plus the three frames the arm listens to."""

    def __init__(self, iface, dry_run):
        self.bus = open_bus(iface)
        if iface.startswith("xcan"):
            from xcan_usb import Message
            self._msg = lambda **kw: Message(kw["arbitration_id"], kw["data"], kw["is_fd"], kw["bitrate_switch"])
        else:
            self._msg = can.Message
        self.dry_run = dry_run
        self.q = None; self.v = None; self.e = None
        self.t = 0.0; self.n = 0; self.n_tx = 0
        self.same = 0; self._last = None
        self.lock = threading.Lock()

    def drain(self):
        """Read EVERY queued frame; a partial drain makes readings seconds stale."""
        got = False
        while True:
            m = self.bus.recv(0.0)
            if m is None:
                break
            if m.arbitration_id != FB_ID or len(m.data) < 42:
                continue
            d = bytes(m.data)
            r = [int.from_bytes(d[i:i + 2], "big", signed=True) for i in range(0, 42, 2)]
            with self.lock:
                self.q = [r[g * 3] / S_POS for g in range(7)]
                self.v = [r[g * 3 + 1] / S_VEL for g in range(7)]
                self.e = [r[g * 3 + 2] / S_EFF for g in range(7)]
                self.same = self.same + 1 if d == self._last else 0
                self._last = d
                self.t = time.time(); self.n += 1
            got = True
        return got

    def _tx(self, cid, payload, fd):
        if self.dry_run:
            return
        self.bus.send(self._msg(arbitration_id=cid, data=payload, is_fd=fd,
                                bitrate_switch=fd, is_extended_id=False), timeout=0.02)
        self.n_tx += 1

    def send_arm(self, p, kp, kd):
        payload = b"".join(encode_group(p[j], 0.0, kp, kd, 0.0) for j in range(N))
        self._tx(CMD_ID, payload.ljust(64, b"\x00"), fd=True)

    def send_grip(self, p, kp, kd):
        self._tx(GRIP_ID, encode_group(p, 0.0, kp, kd, 0.0).ljust(12, b"\x00"), fd=True)

    def send_ff(self, code):
        self._tx(FF_ID, bytes([code]).ljust(8, b"\x00"), fd=False)

    def close(self):
        self.bus.shutdown()


class Jog:
    """The control loop: runs in its own thread at --rate Hz, GUI-agnostic."""

    def __init__(self, arm, a):
        self.arm = arm; self.a = a
        self.armed = False
        self.target = None                      # 6 joint setpoints, rad
        self.bounds = None
        self.vel = [0] * 7                      # -1 / 0 / +1 per axis, 7 = gripper
        self.grip_t = None                      # None until first gripper key
        self.grip_frozen = False
        self.speed = a.speed                    # deg/s, from the slider
        self.homing = False                     # Go-home in progress (slow, all joints)
        self.status = "read-only"
        self.hz = 0.0
        self._stop = False
        self._enable_req = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---- commands from the GUI ------------------------------------------
    def arm_on(self):
        with self.lock:
            if self.arm.q is None or time.time() - self.arm.t > 0.15:
                self.status = "cannot arm: no fresh feedback"; return False
            if self.arm.same > 50:
                self.status = "cannot arm: telemetry FROZEN (released state, FF 2)"; return False
            q = list(self.arm.q[:N])
            self.target = q
            self.bounds = [(min(lo, q[j]), max(hi, q[j])) for j, (lo, hi) in enumerate(LIMITS)]
            self.grip_t = None; self.grip_frozen = False
            self.vel = [0] * 7
            self.armed = True
            self.status = "ARMED (dry-run, no TX)" if self.arm.dry_run else "ARMED, streaming"
            return True

    def arm_off(self, why="disarmed"):
        with self.lock:
            self.armed = False; self.vel = [0] * 7; self.homing = False
            self.status = why

    def set_vel(self, axis, sign):
        with self.lock:
            self.vel[axis] = sign
            if sign and self.homing:
                self.homing = False; self.status = "home cancelled by jog"
            if axis == 6 and sign != 0 and self.grip_t is None:
                self.grip_t = self.a.grip_start

    def request_enable(self):
        with self.lock:
            if not self.armed:
                self.status = "arm first, then enable"; return
            self._enable_req = True

    def set_home_here(self):
        """Take the measured pose as this session's home, and print it so it can
        be passed as --home next time. Use it in the pose the arm rests in
        naturally: commanding a joint even 0.2 deg into its mechanical stop
        makes the motor push there forever, which shows up as effort."""
        with self.lock:
            if self.arm.q is None or time.time() - self.arm.t > 0.15:
                self.status = "no fresh feedback"; return
            self.a.home_rad = list(self.arm.q[:N])
            self.homing = False
            deg = ",".join(f"{math.degrees(x):.1f}" for x in self.a.home_rad)
            self.status = f"home set to here. Next time: --home {deg}"
            print(f"home = current pose. Next time start with:  --home {deg}")

    def go_home(self):
        """Slew every joint to --home at --home-speed. Any jog key cancels it.
        The gripper is left alone."""
        with self.lock:
            if not self.armed:
                self.status = "arm first, then go home"; return
            self.vel = [0] * 7
            self.homing = True

    def stop(self):
        self._stop = True
        self.thread.join(1.0)

    # ---- the loop ----------------------------------------------------------
    def _loop(self):
        a = self.a; period = 1.0 / a.rate
        nxt = time.time(); prev = nxt; win_t = nxt; win_n = 0
        while not self._stop:
            try:
                self.arm.drain()
            except Exception as ex:                       # adapter unplugged, link down
                self.arm_off(f"ABORT: bus error: {ex}"); time.sleep(0.5); continue
            now = time.time()
            if now - win_t >= 0.5:
                self.hz = (self.arm.n - win_n) / (now - win_t); win_t = now; win_n = self.arm.n
            if now < nxt:
                time.sleep(min(0.001, nxt - now)); continue
            nxt += period
            if nxt < now - 0.05:
                nxt = now                                 # fell behind, do not burst
            dt = min(0.05, now - prev); prev = now

            with self.lock:
                if not self.armed:
                    continue
                if now - self.arm.t > 0.15:
                    self.armed = False; self.vel = [0] * 7
                    self.status = f"ABORT: stale feedback ({(now - self.arm.t) * 1e3:.0f} ms)"
                    continue
                step = math.radians(self.speed) * dt
                for j in range(N):
                    if self.vel[j]:
                        lo, hi = self.bounds[j]
                        self.target[j] = min(hi, max(lo, self.target[j] + self.vel[j] * step))
                if self.vel[6] and self.grip_t is not None:
                    closing = self.vel[6] > 0
                    eff = abs(self.arm.e[6]) if self.arm.e else 0.0
                    if closing and eff > a.grip_force:
                        self.grip_frozen = True           # in contact: hold, don't crush
                    elif not closing:
                        self.grip_frozen = False
                    if not self.grip_frozen:
                        g = self.grip_t + self.vel[6] * a.grip_speed * dt
                        self.grip_t = min(a.grip_closed, max(a.grip_start, g))
                if self.homing:
                    step_h = math.radians(a.home_speed) * dt
                    worst = 0.0
                    for j in range(N):
                        lo, hi = self.bounds[j]
                        d = min(hi, max(lo, a.home_rad[j])) - self.target[j]
                        worst = max(worst, abs(d))
                        self.target[j] += max(-step_h, min(step_h, d))
                    if worst < math.radians(0.05):
                        self.homing = False
                        self.status = "at home: holding. Safe to disarm."
                    else:
                        self.status = f"going home at {a.home_speed:g} deg/s, {math.degrees(worst):.1f} deg to go"
                target = list(self.target); grip_t = self.grip_t
                enable = self._enable_req; self._enable_req = False

            try:
                if enable:
                    self._enable(target)
                self.arm.send_arm(target, a.kp, a.kd)
                if grip_t is not None:
                    self.arm.send_grip(grip_t, a.grip_kp, a.kd)
            except Exception as ex:
                self.arm_off(f"ABORT: TX failed: {ex}")

    def _enable(self, target):
        """FF 1 -> 5 -> 6 with p_des pinned to the target before, during, after.

        Enabling with nothing streamed once swung an arm 76 deg at saturated
        torque (docs/SAFETY.md), hence the streaming around every code.
        """
        with self.lock:
            self.status = "enabling: FF 1 -> 5 -> 6 (motors disengage briefly)"
        for code in (1, 5, 6):
            t0 = time.time()
            while time.time() - t0 < 0.3:
                self.arm.drain(); self.arm.send_arm(target, self.a.kp, self.a.kd)
                time.sleep(0.005)
            self.arm.send_ff(code)
        t0 = time.time()
        while time.time() - t0 < 0.3:
            self.arm.drain(); self.arm.send_arm(target, self.a.kp, self.a.kd)
            time.sleep(0.005)
        with self.lock:
            self.status = "enable sequence sent; jog J1 to test"


def check(arm, secs):
    """Read-only: report feedback rate and pose, transmit nothing."""
    print(f"listening {secs:g}s on {arm.bus.channel_info} ... (transmits nothing)")
    t0 = time.time()
    while time.time() - t0 < secs:
        arm.drain(); time.sleep(0.002)
    if arm.n == 0:
        print("NO 0x052 FEEDBACK. The adapter is up but the arm is not talking:\n"
              "  * is the 24 V supply on?\n"
              "  * is the CAN cable seated at both ends (CAN_H / CAN_L / GND)?\n"
              "  * termination R1 / R2 both up on the CAN box?")
        return 2
    print(f"{arm.n / secs:.0f} Hz")
    print("q (deg):   ", [round(math.degrees(x), 1) for x in arm.q[:N]], " gripper grp7:", round(math.degrees(arm.q[6]), 1))
    print("effort:    ", [round(x, 2) for x in arm.e])
    if arm.same > 50:
        print("payload FROZEN: the arm is in the released state (FF 2). Enable needed.")
    else:
        print("OK: reporting live.")
    return 0


def gui(jog, a):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk(); root.title("A1X jog" + (" (dry-run)" if a.dry_run else ""))
    root.resizable(False, False)
    pad = dict(padx=4, pady=2)

    top = ttk.Frame(root); top.grid(row=0, column=0, sticky="ew", **pad)
    status = tk.StringVar(value="read-only")
    ttk.Label(top, textvariable=status, font=("TkDefaultFont", 12, "bold")).pack(side="left")

    ctl = ttk.Frame(root); ctl.grid(row=1, column=0, sticky="ew", **pad)
    arm_btn = ttk.Button(ctl, text="ARM (start streaming)", command=lambda: jog.arm_on())
    arm_btn.pack(side="left", padx=2)
    ttk.Button(ctl, text="STOP / disarm", command=lambda: jog.arm_off("disarmed")).pack(side="left", padx=2)
    ttk.Button(ctl, text="Enable FF 1→5→6", command=jog.request_enable).pack(side="left", padx=2)
    ttk.Button(ctl, text=f"Go home ({a.home_speed:g}°/s)", command=jog.go_home).pack(side="left", padx=2)
    ttk.Button(ctl, text="Set home = here", command=jog.set_home_here).pack(side="left", padx=2)
    ttk.Label(ctl, text="speed deg/s").pack(side="left", padx=(12, 2))
    speed = tk.DoubleVar(value=a.speed)
    ttk.Scale(ctl, from_=1, to=a.max_speed, variable=speed, length=140,
              command=lambda v: setattr(jog, "speed", float(v))).pack(side="left")
    speed_lbl = ttk.Label(ctl, width=4); speed_lbl.pack(side="left")

    grid = ttk.Frame(root); grid.grid(row=2, column=0, **pad)
    for c, h in enumerate(("joint", "keys", "", "", "measured", "target", "effort")):
        ttk.Label(grid, text=h, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=c, padx=6)
    rows = []
    keymap = {(ax, s): k for k, (ax, s) in KEYS.items()}
    for i, name in enumerate(NAMES + ["gripper"]):
        r = i + 1
        ttk.Label(grid, text=name, anchor="w", width=14).grid(row=r, column=0, sticky="w")
        ttk.Label(grid, text=f"{keymap[(i, -1)]} / {keymap[(i, +1)]}").grid(row=r, column=1)
        for col, sign, txt in ((2, -1, "open" if i == 6 else "−"), (3, +1, "close" if i == 6 else "+")):
            b = ttk.Button(grid, text=txt, width=5)
            b.grid(row=r, column=col, padx=1)
            b.bind("<ButtonPress-1>", lambda ev, ax=i, s=sign: jog.set_vel(ax, s))
            b.bind("<ButtonRelease-1>", lambda ev, ax=i: jog.set_vel(ax, 0))
        meas = ttk.Label(grid, width=9, anchor="e"); meas.grid(row=r, column=4)
        tgt = ttk.Label(grid, width=9, anchor="e"); tgt.grid(row=r, column=5)
        eff = ttk.Label(grid, width=7, anchor="e"); eff.grid(row=r, column=6)
        rows.append((meas, tgt, eff))

    foot = ttk.Frame(root); foot.grid(row=3, column=0, sticky="ew", **pad)
    info = tk.StringVar(); ttk.Label(foot, textvariable=info).pack(side="left")

    # Key auto-repeat fires Release/Press pairs while a key is held; a release
    # only counts if no press for the same key follows within 80 ms.
    pending = {}

    def on_press(ev):
        k = ev.keysym.lower()
        if k == "space" or k == "escape":
            jog.arm_off("STOP"); return
        if k in KEYS:
            pending.pop(k, None)
            jog.set_vel(*KEYS[k])

    def on_release(ev):
        k = ev.keysym.lower()
        if k in KEYS:
            if k in pending:
                root.after_cancel(pending[k])
            pending[k] = root.after(80, lambda k=k: (pending.pop(k, None), jog.set_vel(KEYS[k][0], 0)))

    root.bind("<KeyPress>", on_press); root.bind("<KeyRelease>", on_release)

    def refresh():
        arm = jog.arm
        with arm.lock:
            q, e, t, same = arm.q, arm.e, arm.t, arm.same
        with jog.lock:
            target, grip_t, armed = (list(jog.target) if jog.target else None), jog.grip_t, jog.armed
            st = jog.status
        status.set(st)
        speed_lbl.config(text=f"{speed.get():.0f}")
        age = (time.time() - t) if q else float("inf")
        fresh = age < 0.15
        for j, (meas, tgt, eff) in enumerate(rows):
            if q:
                if j < N:
                    meas.config(text=f"{math.degrees(q[j]):8.1f}°")
                    tgt.config(text=f"{math.degrees(target[j]):8.1f}°" if target and armed else "")
                else:
                    meas.config(text=f"{math.degrees(q[6]):8.1f}°")
                    tgt.config(text=f"{grip_t:+.2f}" if grip_t is not None and armed else "")
                eff.config(text=f"{e[j]:6.2f}")
            else:
                meas.config(text="—"); tgt.config(text=""); eff.config(text="")
        fb = ("FROZEN payload (released)" if q and same > 50 else
              f"{jog.hz:5.0f} Hz" if fresh else ("no feedback" if not q else f"STALE {age * 1e3:.0f} ms"))
        info.set(f"feedback: {fb}    tx frames: {arm.n_tx}    keys: hold to jog, space = stop")
        arm_btn.state(["!disabled" if fresh and not armed else "disabled"])
        root.after(50, refresh)

    refresh()
    root.protocol("WM_DELETE_WINDOW", lambda: (jog.arm_off("closing"), root.destroy()))
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="xcan" if platform.system() == "Darwin" else "can0",
                    help="xcan[:<usb addr>] (macOS, libusb driver), a SocketCAN name like can0 "
                         "(Linux), PCAN_USBBUSn (Windows), or <python-can interface>:<channel>")
    ap.add_argument("--check", action="store_true", help="read-only feedback report, then exit")
    ap.add_argument("--dry-run", action="store_true", help="run the GUI but transmit nothing")
    ap.add_argument("--rate", type=float, default=200.0, help="stream rate, Hz")
    ap.add_argument("--speed", type=float, default=8.0, help="initial jog speed, deg/s")
    ap.add_argument("--home", default="0,0,0,0,0,0",
                    help="Go-home pose, six joint angles in degrees. All zeros is the URDF "
                         "zero: shoulder and elbow at their limits, i.e. the arm folded")
    ap.add_argument("--home-speed", type=float, default=9.0, help="Go-home slew, deg/s per joint")
    ap.add_argument("--max-speed", type=float, default=45.0, help="slider ceiling, deg/s")
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--grip-kp", type=float, default=20.0)
    ap.add_argument("--grip-start", type=float, default=-2.0,
                    help="gripper p_des the first gripper key starts from (fully open)")
    ap.add_argument("--grip-closed", type=float, default=0.6, help="gripper p_des fully closed")
    ap.add_argument("--grip-speed", type=float, default=1.0, help="gripper p_des units per second")
    ap.add_argument("--grip-force", type=float, default=1.2,
                    help="stop closing above this |effort|: grips, doesn't crush")
    a = ap.parse_args()
    try:
        a.home_rad = [math.radians(float(x)) for x in a.home.split(",")]
        assert len(a.home_rad) == N
    except (ValueError, AssertionError):
        sys.exit("--home needs six comma-separated angles in degrees")

    try:
        arm = A1X(a.iface, dry_run=a.dry_run or a.check)
    except Exception as ex:
        sys.exit(f"cannot open {a.iface}: {ex}\n"
                 + ("  macOS: is the adapter plugged in, and libusb installed (brew install libusb)?"
                    if a.iface.startswith("xcan") else "  Linux: ./can_up.sh first, then --iface can0"))
    try:
        if a.check:
            return check(arm, 3.0)
        jog = Jog(arm, a)
        try:
            gui(jog, a)
        finally:
            jog.stop()
    finally:
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
