#!/usr/bin/env python3
"""Read an SO-101 leader over its Feetech serial bus. pyserial only, no LeRobot.

The leader is six STS3215 servos on one half-duplex bus behind a USB adapter
(macOS: /dev/cu.usbmodem*, Linux: /dev/ttyACM*). With torque off they are
free to move by hand and they keep reporting position, which is all a leader
needs. This module reads them; it never enables torque and never writes a
register.

    python so101_feetech.py                    # print positions live, Ctrl-C stops
    python so101_feetech.py --calibrate-gripper  # hold it closed, then open: 6 s
    python so101_feetech.py --calibrate          # every joint end to end (ik mode)

UNITS
    Present_Position is 0..4095 ticks per turn, so one tick is 2*pi/4096 rad.
    Joints are returned in radians about the middle of the calibrated range
    (2048 ticks when there is no calibration), which is the convention of
    so101/so101_new_calib.urdf: zero = mid-range. For a RELATIVE mapping only
    the tick scale matters, so joint teleop works without any calibration.
    The gripper is returned as a 0..100 opening and DOES need the calibration,
    because nothing else says which end of its travel is closed.

CALIBRATION FILE  so101/calib_<id>.json
    {"<joint>": {"min": ticks, "max": ticks}, ..., "gripper": {..., "closed": ticks}}
    --calibrate-gripper records only the gripper entry (closed first, then
    open) and leaves any joint entries alone. --calibrate does every joint
    end to end as well, which only --mode ik needs.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import platform
import sys
import time

try:
    import serial
except ImportError:                     # reported at connect time, not import time
    serial = None

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
       "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
ARM = [n for n in IDS if n != "gripper"]
TICKS = 4096
RAD_PER_TICK = 2.0 * math.pi / TICKS
BAUD = 1_000_000
REG_PRESENT_POSITION = 0x38             # 2 bytes, little-endian
INSTR_PING, INSTR_READ, INSTR_SYNC_READ = 0x01, 0x02, 0x82


def default_port() -> str:
    """One USB serial adapter plugged in -> that one. Otherwise the caller
    has to say."""
    pat = "/dev/cu.usbmodem*" if platform.system() == "Darwin" else "/dev/ttyACM*"
    found = sorted(glob.glob(pat))
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit(f"no SO-101 serial port ({pat}). Is the leader's USB cable in?")
    raise SystemExit(f"several serial ports match {pat}: {found}. Pass --port.")


def calib_path(cal_id: str) -> str:
    return os.path.join(HERE, "so101", f"calib_{cal_id}.json")


def _packet(sid: int, instr: int, params: bytes = b"") -> bytes:
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


class FeetechBus:
    """The three reads a leader needs: ping, one position, all positions."""

    def __init__(self, port: str | None = None, baud: int = BAUD):
        if serial is None:
            raise SystemExit("pyserial is missing:  uv pip install pyserial")
        self.port = port or default_port()
        self.s = serial.Serial(self.port, baud, timeout=0.02)

    def close(self):
        try:
            self.s.close()
        except Exception:
            pass

    def _status(self, want_id: int, n_params: int) -> bytes | None:
        """One status packet: FF FF ID LEN ERR <params> CHK. Resyncs on the
        header so a stray byte on the half-duplex line does not poison the
        rest of the read."""
        hdr = self.s.read(2)
        while hdr and hdr != b"\xff\xff":
            nxt = self.s.read(1)
            if not nxt:
                return None
            hdr = hdr[1:] + nxt
        if len(hdr) < 2:
            return None
        rest = self.s.read(3 + n_params + 1)
        if len(rest) < 3 + n_params + 1:
            return None
        sid, ln, err = rest[0], rest[1], rest[2]
        if sid != want_id or ln != n_params + 2:
            return None
        chk = (~(sum(rest[:-1]))) & 0xFF
        if chk != rest[-1]:
            return None
        return rest[3:3 + n_params]

    def ping(self, sid: int) -> bool:
        self.s.reset_input_buffer()
        self.s.write(_packet(sid, INSTR_PING))
        return self._status(sid, 0) is not None

    def read_u16(self, sid: int, reg: int) -> int | None:
        self.s.reset_input_buffer()
        self.s.write(_packet(sid, INSTR_READ, bytes([reg, 2])))
        p = self._status(sid, 2)
        return None if p is None else p[0] | (p[1] << 8)

    def sync_positions(self, ids: list[int]) -> dict[int, int]:
        """Present_Position of every id in one bus transaction. Missing
        replies are simply absent from the result; the caller decides."""
        self.s.reset_input_buffer()
        self.s.write(_packet(0xFE, INSTR_SYNC_READ,
                             bytes([REG_PRESENT_POSITION, 2]) + bytes(ids)))
        out = {}
        for sid in ids:
            p = self._status(sid, 2)
            if p is not None:
                out[sid] = p[0] | (p[1] << 8)
        return out


class SO101Raw:
    """Same contract as the LeRobot-backed reader in so101_bridge.py:
    read() -> {'<joint>.pos': value}. Joints in RADIANS (mid-range = 0),
    gripper in 0..100 (0 = closed), or absent when it cannot be known."""
    units = "rad"

    def __init__(self, port: str | None = None, cal_id: str = "my_leader",
                 allow_missing: bool = False):
        self.bus = FeetechBus(port)
        self.port = self.bus.port
        self.present = [n for n, i in IDS.items() if self.bus.ping(i)]
        self.missing = [n for n in IDS if n not in self.present]
        if self.missing and not allow_missing:
            raise RuntimeError(
                f"SO-101 motors not responding on {self.port}: {self.missing}. "
                f"Check the cable and power at those servos, or pass --allow-missing.")
        self.cal = None
        path = calib_path(cal_id)
        if os.path.exists(path):
            with open(path) as f:
                self.cal = json.load(f)
        self.bad = 0

    @property
    def joints_calibrated(self) -> bool:
        return bool(self.cal) and all(n in self.cal for n in ARM)

    @property
    def gripper_ok(self) -> bool:
        return "gripper" in self.present and bool(self.cal) and "gripper" in self.cal

    def _mid(self, name: str) -> float:
        if self.cal and name in self.cal:
            c = self.cal[name]
            return 0.5 * (c["min"] + c["max"])
        return TICKS / 2

    def convert(self, ticks: dict[int, int]) -> dict[str, float]:
        out = {}
        for name, sid in IDS.items():
            if sid not in ticks:
                continue
            t = ticks[sid]
            if name == "gripper":
                if not self.gripper_ok:
                    continue
                c = self.cal["gripper"]
                closed, lo, hi = c["closed"], c["min"], c["max"]
                opened = hi if abs(closed - lo) < abs(closed - hi) else lo
                span = opened - closed
                f = (t - closed) / span if span else 0.0
                out["gripper.pos"] = 100.0 * min(1.0, max(0.0, f))
            else:
                out[f"{name}.pos"] = (t - self._mid(name)) * RAD_PER_TICK
        return out

    def read(self, retries: int = 3) -> dict[str, float]:
        ids = [IDS[n] for n in self.present]
        last = None
        for _ in range(retries):
            try:
                ticks = self.bus.sync_positions(ids)
                if len(ticks) == len(ids):
                    self.bad = 0
                    return self.convert(ticks)
                last = f"replies from {sorted(ticks)} of {ids}"
            except Exception as ex:
                last = ex
            time.sleep(0.002)
        self.bad += 1
        raise RuntimeError(f"SO-101 bus read failed {retries}x: {last}")

    def close(self):
        self.bus.close()


def _hold(bus: FeetechBus, what: str, secs: float = 2.0) -> int:
    print(f"  HOLD THE GRIPPER {what} ... {secs:g} s")
    time.sleep(1.0)
    got = []
    t0 = time.time()
    while time.time() - t0 < secs:
        p = bus.sync_positions([IDS["gripper"]])
        if IDS["gripper"] in p:
            got.append(p[IDS["gripper"]])
        time.sleep(0.02)
    if not got:
        raise SystemExit("  no gripper reading; calibration NOT saved")
    t = int(round(sum(got) / len(got)))
    print(f"    {what.lower()} = {t} ticks")
    return t


def _save(cal_id: str, entries: dict):
    """Merge into the existing file so a gripper-only calibration keeps
    joint ranges recorded earlier, and vice versa."""
    path = calib_path(cal_id)
    cal = {}
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
    cal.update(entries)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"  saved {path}")


def calibrate_gripper(bus: FeetechBus, cal_id: str):
    closed = _hold(bus, "CLOSED")
    opened = _hold(bus, "FULLY OPEN")
    if abs(opened - closed) < 100:
        raise SystemExit(f"  closed {closed} and open {opened} are too close; "
                         f"did the gripper move? NOT saved")
    _save(cal_id, {"gripper": {"min": min(closed, opened),
                               "max": max(closed, opened), "closed": closed}})


def calibrate(bus: FeetechBus, cal_id: str, secs: float):
    ids = list(IDS.values())
    lo = {i: TICKS for i in ids}
    hi = {i: 0 for i in ids}
    print(f"  {secs:g}s: move EVERY joint end to end, gripper included, "
          f"slowly, a few times.")
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < secs:
        p = bus.sync_positions(ids)
        for i, t in p.items():
            lo[i] = min(lo[i], t)
            hi[i] = max(hi[i], t)
        if time.time() - last > 0.5:
            last = time.time()
            left = secs - (time.time() - t0)
            print(f"    {left:4.0f}s  " + "  ".join(
                f"{n[:7]}:{lo[i]:4d}-{hi[i]:4d}" for n, i in IDS.items()))
        time.sleep(0.02)
    print()
    closed_t = _hold(bus, "CLOSED")
    cal = {n: {"min": lo[i], "max": hi[i]} for n, i in IDS.items()}
    cal["gripper"]["closed"] = closed_t
    narrow = [n for n, i in IDS.items() if hi[i] - lo[i] < 200]
    if narrow:
        print(f"  WARNING tiny range on {narrow} -- did those joints move?")
    _save(cal_id, cal)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default=None, help="serial port (default: the one adapter found)")
    ap.add_argument("--cal-id", default="my_leader")
    ap.add_argument("--calibrate", action="store_true", help="every joint + gripper")
    ap.add_argument("--calibrate-gripper", action="store_true",
                    help="gripper only: hold closed, then open")
    ap.add_argument("--secs", type=float, default=20.0, help="calibration recording time")
    a = ap.parse_args()

    if a.calibrate or a.calibrate_gripper:
        bus = FeetechBus(a.port)
        print(f"  {bus.port}: servos answering: "
              f"{[n for n, i in IDS.items() if bus.ping(i)]}")
        try:
            if a.calibrate:
                calibrate(bus, a.cal_id, a.secs)
            else:
                calibrate_gripper(bus, a.cal_id)
        finally:
            bus.close()
        return 0

    lead = SO101Raw(a.port, a.cal_id, allow_missing=True)
    print(f"  {lead.port}: present {lead.present}"
          + (f"  MISSING {lead.missing}" if lead.missing else ""))
    print(f"  calibration: {'loaded' if lead.cal else 'none (joints relative only, no gripper)'}")
    print("  Ctrl-C stops.\n")
    try:
        while True:
            r = lead.read()
            js = "  ".join(f"{n[:7]}:{math.degrees(r[f'{n}.pos']):6.1f}"
                           for n in ARM if f"{n}.pos" in r)
            g = f"  grip:{r['gripper.pos']:5.1f}%" if "gripper.pos" in r else "  grip: n/a"
            print(f"\r  {js}{g}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        lead.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
