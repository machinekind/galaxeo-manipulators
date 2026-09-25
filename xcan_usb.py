#!/usr/bin/env python3
"""Userspace PCAN-USB FD driver over libusb, read-only proof of concept.

Speaks the uCAN protocol the Linux kernel's peak_usb driver uses for the
PCAN-USB FD (and the XCAN clones that report the same 0c72:0012 id), so the
arm can be read from macOS without SocketCAN and without the PCBUSB library.
Every command and record layout below is transcribed from
drivers/net/can/usb/peak_usb/pcan_usb_fd.c and include/linux/can/dev/peak_canfd.h.

    python xcan_usb.py            # listen-only, decode 0x052, 5 s
    python xcan_usb.py --secs 20 --addr 6

Requires pyusb and libusb (brew install libusb).
"""
import argparse
import math
import struct
import sys
import time

import usb.core
import usb.util
import usb.backend.libusb1

VID, PID = 0x0C72, 0x0012
EP_CMD_OUT, EP_CMD_IN, EP_MSG_OUT, EP_MSG_IN = 0x01, 0x81, 0x02, 0x82

# uCAN command opcodes (low 10 bits of opcode_channel; channel in the top nibble)
CMD_RESET_MODE, CMD_NORMAL_MODE, CMD_LISTEN_ONLY = 0x001, 0x002, 0x003
CMD_TIMING_SLOW, CMD_TIMING_FAST = 0x004, 0x005
CMD_FILTER_STD, CMD_WR_ERR_CNT = 0x008, 0x00A
CMD_SET_EN_OPTION, CMD_CLR_DIS_OPTION = 0x00B, 0x00C
CMD_CLK_SET, CMD_LED_SET = 0x80, 0x86
OPTION_ERROR, OPTION_CANDFDISO, USB_OPT_CALIBRATION = 0x0001, 0x0004, 0x8000
MSG_CAN_RX, MSG_ERROR, MSG_STATUS, MSG_BUSLOAD = 0x0001, 0x0002, 0x0003, 0x0004
MSG_CALIBRATION, MSG_OVERRUN = 0x100, 0x101
FLAG_EXT_DATA_LEN, FLAG_EXT_ID, FLAG_RTR = 0x10, 0x02, 0x01
DLC2LEN = list(range(9)) + [12, 16, 20, 24, 32, 48, 64]

# vendor's timings, 80 MHz clock: 1 Mbit/s and 5 Mbit/s at sample point 0.875
SLOW = dict(brp=1, tseg1=69, tseg2=10, sjw=10)
FAST = dict(brp=1, tseg1=13, tseg2=2, sjw=2)


def opc(opcode, channel=0):
    return struct.pack("<H", (channel << 12) | (opcode & 0x3FF))


class PcanUsbFd:
    def __init__(self, addr=None, verbose=False):
        backend = usb.backend.libusb1.get_backend(
            find_library=lambda n: "/opt/homebrew/lib/libusb-1.0.dylib")
        devs = list(usb.core.find(find_all=True, idVendor=VID, idProduct=PID, backend=backend))
        if addr is not None:
            devs = [d for d in devs if d.address == addr]
        if not devs:
            raise SystemExit("no PCAN-USB FD / XCAN device found")
        self.dev = devs[0]
        self.verbose = verbose
        self.dev.set_configuration()
        usb.util.claim_interface(self.dev, 0)
        self.dev.reset() if False else None

    # ---- vendor control requests (pcan_usb_pro_send_req) --------------------
    def fw_info(self):
        raw = self.dev.ctrl_transfer(0xC3, 0, 1, 0, 36, timeout=2000)   # REQ_INFO, INFO_FW
        raw = bytes(raw)
        if len(raw) < 28:
            raise RuntimeError(f"short fw info ({len(raw)} bytes): {raw.hex()}")
        size, typ, hw_type = struct.unpack_from("<HHB", raw, 0)
        bl = raw[5:8]; hw_ver = raw[8]; fw = raw[9:12]
        dev_id0, dev_id1, ser, flags = struct.unpack_from("<IIII", raw, 12)
        info = dict(size=size, type=typ, hw_type=hw_type, bl=f"{bl[0]}.{bl[1]}.{bl[2]}",
                    hw_version=hw_ver, fw=f"{fw[0]}.{fw[1]}.{fw[2]}", dev_id=dev_id0,
                    serial=ser, flags=flags)
        if typ >= 2 and len(raw) >= 33:
            info.update(cmd_out_ep=raw[28], cmd_in_ep=raw[29], data_out_ep=raw[30],
                        data_in_ep=raw[32])
        return info, raw

    def drv_loaded(self, loaded):
        buf = bytes([0, 1 if loaded else 0]) + bytes(14)
        self.dev.ctrl_transfer(0x43, 2, 5, 0, buf, timeout=2000)          # REQ_FCT, FCT_DRVLD

    # ---- uCAN commands over the bulk command pipe --------------------------------
    def send_cmds(self, cmds):
        """cmds: list of 8-byte records. Terminated with an 0xff.. end-of-collection
        record when it fits in 512 bytes, then sent in <=64-byte packets because
        the device is full-speed (pcan_usb_fd_send_cmd)."""
        buf = b"".join(cmds)
        assert all(len(c) == 8 for c in cmds)
        if len(buf) <= 512 - 8:
            buf += b"\xff" * 8
        for i in range(0, len(buf), 64):
            n = self.dev.write(EP_CMD_OUT, buf[i:i + 64], timeout=1000)
            if n != len(buf[i:i + 64]):
                raise RuntimeError(f"short cmd write {n}")
        if self.verbose:
            print(f"  cmd -> {len(buf)} B: {buf[:16].hex()}...")

    def start(self, listen_only=True):
        # pcan_usb_fd_init tail
        self.send_cmds([opc(CMD_CLK_SET) + bytes([0]) + bytes(5)])           # 80 MHz
        self.send_cmds([opc(CMD_LED_SET) + bytes([0]) + bytes(5)])           # LED: device default
        # peak_usb_set_bittiming / set_data_bittiming
        self.send_cmds([opc(CMD_TIMING_SLOW) + bytes([96, (SLOW["sjw"] - 1) & 0x7F,
                        SLOW["tseg2"] - 1, SLOW["tseg1"] - 1]) + struct.pack("<H", SLOW["brp"] - 1)])
        self.send_cmds([opc(CMD_TIMING_FAST) + bytes([0, (FAST["sjw"] - 1) & 0xF,
                        FAST["tseg2"] - 1, FAST["tseg1"] - 1]) + struct.pack("<H", FAST["brp"] - 1)])
        # pcan_usb_fd_start: accept every standard id (64 rows x 32 ids), then options
        self.send_cmds([opc(CMD_FILTER_STD) + struct.pack("<HI", i, 0xFFFFFFFF) for i in range(64)])
        self.send_cmds([opc(CMD_SET_EN_OPTION) + struct.pack("<HHH", OPTION_ERROR, 0, USB_OPT_CALIBRATION)])
        # pcan_usb_fd_set_bus(1): reset error counters, ISO FD, then the mode
        self.send_cmds([
            opc(CMD_WR_ERR_CNT) + struct.pack("<HBBH", 0xC000, 0, 0, 0),
            opc(CMD_SET_EN_OPTION) + struct.pack("<HI", OPTION_CANDFDISO, 0),
            opc(CMD_LISTEN_ONLY if listen_only else CMD_NORMAL_MODE) + bytes(6),
        ])

    def stop(self):
        try:
            self.send_cmds([opc(CMD_RESET_MODE) + bytes(6)])
            self.send_cmds([opc(CMD_CLR_DIS_OPTION) + struct.pack("<HHH", OPTION_ERROR, 0, USB_OPT_CALIBRATION)])
            self.drv_loaded(False)
        except Exception as ex:
            print("stop:", ex)
        usb.util.release_interface(self.dev, 0)

    # ---- transmit (pcan_usb_fd_encode_msg) ---------------------------------------------
    def send(self, can_id, data, fd=True, brs=True):
        """One CAN or CAN FD frame on the message pipe. FD lengths are rounded up
        to a legal DLC and zero-padded, as the wire would carry them anyway."""
        data = bytes(data)
        if fd:
            dlc = next(i for i, ln in enumerate(DLC2LEN) if ln >= len(data))
            data = data.ljust(DLC2LEN[dlc], b"\x00")
            flags = FLAG_EXT_DATA_LEN | (0x20 if brs else 0)
        else:
            dlc = len(data); flags = 0
        size = (20 + len(data) + 3) & ~3
        rec = struct.pack("<HHIIBBHI", size, 0x1000, 0, 0, dlc << 4, 0, flags, can_id & 0x7FF)
        rec = (rec + data).ljust(size, b"\x00") + bytes(4)          # null size = end of list
        n = self.dev.write(EP_MSG_OUT, rec, timeout=100)
        if n != len(rec):
            raise RuntimeError(f"short msg write {n}/{len(rec)}")

    # ---- receive -----------------------------------------------------------------------
    def read(self, timeout_ms=100):
        """One bulk IN transfer, parsed into records: (type, payload dict)."""
        try:
            raw = bytes(self.dev.read(EP_MSG_IN, 2048, timeout=timeout_ms))
        except usb.core.USBTimeoutError:
            return []
        out = []; p = 0
        while p + 4 <= len(raw):
            size, typ = struct.unpack_from("<HH", raw, p)
            if size == 0:
                break
            if size < 4 or p + size > len(raw):
                out.append(("bad", raw[p:].hex())); break
            rec = raw[p:p + size]
            if typ == MSG_CAN_RX and size >= 32:
                ch_dlc, client, flags, can_id = struct.unpack_from("<BBHI", rec, 20)
                dlc = ch_dlc >> 4
                ln = DLC2LEN[dlc] if flags & FLAG_EXT_DATA_LEN else min(dlc, 8)
                out.append(("rx", dict(id=can_id, fd=bool(flags & FLAG_EXT_DATA_LEN),
                                       flags=flags, data=rec[28:28 + ln], raw=rec)))
            elif typ == MSG_ERROR and size >= 16:
                out.append(("error", dict(chan_type=rec[12], code=rec[13], tx_err=rec[14], rx_err=rec[15])))
            elif typ == MSG_STATUS and size >= 16:
                s = rec[12]
                out.append(("status", dict(busoff=bool(s & 0x80), passive=bool(s & 0x20),
                                           warning=bool(s & 0x40), rx_barrier=bool(s & 0x10))))
            elif typ == MSG_CALIBRATION:
                out.append(("calib", None))
            elif typ == MSG_OVERRUN:
                out.append(("overrun", None))
            else:
                out.append((f"type{typ:#x}", rec.hex()))
            p += size
        return out


def pick_by_traffic(secs=0.5):
    """With several adapters plugged in, take the one that hears the arm.
    Both XCAN units report the same ids and serial, so traffic is the only
    discriminator (the same trick can_up.sh uses on Linux). None if only one."""
    backend = usb.backend.libusb1.get_backend(find_library=lambda n: "/opt/homebrew/lib/libusb-1.0.dylib")
    addrs = [d.address for d in usb.core.find(find_all=True, idVendor=VID, idProduct=PID, backend=backend)]
    if len(addrs) <= 1:
        return None
    best, best_n = None, -1
    for addr in addrs:
        d = PcanUsbFd(addr)
        try:
            d.drv_loaded(True); d.start(listen_only=True)
            n = 0; t0 = time.time()
            while time.time() - t0 < secs:
                n += sum(1 for k, _ in d.read(50) if k == "rx")
        finally:
            d.stop()
        print(f"  xcan usb addr {addr}: {n} frames in {secs:g}s")
        if n > best_n:
            best, best_n = addr, n
    return best


class Message:
    """The subset of can.Message the jog tool uses."""
    __slots__ = ("arbitration_id", "data", "is_fd", "bitrate_switch")

    def __init__(self, arbitration_id, data, is_fd=False, bitrate_switch=False):
        self.arbitration_id = arbitration_id; self.data = bytes(data)
        self.is_fd = is_fd; self.bitrate_switch = bitrate_switch


class XcanBus:
    """python-can-shaped wrapper: recv(timeout) / send(msg) / shutdown().

    A reader thread keeps the bulk IN pipe drained into a queue, so a caller
    that polls recv(0.0) in its own loop never leaves frames in the device
    (the arm's 200 Hz stream would otherwise overrun the adapter's buffer).
    """

    def __init__(self, addr=None):
        import collections, threading
        if addr is None:
            addr = pick_by_traffic()
        self.dev = PcanUsbFd(addr)
        self.info, _ = self.dev.fw_info()
        self.channel_info = f"xcan usb addr {self.dev.dev.address} fw {self.info['fw']}"
        self.dev.drv_loaded(True)
        self.dev.start(listen_only=False)
        self.q = collections.deque(maxlen=4096)
        self.state = "ok"; self.errors = 0
        self._stop = False
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._reader, daemon=True); self._t.start()

    def _reader(self):
        while not self._stop:
            try:
                recs = self.dev.read(50)
            except usb.core.USBError as ex:
                if self._stop: return
                self.state = f"usb error: {ex}"; time.sleep(0.05); continue
            for kind, rec in recs:
                if kind == "rx":
                    with self._cv:
                        self.q.append(Message(rec["id"], rec["data"], rec["fd"]))
                        self._cv.notify()
                elif kind == "status":
                    self.state = ("bus-off" if rec["busoff"] else "error-passive" if rec["passive"]
                                  else "warning" if rec["warning"] else "ok")
                elif kind == "error":
                    self.errors += 1

    def recv(self, timeout=0.0):
        with self._cv:
            if not self.q and timeout > 0:
                self._cv.wait(timeout)
            return self.q.popleft() if self.q else None

    def send(self, msg, timeout=None):
        self.dev.send(msg.arbitration_id, msg.data, fd=msg.is_fd, brs=msg.bitrate_switch)

    def status_string(self):
        return self.state

    def shutdown(self):
        self._stop = True; self._t.join(0.5)
        self.dev.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", type=int, help="USB device address (ioreg / pyusb), default first")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--normal", action="store_true", help="normal mode: the adapter ACKs frames (still sends none)")
    a = ap.parse_args()

    d = PcanUsbFd(a.addr, a.v)
    info, raw = d.fw_info()
    print(f"device addr {d.dev.address}: fw info {info}")
    if a.v:
        print("  raw:", raw.hex())
    d.drv_loaded(True)
    d.start(listen_only=not a.normal)
    print(f"listening {a.secs:g}s ({'normal mode, ACK only' if a.normal else 'listen-only mode'}, sends no frames)")
    counts = {}; q = None; e = None; n052 = 0; t0 = time.time(); errs = []; last = None; same = 0
    try:
        while time.time() - t0 < a.secs:
            for kind, rec in d.read(100):
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "rx":
                    key = f"rx 0x{rec['id']:03x} len{len(rec['data'])}{' fd' if rec['fd'] else ''}"
                    counts[key] = counts.get(key, 0) + 1
                    if a.v and counts["rx"] <= 3:
                        print("  raw rx record:", rec["raw"].hex())
                    if rec["id"] == 0x052 and len(rec["data"]) >= 42:
                        r = struct.unpack(">21h", rec["data"][:42])
                        q = [r[g * 3] / 4700.0 for g in range(7)]
                        e = [r[g * 3 + 2] / 600.0 for g in range(7)]
                        n052 += 1
                        same = same + 1 if rec["data"] == last else 0; last = rec["data"]
                elif kind in ("error", "status") and len(errs) < 5:
                    errs.append((kind, rec))
    finally:
        d.stop()
    print("records:", counts)
    if errs:
        print("first error/status records:", errs)
    if q:
        print(f"0x052 at {n052 / a.secs:.0f} Hz")
        print("q (deg):", [round(math.degrees(x), 1) for x in q[:6]], "grip grp7:", round(math.degrees(q[6]), 1))
        print("effort: ", [round(x, 2) for x in e])
        print("identical-payload streak at end:", same, "(FROZEN = released state)" if same > 50 else "(live)")
    else:
        print("no 0x052 feedback decoded")
    return 0 if q else 2


if __name__ == "__main__":
    raise SystemExit(main())
