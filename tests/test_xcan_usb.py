"""galaxeo.xcan_usb without a dongle: the TX record bytes and the --grip-test logic."""

from __future__ import annotations

import argparse
import struct

from galaxeo import protocol as P
from galaxeo import xcan_usb as X


def test_gripper_record_matches_pcan_usb_fd_encode_msg_by_hand():
    """A 10-byte 0x051 with BRS, laid out as struct pucan_tx_msg (peak_canfd.h), computed by hand.

    p_des -1.5 * 4700 = -7050 = 0xe476, kp 20 * 60 = 1200 = 0x04b0, kd 1 * 150 = 0x0096.
    """
    payload = P.encode_gripper(-1.5, 20.0, 1.0)
    assert payload.hex() == "e476000004b000960000"
    expected = bytes.fromhex(
        "2000"          # size: ALIGN(20 + 10, 4) = 32, the kernel's own value
        "0010"          # type: PUCAN_MSG_CAN_TX 0x1000
        "00000000"      # tag_low  (kernel: not written)
        "00000000"      # tag_high (kernel: not written)
        "90"            # channel_dlc: channel 0 | DLC 9 << 4 (can_fd_len2dlc(10) = 9)
        "00"            # client (kernel: not written)
        "3000"          # flags: EXT_DATA_LEN 0x10 | BITRATE_SWITCH 0x20
        "51000000"      # can_id 0x051
        "e476000004b000960000"   # d[0..9]
        "0000"          # d[10..11]: alignment padding, sent on the wire as DLC 9's last 2 bytes
        "00000000"      # null size: end of the record list
    )
    assert X.encode_tx_record(P.GRIP_ID, payload) == expected


def test_arm_and_function_frame_records_are_unchanged():
    """The bytes the arm has been driven with: 60 B padded to 64 in an 84-byte record, and 0x053 classic."""
    arm = P.encode_arm([0.1] * 6, 20.0, 1.0)
    rec = X.encode_tx_record(P.CMD_ID, arm)
    assert rec[:20] == struct.pack("<HHIIBBHI", 84, 0x1000, 0, 0, 0xF0, 0, 0x30, 0x050)
    assert rec[20:80] == arm and rec[80:] == bytes(8) and len(rec) == 88
    ff = X.encode_tx_record(P.FF_ID, P.encode_ff(1), fd=False)
    assert ff == struct.pack("<HHIIBBHI", 24, 0x1000, 0, 0, 0x10, 0, 0, 0x053) + b"\x01" + bytes(3) + bytes(4)


def test_every_fd_length_rounds_to_its_dlc():
    for n in range(0, 65):
        rec = X.encode_tx_record(0x123, bytes([0xAA]) * n)
        size, _t, _a, _b, ch_dlc, _c, flags, cid = struct.unpack_from("<HHIIBBHI", rec)
        wire = X.DLC2LEN[ch_dlc >> 4]
        assert wire >= n and (ch_dlc >> 4 == 0 or X.DLC2LEN[(ch_dlc >> 4) - 1] < n)
        assert size == (20 + wire + 3) & ~3 and len(rec) == size + 4
        assert rec[20:20 + n] == bytes([0xAA]) * n and rec[20 + n:] == bytes(len(rec) - 20 - n)
        assert flags == 0x30 and cid == 0x123


class FakeDev:
    """PcanUsbFd stand-in: 0x052 every read, 0x054 once a second, a gripper that follows p_des."""

    def __init__(self, obeys=True):
        self.obeys = obeys
        self.time = lambda: self.clock          # stands in for the time module: runs fast
        self.p = 0.0
        self.sent = []
        self.t_status = 0.0
        self.clock = 0.0

    def read(self, timeout_ms=100):
        self.clock += 0.005
        fb = P.encode_feedback([0.0] * 6 + [self.p], None, [0.0] * 6 + [0.5 if self.sent else 0.0])
        out = [("rx", dict(id=P.FB_ID, data=fb, fd=True, flags=0x10, raw=b""))]
        if self.clock - self.t_status >= 1.0:
            self.t_status = self.clock
            word = 0x0000 if (self.obeys and self.sent) else 0x0010
            out.append(("rx", dict(id=P.STATUS_ID, data=struct.pack(">8H", *([0x10] * 6 + [word, 1])),
                                   fd=True, flags=0x10, raw=b"")))
        return out

    def send(self, can_id, data, fd=True, brs=True):
        assert can_id == P.GRIP_ID and len(data) == 10
        self.sent.append(bytes(data))
        if self.obeys:
            self.p = struct.unpack(">h", data[:2])[0] / 4700.0


def _args():
    return argparse.Namespace(open=-1.5, close=0.0, each=3.0, rate=200.0, grip_kp=20.0, kd=1.0)


def test_grip_test_reports_a_gripper_that_follows(monkeypatch):
    d = FakeDev(obeys=True)
    monkeypatch.setattr(X, "time", d)
    assert X.grip_test(d, _args()) == 0
    assert d.sent and all(struct.unpack(">h", f[4:6])[0] == 1200 for f in d.sent)   # kp 20
    assert min(struct.unpack(">h", f[:2])[0] for f in d.sent) == -7050              # reached open
    assert 1350 <= len(d.sent) <= 1900                                               # ~200 Hz x 9 s
    assert struct.unpack(">h", d.sent[-1][:2])[0] == -7050                           # ends open


def test_grip_test_reports_a_deaf_gripper(monkeypatch):
    d = FakeDev(obeys=False)
    monkeypatch.setattr(X, "time", d)
    assert X.grip_test(d, _args()) == 1


def test_grip_test_sends_nothing_without_feedback(monkeypatch):
    d = FakeDev()
    d.read = lambda timeout_ms=100: (setattr(d, "clock", d.clock + 0.005), [])[1]
    monkeypatch.setattr(X, "time", d)
    assert X.grip_test(d, _args()) == 2 and d.sent == []
