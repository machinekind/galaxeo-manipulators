"""galaxeo.bus without hardware: a fake socket, a fake python-can and a fake xcan driver."""

from __future__ import annotations

import errno
import struct
import subprocess
import sys
import types

import pytest

from galaxeo import bus as B
from galaxeo import protocol as P


class FakeSocket:
    """Records what SocketCanBus does to its socket; `rx` is the kernel's receive queue."""

    instances: list = []

    def __init__(self, family, kind, proto):
        self.args = (family, kind, proto)
        self.opts, self.bound, self.blocking, self.closed = [], None, True, False
        self.sent: list[bytes] = []
        self.rx: list[bytes] = []
        FakeSocket.instances.append(self)

    def setsockopt(self, *a):
        self.opts.append(a)

    def bind(self, addr):
        if FakeSocket.fail_bind:
            raise OSError(errno.ENODEV, "No such device")
        self.bound = addr

    def setblocking(self, flag):
        self.blocking = flag

    def recv(self, n):
        if not self.rx:
            raise BlockingIOError
        return self.rx.pop(0)[:n]

    def send(self, buf):
        self.sent.append(bytes(buf))
        return len(buf)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_socket(monkeypatch):
    FakeSocket.instances, FakeSocket.fail_bind = [], False
    monkeypatch.setattr(B.socket, "socket", FakeSocket)
    return FakeSocket


def test_socketcan_opens_an_fd_raw_socket_bound_to_the_interface(fake_socket):
    b = B.SocketCanBus("can1")
    s = fake_socket.instances[0]
    assert s.args == (B.AF_CAN, B.socket.SOCK_RAW, B.CAN_RAW)
    assert (B.SOL_CAN_RAW, B.CAN_RAW_FD_FRAMES, 1) in s.opts
    assert s.bound == ("can1",) and s.blocking is False
    b.close()
    assert s.closed


def test_socketcan_tx_struct_is_16_or_72_bytes_with_true_length(fake_socket):
    b = B.SocketCanBus("can0")
    s = fake_socket.instances[0]
    b.send(P.CMD_ID, P.encode_arm([0.0] * 6, 20.0, 1.0))
    b.send(P.GRIP_ID, P.encode_gripper(-2.0, 25.0, 1.0))
    b.send(P.FF_ID, P.encode_ff(1), fd=False)
    arm, grip, ff = s.sent
    assert len(arm) == B.CANFD_MTU == 72 and len(grip) == 72 and len(ff) == B.CAN_MTU == 16
    for buf, cid, n, flags in ((arm, 0x050, 60, B.CANFD_BRS), (grip, 0x051, 10, B.CANFD_BRS), (ff, 0x053, 1, 0)):
        can_id, length, fl, _, _ = struct.unpack_from("=IBBBB", buf, 0)
        assert (can_id, length, fl) == (cid, n, flags)
    assert grip[8:18] == P.encode_gripper(-2.0, 25.0, 1.0) and grip[18:] == bytes(54)
    assert ff[8] == 1 and ff[9:] == bytes(7)


def test_classic_frame_refuses_more_than_eight_bytes(fake_socket):
    b = B.SocketCanBus("can0")
    with pytest.raises(ValueError):
        b.send(P.CMD_ID, bytes(60), fd=False)
    with pytest.raises(ValueError):
        B.pack_frame(P.CMD_ID, bytes(65))


def test_socketcan_recv_parses_frames_and_skips_truncated_reads(fake_socket):
    b = B.SocketCanBus("can0")
    s = fake_socket.instances[0]
    fb = P.encode_feedback([0.1] * 7)
    s.rx = [bytes(8), B.pack_frame(P.FB_ID | 0x80000000, fb)]     # a short read, then an FD frame (EFF flag set)
    f = b.recv()
    assert f == B.Frame(P.FB_ID, fb)
    assert b.recv() is None                                     # queue empty -> None, not an exception


def test_socketcan_bind_failure_names_the_ip_link_command(fake_socket):
    fake_socket.fail_bind = True
    with pytest.raises(RuntimeError) as ei:
        B.SocketCanBus("can0")
    msg = str(ei.value)
    assert "ip link set can0 type can bitrate 1000000" in msg and "dbitrate 5000000" in msg and "fd on" in msg
    assert fake_socket.instances[0].closed


def test_socketcan_off_linux_says_what_to_use_instead(monkeypatch):
    def refuse(*a):
        raise OSError(errno.EAFNOSUPPORT, "Address family not supported")
    monkeypatch.setattr(B.socket, "socket", refuse)
    with pytest.raises(RuntimeError, match="xcan"):
        B.SocketCanBus("can0")


def test_open_bus_dispatches_on_the_interface_string(monkeypatch):
    seen = []
    monkeypatch.setattr(B, "SocketCanBus", lambda iface: seen.append(("socketcan", iface)) or "S")
    monkeypatch.setattr(B, "XcanBus", lambda addr=None: seen.append(("xcan", addr)) or "X")
    monkeypatch.setattr(B, "PythonCanBus", lambda i, c, **kw: seen.append(("python-can", i, c, bool(kw))) or "P")
    assert [B.open_bus(s) for s in ("can0", "xcan", "xcan:6", "PCAN_USBBUS1", "virtual:x")] == list("SXXPP")
    assert seen == [("socketcan", "can0"), ("xcan", None), ("xcan", 6), ("python-can", "pcan", "PCAN_USBBUS1", True),
                    ("python-can", "virtual", "x", False)]
    with pytest.raises(ValueError):
        B.open_bus("")


def test_xcan_is_only_offered_on_darwin(monkeypatch):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="SocketCAN"):
        B.open_bus("xcan")
    assert B.default_iface() == "can0"
    monkeypatch.setattr(B.platform, "system", lambda: "Darwin")
    assert B.default_iface() == "xcan"


def test_xcan_module_imports_on_any_os_without_pyusb():
    code = ("import sys; sys.modules['usb'] = None\n"
            "import galaxeo.xcan_usb as x, galaxeo.bus, galaxeo.protocol\n"
            "assert issubclass(x.XcanError, RuntimeError)\n"
            "try:\n    x._usb()\nexcept ImportError as ex:\n    assert 'galaxeo[xcan]' in str(ex)\n"
            "else:\n    raise SystemExit('pyusb should look missing')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_xcan_driver_needs_darwin_and_raises_xcan_error_not_systemexit(monkeypatch):
    from galaxeo import xcan_usb
    monkeypatch.setattr(xcan_usb.platform, "system", lambda: "Linux")
    with pytest.raises(xcan_usb.XcanError, match="SocketCAN"):
        xcan_usb.PcanUsbFd()


def test_xcan_bus_adapts_the_driver_to_frames(monkeypatch):
    from galaxeo import xcan_usb

    class Driver:
        def __init__(self, addr):
            self.addr, self.sent, self.q, self.state, self.closed = addr, [], [], "ok", False
            self.channel_info = f"fake {addr}"

        def recv(self, timeout=0.0):
            return self.q.pop(0) if self.q else None

        def send(self, msg, timeout=None):
            self.sent.append((msg.arbitration_id, msg.data, msg.is_fd, msg.bitrate_switch))

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(B.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(xcan_usb, "XcanBus", Driver)
    b = B.open_bus("xcan:3")
    d = b._bus
    assert d.addr == 3 and b.state == "ok"
    d.q.append(xcan_usb.Message(P.FB_ID, bytes(48), True))
    assert b.recv() == B.Frame(P.FB_ID, bytes(48)) and b.recv() is None
    b.send(P.FF_ID, b"\x01", fd=False)
    b.send(P.CMD_ID, bytes(60))
    assert d.sent == [(P.FF_ID, b"\x01", False, False), (P.CMD_ID, bytes(60), True, True)]
    d.state = "bus-off"
    assert b.state == "bus-off"
    b.close()
    assert d.closed


def _fake_can_module():
    can = types.ModuleType("can")
    wire: dict[str, list] = {}

    class Message:
        def __init__(self, arbitration_id=0, data=b"", is_fd=False, bitrate_switch=False, is_extended_id=False):
            self.arbitration_id, self.data, self.is_fd = arbitration_id, bytearray(data), is_fd
            self.bitrate_switch, self.is_extended_id = bitrate_switch, is_extended_id

    class Bus:
        def __init__(self, interface, channel, fd=False, **kw):
            self.interface, self.channel, self.fd, self.kw = interface, channel, fd, kw
            self.me = object()
            wire.setdefault(channel, [])

        def send(self, msg, timeout=None):
            wire[self.channel].append((self.me, msg))

        def recv(self, timeout=None):
            for i, (who, msg) in enumerate(wire[self.channel]):
                if who is not self.me:
                    return wire[self.channel].pop(i)[1]
            return None

        def shutdown(self):
            self.closed = True

    can.Message, can.Bus = Message, Bus
    return can


def test_python_can_bus_with_a_fake_module_round_trips_and_sends_true_lengths(monkeypatch):
    monkeypatch.setitem(sys.modules, "can", _fake_can_module())
    a, b = B.open_bus("virtual:x"), B.open_bus("virtual:x")
    assert a._bus.fd is True and a._bus.interface == "virtual"
    a.send(P.GRIP_ID, P.encode_gripper(0.6, 25.0, 1.0))
    a.send(P.FF_ID, b"\x06", fd=False)
    assert b.recv() == B.Frame(P.GRIP_ID, P.encode_gripper(0.6, 25.0, 1.0))
    assert b.recv() == B.Frame(P.FF_ID, b"\x06")
    assert b.recv() is None
    pcan = B.open_bus("PCAN_USBBUS1")
    assert pcan._bus.interface == "pcan" and pcan._bus.kw == B.PCAN_FD_TIMING


def test_python_can_missing_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "can", None)
    with pytest.raises(ImportError, match=r"galaxeo\[pcan\]"):
        B.open_bus("virtual:x")


def test_python_can_virtual_round_trip():
    pytest.importorskip("can")
    a, b = B.PythonCanBus("virtual", "galaxeo-test"), B.PythonCanBus("virtual", "galaxeo-test")
    try:
        a.send(P.GRIP_ID, P.encode_gripper(-1.0, 25.0, 1.0))
        f = b.recv(1.0)
        assert f == B.Frame(P.GRIP_ID, P.encode_gripper(-1.0, 25.0, 1.0))
    finally:
        a.close()
        b.close()
