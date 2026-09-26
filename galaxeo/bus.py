"""CAN transports for the A1X behind one small interface: recv / send / close.

    bus = open_bus("can0")            # Linux: raw SocketCAN, stdlib only
    bus = open_bus("xcan")            # macOS: galaxeo.xcan_usb (pyusb + libusb)
    bus = open_bus("xcan:6")          #        ... the dongle at USB address 6
    bus = open_bus("PCAN_USBBUS1")    # Windows: PEAK PCAN-Basic via python-can
    bus = open_bus("virtual:x")       # any python-can <interface>:<channel>

Every backend imports its third-party module only when it is opened, so this
module imports on any OS with nothing but the standard library. A missing module
is an ImportError that names the pip extra to install.

send(can_id, data, fd=True) puts the payload on the wire with its TRUE length:
the gripper answers only a 10-byte 0x051 frame and ignores a 12-byte one. Use
fd=False for the 1-byte function frames on 0x053 (classic CAN, as the vendor
sends them).
"""

from __future__ import annotations

import platform
import select
import socket
import struct
from typing import NamedTuple, Optional

try:                                    # Python 3.8+: typing.Protocol
    from typing import Protocol
except ImportError:                     # pragma: no cover
    Protocol = object                   # type: ignore[assignment,misc]

# ---- SocketCAN constants (linux/can.h); socket may not define them off Linux ------------
AF_CAN = getattr(socket, "AF_CAN", 29)
CAN_RAW = getattr(socket, "CAN_RAW", 1)
SOL_CAN_RAW = 101
CAN_RAW_FD_FRAMES = 5
CAN_EFF_MASK = 0x1FFFFFFF
CAN_MTU = 16                            # struct can_frame:   8 B header + 8 B data
CANFD_MTU = 72                          # struct canfd_frame: 8 B header + 64 B data
CANFD_BRS = 0x01
_HEADER = struct.Struct("=IBBBB")

#: Bringing the link up on Linux (docs/HARDWARE.md, ./can_up.sh does the same).
IP_LINK_UP = ("sudo ip link set {iface} down; sudo ip link set {iface} type can bitrate 1000000 "
              "sample-point 0.875 dbitrate 5000000 dsample-point 0.875 fd on restart-ms 100; "
              "sudo ip link set {iface} up")

#: PEAK-USB FD, 80 MHz clock: 1 Mbit/s and 5 Mbit/s, both at sample point 0.875
#: (the vendor's own timings, docs/HARDWARE.md). python-can keyword arguments.
PCAN_FD_TIMING = dict(f_clock_mhz=80, nom_brp=1, nom_tseg1=69, nom_tseg2=10, nom_sjw=10,
                      data_brp=1, data_tseg1=13, data_tseg2=2, data_sjw=2)


class Frame(NamedTuple):
    """One received frame: 11-bit id and the payload as it came off the wire."""

    can_id: int
    data: bytes


class CanBus(Protocol):
    """What every transport offers. recv never raises for "nothing queued": it returns None."""

    def recv(self, timeout: float = 0.0) -> Optional[Frame]: ...

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None: ...

    def close(self) -> None: ...


def pack_frame(can_id: int, data: bytes, fd: bool = True) -> bytes:
    """The complete SocketCAN struct: 16 bytes (classic) or 72 (FD), true length in `len`.

    SocketCAN wants the whole struct written, not header + payload (EINVAL). The len
    field carries the ACTUAL payload length: the kernel maps it to the next FD DLC
    itself, while rounding it here changes what the peer sees (the gripper ignores a
    12-byte 0x051).
    """
    data = bytes(data)
    n = len(data)
    if not fd:
        if n > 8:
            raise ValueError(f"classic CAN carries at most 8 bytes, got {n}")
        return _HEADER.pack(can_id, n, 0, 0, 0) + data.ljust(8, b"\x00")
    if n > 64:
        raise ValueError(f"CAN-FD carries at most 64 bytes, got {n}")
    return _HEADER.pack(can_id, n, CANFD_BRS, 0, 0) + data.ljust(64, b"\x00")


def unpack_frame(buf: bytes) -> Optional[Frame]:
    """Frame from a SocketCAN struct; None for a truncated read."""
    if len(buf) < CAN_MTU:
        return None
    can_id, length, _flags, _r0, _r1 = _HEADER.unpack_from(buf, 0)
    return Frame(can_id & CAN_EFF_MASK, bytes(buf[8:8 + length]))


class SocketCanBus:
    """Raw SocketCAN with CAN_RAW_FD_FRAMES (Linux). Standard library only.

    CAN_RAW_FD_FRAMES is mandatory: the A1X transmits only FD frames, so a socket
    without it silently receives nothing.
    """

    def __init__(self, iface: str = "can0"):
        self.iface = iface
        self.channel_info = f"socketcan {iface}"
        try:
            s = socket.socket(AF_CAN, socket.SOCK_RAW, CAN_RAW)
        except (OSError, AttributeError) as ex:
            raise RuntimeError(f"SocketCAN is Linux-only ({ex}). On macOS use iface 'xcan', on Windows "
                               "'PCAN_USBBUS1'.") from ex
        try:
            s.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
            s.bind((iface,))
        except OSError as ex:
            s.close()
            raise RuntimeError(f"cannot bind {iface}: {ex}. The CAN link is down or missing. Bring it up: "
                               + IP_LINK_UP.format(iface=iface) + "  (or ./can_up.sh)") from ex
        s.setblocking(False)
        self._s = s

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        while True:
            try:
                buf = self._s.recv(CANFD_MTU)
            except (BlockingIOError, InterruptedError):
                if timeout <= 0:
                    return None
                ready, _, _ = select.select([self._s], [], [], timeout)
                if not ready:
                    return None
                timeout = 0.0
                continue
            except OSError as ex:           # link went down mid-run
                raise RuntimeError(f"{self.iface}: {ex}. Bring the link up again: "
                                   + IP_LINK_UP.format(iface=self.iface)) from ex
            frame = unpack_frame(buf)
            if frame is not None:
                return frame                # a truncated read is skipped, not a "queue empty"

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        self._s.send(pack_frame(can_id, data, fd))

    def close(self) -> None:
        try:
            self._s.close()
        except OSError:                     # pragma: no cover
            pass


class XcanBus:
    """The XCAN / PCAN-USB FD dongle on macOS through galaxeo.xcan_usb (pyusb + libusb).

    Caveat: the adapter can only send legal FD lengths, so a 10-byte 0x051 goes out
    as 12 bytes (DLC 9), which the gripper has been seen to ignore. Arm frames are
    unaffected.
    """

    def __init__(self, addr: Optional[int] = None):
        if platform.system() != "Darwin":
            raise RuntimeError("xcan is the macOS driver. On Linux the same dongle is SocketCAN "
                               "(kernel peak_usb): ./can_up.sh, then iface can0.")
        from . import xcan_usb             # stdlib-only import; pyusb loads on open
        self._xcan = xcan_usb
        self._bus = xcan_usb.XcanBus(addr)
        self.channel_info = self._bus.channel_info

    @property
    def state(self) -> str:
        """'ok', 'warning', 'error-passive', 'bus-off' or 'usb error: ...' from the adapter."""
        return str(self._bus.state)

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        m = self._bus.recv(timeout)
        return None if m is None else Frame(int(m.arbitration_id), bytes(m.data))

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        if not fd and len(data) > 8:
            raise ValueError(f"classic CAN carries at most 8 bytes, got {len(data)}")
        self._bus.send(self._xcan.Message(can_id, data, is_fd=fd, bitrate_switch=fd))

    def close(self) -> None:
        self._bus.shutdown()


class PythonCanBus:
    """Any python-can interface: 'virtual' for hardware-free loopback, 'pcan' on Windows."""

    def __init__(self, interface: str, channel: str, **kwargs):
        try:
            import can
        except ImportError as ex:
            raise ImportError(f'CAN interface {interface!r} needs python-can:  pip install "galaxeo[pcan]"'
                              f"  ({ex})") from ex
        self._can = can
        self.channel_info = f"python-can {interface}:{channel}"
        self._bus = can.Bus(interface=interface, channel=channel, fd=True, **kwargs)

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        m = self._bus.recv(timeout)
        return None if m is None else Frame(int(m.arbitration_id), bytes(m.data))

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        if not fd and len(data) > 8:
            raise ValueError(f"classic CAN carries at most 8 bytes, got {len(data)}")
        self._bus.send(self._can.Message(arbitration_id=can_id, data=bytes(data), is_fd=fd,
                                         bitrate_switch=fd, is_extended_id=False))

    def close(self) -> None:
        self._bus.shutdown()


def open_bus(iface: str) -> CanBus:
    """The transport named by `iface`: xcan[:addr], PCAN_*, <interface>:<channel>, or a SocketCAN name."""
    iface = (iface or "").strip()
    if not iface:
        raise ValueError("no CAN interface given (can0 / xcan / xcan:6 / PCAN_USBBUS1 / virtual:x)")
    if iface == "xcan" or iface.startswith("xcan:"):
        addr = int(iface.split(":", 1)[1]) if ":" in iface else None
        return XcanBus(addr)
    if iface.upper().startswith("PCAN"):
        return PythonCanBus("pcan", iface, **PCAN_FD_TIMING)
    if ":" in iface:
        interface, channel = iface.split(":", 1)
        return PythonCanBus(interface, channel)
    return SocketCanBus(iface)


def default_iface() -> str:
    """The usual interface on this OS: xcan on macOS, can0 elsewhere."""
    return "xcan" if platform.system() == "Darwin" else "can0"
