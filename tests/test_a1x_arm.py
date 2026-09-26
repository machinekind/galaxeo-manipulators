from __future__ import annotations

import pytest

import a1x_arm as A


class FakeSocket:
    def __init__(self):
        self.blocking = True
        self.closed = False

    def setblocking(self, flag):
        self.blocking = flag

    def close(self):
        self.closed = True


class FakeCanIO:
    def __init__(self):
        self.calls = []
        self.sent = []

    def open_socket(self, iface, rx_only=True):
        self.calls.append((iface, rx_only))
        return FakeSocket()

    def send_frame(self, sock, can_id, payload):
        self.sent.append((sock, can_id, payload))


def test_a1x_arm_opens_rx_socket_in_dry_run(monkeypatch):
    fake_can_io = FakeCanIO()
    monkeypatch.setattr(A, "can_io", fake_can_io)

    arm = A.A1XArm("can0", dry_run=True)
    assert fake_can_io.calls == [("can0", True)]
    assert arm.sock is not None and arm.sock.blocking is False
    arm.close()
    assert arm.sock is None


def test_a1x_arm_live_socket_is_not_rx_only(monkeypatch):
    fake_can_io = FakeCanIO()
    monkeypatch.setattr(A, "can_io", fake_can_io)

    arm = A.A1XArm("can0", dry_run=False, tx=True)
    assert fake_can_io.calls == [("can0", False)]
    arm.close()


def test_a1x_arm_live_send_uses_can_io_helper(monkeypatch):
    fake_can_io = FakeCanIO()
    monkeypatch.setattr(A, "can_io", fake_can_io)

    arm = A.A1XArm("can0", dry_run=False, tx=True)
    arm._send(0x050, b"\x01\x02")

    assert fake_can_io.sent == [(arm.sock, 0x050, b"\x01\x02")]
    assert arm.n_tx == 1
    arm.close()


def test_a1x_arm_live_send_requires_can_io(monkeypatch):
    fake_can_io = FakeCanIO()
    monkeypatch.setattr(A, "can_io", fake_can_io)
    arm = A.A1XArm("can0", dry_run=False, tx=True)
    monkeypatch.setattr(A, "can_io", None)

    with pytest.raises(RuntimeError, match="cannot transmit"):
        arm._send(0x050, b"\x01\x02")

    arm.close()


def test_a1x_arm_needs_can_io_for_feedback(monkeypatch):
    monkeypatch.setattr(A, "can_io", None)

    with pytest.raises(SystemExit, match="can_io is missing"):
        A.A1XArm("can0", dry_run=True)
