"""move_to_point_a1x.py end to end against a fake arm ticking in real time on a fake bus."""

from __future__ import annotations

import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import Arm  # noqa: E402
from galaxeo.arm.fake import PERIOD, FakeA1XBus  # noqa: E402


class LiveFake(FakeA1XBus):
    """The fake arm on the wall clock: a thread emits 0x052 at 200 Hz until closed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self.closed:
            with self._lock:
                self.tick()
            time.sleep(PERIOD)

    def recv(self, timeout=0.0):
        with self._lock:
            return super().recv(timeout)

    def send(self, can_id, data, *, fd=True):
        with self._lock:
            super().send(can_id, data, fd=fd)


@pytest.fixture(scope="module")
def center():
    arm = Arm()
    return arm.kin.ik(arm.reach.center, seed=arm.kin.home).q, arm.reach.center


def run(monkeypatch, bus, *args):
    import move_to_point_a1x as mtp

    monkeypatch.setattr(mtp, "open_bus", lambda iface: bus)
    monkeypatch.setattr(sys, "argv", ["move_to_point_a1x.py", "--iface", "fake", *args])
    return mtp.main()


def test_check_and_dry_run_transmit_nothing(monkeypatch, center, capsys):
    q, c = center
    bus = LiveFake(start_deg=np.degrees(q))
    assert run(monkeypatch, bus, "--check", "--secs", "0.3") == 0
    assert "inside the reach box" in capsys.readouterr().out
    bus = LiveFake(start_deg=np.degrees(q))
    assert run(monkeypatch, bus, "--dry-run", "--tx", "--point", *map(str, c + [0.02, 0, 0])) == 0
    assert "dry run" in capsys.readouterr().out
    assert bus.tx == []


def test_point_outside_the_box_is_refused_before_anything_is_sent(monkeypatch, center, capsys):
    q, _ = center
    bus = LiveFake(start_deg=np.degrees(q))
    assert run(monkeypatch, bus, "--tx", "--point", "0.35", "0", "0.01") == 2
    assert "outside volume" in capsys.readouterr().out
    assert bus.tx == []


def test_live_move_to_an_absolute_point_with_enable(monkeypatch, center, capsys):
    q, c = center
    bus = LiveFake(start_deg=np.degrees(q))
    target = c + np.array([0.03, 0.02, 0.0])
    assert run(monkeypatch, bus, "--tx", "--enable", "--speed", "20", "--point", *map(str, target)) == 0
    out = capsys.readouterr().out
    assert "reached" in out and "deg to go" in out
    codes = [d[0] for i, d, _ in bus.tx if i == P.FF_ID]
    assert codes == [1, 5, 6]
    kinds = [i for i, _, _ in bus.tx]
    assert kinds[0] == P.CMD_ID                                   # p_des streams before the first code
    reached = Arm().fk(np.array(bus.q[:6]))[:3, 3]
    assert np.linalg.norm(reached - target) < 0.005


def test_jog_stops_at_the_table_and_sends_nothing_until_armed(monkeypatch, center):
    import types

    import jog_a1x

    arm_model = Arm()
    q = arm_model.kin.ik([0.30, 0.0, 0.07], seed=center[0]).q
    assert arm_model.collision.config_clear(q) is None
    bus = LiveFake(start_deg=np.degrees(q))
    monkeypatch.setattr(jog_a1x, "open_bus", lambda iface: bus)
    a = types.SimpleNamespace(rate=200.0, speed=20.0, kp=20.0, kd=1.0, grip_kp=20.0, grip_start=-2.0,
                              grip_closed=0.6, grip_speed=1.0, grip_force=1.2, home_rad=[0.0] * 6,
                              home_speed=9.0)
    arm = jog_a1x.A1X("fake", dry_run=False)
    jog = jog_a1x.Jog(arm, a)
    try:
        time.sleep(0.3)
        assert bus.tx == []                                       # not armed: nothing on the wire
        assert jog.arm_on()
        down = +1 if arm_model.fk(q + [0, 0.05, 0, 0, 0, 0])[2, 3] < arm_model.fk(q)[2, 3] else -1
        jog.set_vel(1, down)
        t0 = time.time()
        while "blocked" not in jog.status and time.time() - t0 < 5.0:
            time.sleep(0.05)
        assert "blocked" in jog.status and "table" in jog.status, jog.status
        sent = [np.array(bus.pdes(d)) for i, d, _ in bus.tx if i == P.CMD_ID]
        assert sent and all(jog.checker.config_clear(p) is None for p in sent[::10])
    finally:
        jog.arm_off("test"); jog.stop(); arm.close()


def test_jog_go_center_unfolds_a_folded_arm_into_the_box(monkeypatch, center):
    import types

    import jog_a1x

    folded = [3.2, -0.3, 0.0, -0.9, 0.8, -2.9]                   # the rest pose on the table
    bus = LiveFake(start_deg=folded)
    monkeypatch.setattr(jog_a1x, "open_bus", lambda iface: bus)
    a = types.SimpleNamespace(rate=200.0, speed=20.0, kp=20.0, kd=1.0, grip_kp=20.0, grip_start=-2.0,
                              grip_closed=0.6, grip_speed=1.0, grip_force=1.2, home_rad=[0.0] * 6,
                              home_speed=120.0)
    arm = jog_a1x.A1X("fake", dry_run=False)
    jog = jog_a1x.Jog(arm, a)
    try:
        time.sleep(0.3)
        jog.go_center()
        assert "arm first" in jog.status
        assert jog.arm_on()
        assert not jog.model.reach.contains(jog.model.fk(jog.target)[:3, 3])
        jog.go_center()
        t0 = time.time()
        while "at box centre" not in jog.status and time.time() - t0 < 5.0:
            time.sleep(0.05)
        assert "at box centre" in jog.status, jog.status
        tip = jog.model.fk(jog.target)[:3, 3]
        assert np.linalg.norm(tip - center[1]) < 0.003
        sent = [np.array(bus.pdes(d)) for i, d, _ in bus.tx if i == P.CMD_ID]
        assert sent and all(jog.checker.config_clear(p) is None for p in sent[::10])
        assert np.max(np.abs(np.diff(sent, axis=0))) < np.radians(120.0 / 200.0) * 1.5   # slewed, no jump
    finally:
        jog.arm_off("test"); jog.stop(); arm.close()
