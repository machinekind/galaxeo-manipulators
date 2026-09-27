"""head_steer_a1x.py without a camera or an arm: the stream on the fake CAN bus, and the head mapping."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

import head_steer_a1x as hs  # noqa: E402
from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import BusTransport  # noqa: E402
from galaxeo.arm.fake import FakeA1XBus  # noqa: E402

START = (78.6, 44.5, -13.1, -23.8, 0.0, -38.9)
LO = np.radians([-165, 0, -190, -90, -90, -165])
HI = np.radians([165, 180, 0, 90, 90, 165])


class Rig:
    def __init__(self, max_speed=30.0, **kw):
        self.bus = FakeA1XBus(start_deg=START)
        self.bus.tick(n=10)
        self.t = BusTransport(self.bus, tx=True, clock=self.bus.now, sleep=self.bus.sleep)
        self.s = hs.Streamer(self.t, LO, HI, max_speed=max_speed, **kw)

    def run(self, secs):
        for _ in range(int(round(secs / 0.005))):
            self.s.step()
            self.bus.sleep(0.005)

    def cmds(self):
        return [np.degrees(self.bus.pdes(d)) for d in self.bus.frames(P.CMD_ID)]


def test_nothing_goes_out_before_a_target():
    rig = Rig()
    rig.run(0.5)
    assert rig.bus.frames(P.CMD_ID) == [] and rig.s.fault == ""
    assert np.degrees(rig.s.fresh().q) == pytest.approx(START, abs=0.01)


def test_stream_starts_at_the_measured_pose_and_slews_at_max_speed():
    rig = Rig(max_speed=30.0)
    rig.run(0.05)
    goal = np.radians(START) + np.radians([20, 0, 0, 0, 0, 0])
    rig.s.set_target(goal)
    rig.run(0.2)
    cmds = rig.cmds()
    assert cmds[0] == pytest.approx(START, abs=0.2)                  # first p_des = where the arm is
    steps = np.abs(np.diff([c[0] for c in cmds]))
    assert steps.max() <= 30.0 * 0.005 * 2 + 0.01                    # <= 30 deg/s (dt capped at 2 periods)
    rig.run(1.0)
    assert rig.cmds()[-1][0] == pytest.approx(START[0] + 20, abs=0.05)


def test_targets_are_clamped_to_the_limits():
    rig = Rig(max_speed=45.0)
    rig.run(0.05)
    rig.s.set_target(np.radians([200, 44.5, 50, -23.8, 0, -38.9]))
    rig.run(6.0)
    last = rig.cmds()[-1]
    assert last[0] == pytest.approx(165.0, abs=0.05) and last[2] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("knob, words", [("silent", "old"), ("frozen", "frozen"), ("deaf", "deaf")])
def test_silent_frozen_or_deaf_arm_trips_and_stops_sending(knob, words):
    rig = Rig()
    rig.run(0.05)
    rig.s.set_target(np.radians(START) + np.radians([10, 0, 0, 0, 0, 0]))
    rig.run(0.05)
    setattr(rig.bus, knob, True)
    rig.run(2.0)
    assert words in rig.s.fault
    n = len(rig.bus.frames(P.CMD_ID))
    rig.run(0.5)
    assert len(rig.bus.frames(P.CMD_ID)) == n                        # nothing more goes out
    rig.s.set_target(np.radians(START))                              # ignored until clear()
    rig.run(0.1)
    assert len(rig.bus.frames(P.CMD_ID)) == n


def test_hold_keeps_the_arm_where_it_is():
    rig = Rig()
    rig.run(0.05)
    rig.s.set_target(np.radians(START) + np.radians([30, 0, 0, 0, 0, 0]))
    rig.run(0.3)
    rig.s.hold()
    here = np.degrees(rig.s.fresh().q)
    rig.run(1.0)
    assert np.degrees(rig.s.fresh().q)[0] == pytest.approx(here[0], abs=0.3)


# ------------------------------------------------------------------------ mapping
def head(yaw=0.0, pitch=0.0, roll=0.0):
    return hs.HeadPose(yaw, pitch, roll, np.zeros((1, 2)))


def steer(rig=None, **kw):
    windows = {"yaw": 45.0, "pitch": 30.0, "roll": 45.0}
    opts = dict(deadzone=0.0, filter_cutoff=1e6, filter_beta=0.0)
    opts.update(kw)
    clock = rig.bus.now if rig is not None else (lambda: 0.0)
    return hs.HeadSteer(list(START), windows, stream=rig.s if rig else None, clock=clock, **opts)


def test_head_pose_signs():
    assert hs.head_pose(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    t = math.radians(20)
    turn = [[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]
    assert hs.head_pose(turn)[0] == pytest.approx(20.0)
    up = [[1, 0, 0], [0, math.cos(t), math.sin(t)], [0, -math.sin(t), math.cos(t)]]
    assert hs.head_pose(up)[1] == pytest.approx(20.0)


def test_relative_mapping_signs_windows_and_no_jump_on_engage():
    s = steer()
    s.engage(True)
    s.step(head(yaw=30), 0.03)                                      # anchor at 30 deg: no jump
    assert s.targets == pytest.approx(START, abs=1e-3)
    s.step(head(yaw=40, pitch=5, roll=-8), 0.03)
    assert s.targets[0] == pytest.approx(START[0] + 10, abs=1e-3)
    assert s.targets[3] == pytest.approx(START[3] - 5, abs=1e-3)    # nod up -> J4 negative (tool up)
    assert s.targets[5] == pytest.approx(START[5] - 8, abs=1e-3)
    s.step(head(yaw=500, pitch=500), 0.03)
    assert s.targets[0] == pytest.approx(START[0] + 45, abs=1e-3)
    assert s.targets[3] == pytest.approx(START[3] - 30, abs=1e-3)


def test_lost_face_holds_and_returning_face_reanchors():
    s = steer()
    s.engage(True)
    s.step(head(), 0.03)
    s.step(head(yaw=10), 0.03)
    s.last_face = -10.0
    s.step(None, 0.03)
    assert s.anchor_head is None
    s.step(head(yaw=-30), 0.03)
    assert s.targets[0] == pytest.approx(START[0] + 10, abs=1e-3)


def test_engage_on_the_rig_seeds_from_the_measured_pose_and_effort_jump_stops():
    rig = Rig()
    rig.run(0.05)
    s = steer(rig)
    s.engage(True)
    s.step(head(), 0.03)
    s.step(head(yaw=10), 0.03)
    rig.run(1.0)
    assert np.degrees(rig.s.fresh().q)[0] == pytest.approx(START[0] + 10, abs=0.3)
    rig.bus.eff[0] = 15.0
    rig.run(0.02)
    s.step(head(yaw=20), 0.03)
    assert not s.engaged and "effort" in s.stopped


def test_engage_refused_without_fresh_feedback_and_after_a_fault():
    rig = Rig()
    rig.run(0.05)
    rig.bus.silent = True
    rig.run(0.5)
    s = steer(rig)
    s.engage(True)
    assert not s.engaged
    rig.bus.silent = False
    rig.run(0.05)
    s.engage(True)
    assert s.engaged and rig.s.fault == ""


def test_rest_and_look_poses():
    assert hs.rest_pose(START) == [78.6, 0.5, -0.5, 0.0, 0.0, -38.9]
    assert hs.look_pose([10, 0, 0, 0, 0, 5]) == [10, *hs.LOOK, 5]
