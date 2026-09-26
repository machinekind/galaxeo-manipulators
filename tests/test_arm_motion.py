"""The guarded move on the fake CAN bus (virtual clock) and in the MuJoCo sim. No hardware.

The fake bus speaks the real protocol: every p_des the primitive writes is a 0x050 frame
in `bus.tx`, decoded back here to check what the arm would have been told.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm import Arm, Box, BusTransport, SimTransport, enable  # noqa: E402
from galaxeo.arm import motion as M  # noqa: E402
from galaxeo.arm.fake import FakeA1XBus  # noqa: E402

LSB_DEG = math.degrees(1.0 / P.POS_SCALE)


@pytest.fixture(scope="module")
def planner():
    return Arm()


@pytest.fixture(scope="module")
def center(planner):
    sol = planner.kin.ik(planner.reach.center, seed=planner.kin.home)
    assert sol.ok
    return sol.q


class Rig:
    def __init__(self, start_rad, armed=True, tx=True, **kw):
        self.bus = FakeA1XBus(start_deg=np.degrees(start_rad))
        self.bus.tick(n=10)
        self.transport = BusTransport(self.bus, tx=tx, clock=self.bus.now, sleep=self.bus.sleep)
        self.arm = Arm(self.transport, armed=armed, **kw)
        self.hooks = []
        self.arm.motion.on_tick = self._tick

    def _tick(self, now, cmd, reading, phase):
        for h in self.hooks:
            h(now, cmd, reading, phase)

    def cmd_frames(self, since=0):
        return [np.degrees(self.bus.pdes(d)) for i, d, _ in self.bus.tx[since:] if i == P.CMD_ID]

    def ff_codes(self):
        return [d[0] for i, d, _ in self.bus.tx if i == P.FF_ID]


def far(center):
    """About 25 deg of J1 away from the centre pose, so the ramp takes over a second at 20 deg/s."""
    q = center.copy()
    q[0] += math.radians(25.0)
    return q


def windowed_rates(series, dt, window):
    n = int(round(window / dt))
    return [abs(series[i + n] - series[i]) / window for i in range(len(series) - n)]


def test_a_move_reaches_starts_at_the_measured_pose_and_respects_the_speed_cap(center):
    rig = Rig(center)
    goal = far(center)
    r = rig.arm.move(goal, 20.0)
    assert r.state == "reached", r
    frames = rig.cmd_frames()
    assert np.max(np.abs(frames[0] - np.degrees(center))) <= LSB_DEG
    assert np.max(np.abs(frames[-1] - np.degrees(goal))) <= 2 * LSB_DEG
    j1 = [f[0] for f in frames]
    assert max(windowed_rates(j1, 0.005, 0.1)) <= 1.5 * 20.0 + 5.0 + 1.0
    assert rig.ff_codes() == []
    # 200 Hz on the bus, 50 Hz works the same
    rig2 = Rig(center, rate_hz=50.0)
    assert rig2.arm.move(goal, 20.0).state == "reached"
    assert len(rig2.cmd_frames()) < len(frames) / 3


def test_not_armed_or_not_transmitting_puts_zero_0x050_frames_on_the_wire(center):
    rig = Rig(center, armed=False)
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("rejected", "not armed")
    rig.arm.stop()
    assert rig.bus.tx == []
    dry = Rig(center, tx=False)
    dry.arm.plan(dry.arm.reach.center)
    assert dry.bus.tx == []


def test_effort_spike_mid_ramp_holds_then_backs_off_slowly_toward_the_earlier_pose(center):
    rig = Rig(center)
    trip = {}

    def spike(now, cmd, reading, phase):
        if phase == "moving" and now - rig.t0 > 0.6 and "tx" not in trip:
            rig.bus.eff[0] = 30.0                   # 0x052 effort units, above the 20 threshold
            trip["before"] = reading.q.copy()
        if phase == "backoff" and "tx" not in trip:
            rig.bus.eff[0] = 0.0                    # the obstacle is gone once we pull back
            trip["tx"] = len(rig.bus.tx)
    rig.hooks.append(spike)
    rig.t0 = rig.bus.now()
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("backoff", "impact effort arm_joint1"), r
    assert r.impact.kind == "effort" and r.impact.value == pytest.approx(30.0, abs=0.01)
    back = rig.cmd_frames(trip["tx"] - 1)
    j1 = [f[0] for f in back]
    # the first backoff frame is the hold: the measured pose, not the ramp command ahead of it
    held = [f for f in rig.cmd_frames()][len(rig.cmd_frames()) - len(back) - 1]
    assert abs(held[0] - math.degrees(trip["before"][0])) < 1.0
    assert abs(j1[-1] - r.impact.backoff_pose["arm_joint1"]) < 0.1
    assert (j1[-1] - j1[0]) < -3.0 + 1e-6            # back against the direction of travel, >= 3 deg
    assert max(windowed_rates(j1, 0.005, 0.2)) <= 1.5 * 8.0 + 1.0
    assert max(windowed_rates(j1, 0.005, 0.1)) <= M.BACKOFF_CAP_DEG_S + 2 * LSB_DEG / 0.1
    assert rig.ff_codes() == []


def test_stall_at_a_wall_trips_the_lag_rule_and_backs_off_and_without_the_wall_it_reaches(center):
    control = Rig(center)
    assert control.arm.move(far(center), 20.0).state == "reached"

    rig = Rig(center)
    rig.bus.wall = (0, center[0] + math.radians(8.0), 1.0)
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("backoff", "impact lag arm_joint1"), r
    q = rig.transport.read().q
    assert math.degrees(center[0] + math.radians(8.0) - q[0]) >= 3.0 - 0.1
    frames = rig.cmd_frames()
    # after the backoff the command sits on the arm: nothing left pressing into the wall
    assert abs(frames[-1][0] - math.degrees(q[0])) < 0.5


def test_a_second_trip_during_the_backoff_only_holds(center):
    rig = Rig(center)

    def spike(now, cmd, reading, phase):
        if phase == "moving" and now - rig.t0 > 0.6:
            rig.bus.eff[1] = -35.0
    rig.hooks.append(spike)
    rig.t0 = rig.bus.now()
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("hold", "impact effort arm_joint2"), r
    assert "second trip" in r.detail
    n = len(rig.bus.tx)
    rig.bus.sleep(1.0)
    assert len(rig.bus.tx) == n                       # nothing streams after the hold


def test_stop_mid_ramp_from_another_thread_leaves_the_arm_holding(center):
    rig = Rig(center)
    rig.transport.sleep = lambda s: (time.sleep(s), rig.bus.sleep(s))
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", rig.arm.move(far(center), 20.0)))
    th.start()
    time.sleep(0.5)
    rig.arm.stop()
    th.join(5.0)
    r = out["r"]
    assert (r.state, r.reason) == ("hold", "stop"), r
    frames = rig.cmd_frames()
    assert np.max(np.abs(frames[-1] - np.degrees(rig.transport.read().q))) < 0.5
    assert 2.0 < frames[-1][0] - math.degrees(center[0]) < 23.0      # stopped part way
    n = len(rig.bus.tx)
    assert rig.arm.move(far(center), 20.0).reason == "not armed"
    assert len(rig.bus.tx) == n


def test_stale_feedback_mid_move_stops_the_stream(center):
    rig = Rig(center)

    def silence(now, cmd, reading, phase):
        if now - rig.t0 > 0.5:
            rig.bus.silent = True
    rig.hooks.append(silence)
    rig.t0 = rig.bus.now()
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("hold", "stale feedback"), r
    last_fb = rig.transport.read().t
    sent_after = [f for f in rig.bus.tx if f[0] == P.CMD_ID]
    assert rig.bus.now() - last_fb <= M.FB_STALE_S + 0.02
    rig.bus.sleep(0.5)
    assert len([f for f in rig.bus.tx if f[0] == P.CMD_ID]) == len(sent_after)


def test_frozen_telemetry_missing_feedback_and_jumps_refuse_to_start(center):
    rig = Rig(center)
    rig.bus.frozen = True
    rig.bus.sleep(0.5)
    r = rig.arm.move(far(center), 20.0)
    assert (r.state, r.reason) == ("rejected", "frozen telemetry")

    rig = Rig(center)
    rig.bus.silent = True
    rig.bus.sleep(0.3)
    assert rig.arm.move(far(center), 20.0).reason == "stale feedback"

    rig = Rig(center)
    q = center.copy()
    q[1] += math.radians(50.0)
    r = rig.arm.move(q, 20.0)
    assert (r.state, r.reason) == ("rejected", "jump")
    assert rig.bus.tx == []


def test_limits_are_widened_to_include_the_start_pose():
    start = np.radians([0.0, 20.0, 1.5, 0.0, 0.0, 0.0])      # J3 rests ~1.5 deg past its 0 limit
    rig = Rig(start)
    goal = start.copy()
    goal[0] = math.radians(10.0)
    goal[2] = math.radians(5.0)                               # beyond the widened limit: clamped to 1.5
    r = rig.arm.move(goal, 20.0)
    assert r.state == "reached", r
    frames = rig.cmd_frames()
    assert all(abs(f[2] - 1.5) <= 2 * LSB_DEG for f in frames)


def test_a_loaded_joint_gets_no_effort_rule_and_no_trim(center):
    rig = Rig(center)
    rig.bus.eff[1] = 24.0                                     # J2 carries a load before the move starts
    rig.bus.sleep(0.05)
    r = rig.arm.move(far(center), 20.0)
    assert r.state == "reached", r
    assert np.max(np.abs(rig.cmd_frames()[-1] - np.degrees(far(center)))) <= 2 * LSB_DEG


def test_enable_is_1_5_6_with_p_des_streamed_around_every_code_and_release_is_refused(center):
    rig = Rig(center)
    enable(rig.transport)
    assert rig.ff_codes() == [1, 5, 6]
    kinds = [i for i, _, _ in rig.bus.tx]
    ff_at = [k for k, i in enumerate(kinds) if i == P.FF_ID]
    edges = [0] + ff_at + [len(kinds)]
    for a, b in zip(edges, edges[1:]):
        assert sum(1 for i in kinds[a:b] if i == P.CMD_ID) >= 50
    assert all(fd is False for i, _, fd in rig.bus.tx if i == P.FF_ID)
    for code in P.RELEASE_CODES:
        with pytest.raises(ValueError):
            rig.transport.send_ff(code)
    moved = [np.max(np.abs(f - np.degrees(center))) for f in rig.cmd_frames()]
    assert max(moved) <= 2 * LSB_DEG


def test_busy_while_moving(center):
    rig = Rig(center)
    inner = {}
    rig.hooks.append(lambda *a: inner.setdefault("r", rig.arm.move(center, 20.0)))
    assert rig.arm.move(far(center), 20.0).state == "reached"
    assert inner["r"].reason == "busy"


# ---- MuJoCo -------------------------------------------------------------------------------------
def test_sim_move_lands_the_tool_on_the_point_thanks_to_the_trim(planner, center):
    sim = SimTransport(center)
    arm = Arm(sim, armed=True)
    target = planner.reach.center + np.array([0.05, 0.10, -0.05])
    plan = arm.plan(target)
    assert plan, plan
    r = arm.move(plan.joints, 20.0)
    assert r.state == "reached", r
    assert np.linalg.norm(arm.fk(sim.read().q)[:3, 3] - target) < 0.003


def test_sim_descend_does_not_trim(planner, center):
    sim = SimTransport(center)
    arm = Arm(sim, armed=True)
    plan = arm.plan(planner.reach.center + np.array([0.0, 0.0, -0.05]))
    last = {}
    arm.motion.on_tick = lambda now, cmd, r, phase: last.update(cmd=cmd.copy())
    assert arm.move(plan.joints, 10.0, profile="descend").state == "reached"
    assert np.max(np.abs(last["cmd"] - plan.joints)) < 1e-9


def test_sim_driving_into_an_obstacle_trips_and_backs_off_off_the_push(planner, center):
    goal = far(center)
    tip = planner.fk(center + (goal - center) * 0.8)[:3, 3]
    sim = SimTransport(center, obstacles=[Box(tuple(tip), (0.02, 0.02, 0.02), "wall")])
    arm = Arm(sim, armed=True)
    at_trip = {}
    arm.motion.on_tick = lambda now, cmd, r, phase: at_trip.setdefault(phase, r.q.copy()) if phase == "backoff" else None
    r = arm.move(goal, 20.0)
    assert (r.state, r.reason) == ("backoff", "impact effort arm_joint1"), r
    q = sim.read().q
    assert math.degrees(at_trip["backoff"][0] - q[0]) >= 3.0
    limit = sim.effort_limits(False)[0]
    sim.sleep(0.5)
    assert abs(sim.read().effort[0]) < 0.5 * limit          # not left pressing into the wall
