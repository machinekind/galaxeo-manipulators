"""Impact guard on synthetic 50 Hz loop samples: ramps, a blocked joint, effort spikes, backoff pose choice."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from galaxeo.arm.guard import Impact, ImpactGuard  # noqa: E402

J = [f"arm_joint{i}" for i in range(1, 7)]
DT = 0.02


def pose(j1: float, **rest: float) -> dict[str, float]:
    p = {j: 0.0 for j in J}
    p["arm_joint1"] = j1
    p.update(rest)
    return p


def ramp(guard: ImpactGuard, t0: float, a: float, b: float, speed: float, lead: float,
         wall: float | None = None) -> tuple[float, Impact | None, float]:
    """Ramp the arm_joint1 command from `a` to `b` at `speed` [deg/s], measurement `lead` deg behind.

    The joint gets stuck at `wall`. Once the command arrives, the measurement catches up to `b` at
    1.5 x `speed`, like a real joint.
    """
    t, cmd, meas = t0, a, a
    sign = 1.0 if b >= a else -1.0
    while True:
        t += DT
        cmd = cmd + sign * speed * DT if sign * (b - cmd) > speed * DT else b
        want = b if cmd == b else (cmd - sign * lead if sign * (cmd - a) > lead else a)
        step = 1.5 * speed * DT
        meas = meas + max(-step, min(step, want - meas))
        if wall is not None and sign * (meas - wall) > 0:
            meas = wall
        hit = guard.feed(t, pose(cmd), pose(meas), True)
        if hit is not None:
            return t, hit, meas
        if cmd == b and t > t0 + abs(b - a) / speed + 1.0:
            return t, None, meas


def test_a_normal_ramp_with_the_command_just_ahead_never_trips():
    guard = ImpactGuard(J)
    t, hit, _ = ramp(guard, 0.0, 0.0, 60.0, speed=30.0, lead=2.0)
    assert hit is None
    t, hit, _ = ramp(guard, t, 60.0, -20.0, speed=40.0, lead=2.0)
    assert hit is None
    # a fast joint that still keeps up (command 8 deg ahead, joint at 60 deg/s)
    t, hit, _ = ramp(guard, t, -20.0, 60.0, speed=60.0, lead=8.0)
    assert hit is None
    # an idle arm holding its pose
    for k in range(100):
        assert guard.feed(t + k * DT, pose(60.0), pose(60.0), True) is None


def test_a_blocked_joint_trips_by_lag_after_about_lag_s_with_a_backoff_pose_behind_it():
    guard = ImpactGuard(J)
    t, hit, meas = ramp(guard, 0.0, 0.0, 80.0, speed=20.0, lead=1.0, wall=30.0)
    assert hit is not None and hit.kind == "lag" and hit.joint == "arm_joint1"
    assert meas == 30.0 and hit.value > 6.0
    # At 20 deg/s with the wall at 30 deg, the command is 6 deg past the wall after (30 + 6) / 20 s, then lag_s.
    t_over = (30.0 + 6.0) / 20.0
    assert t_over + 0.15 - DT <= hit.t <= t_over + 0.15 + 2 * DT
    bp = hit.backoff_pose
    assert bp is not None and set(bp) == set(J)
    assert 30.0 - bp["arm_joint1"] >= 3.0                                        # away from the wall
    assert bp["arm_joint1"] > 15.0                                               # but not back at the start
    # a second trip needs a fresh lag_s
    assert guard.feed(hit.t + DT, pose(80.0), pose(30.0), True) is None


def test_lag_while_the_joint_still_moves_fast_is_not_a_block():
    guard = ImpactGuard(J)
    t = 0.0
    for k in range(50):                                                          # step command, joint at 40 deg/s
        t += DT
        assert guard.feed(t, pose(90.0), pose(min(90.0, 40.0 * t)), True) is None


def test_an_effort_spike_trips_after_three_consecutive_samples():
    guard = ImpactGuard(J, effort_abs={"arm_joint2": 20.0, "arm_joint3": 20.0})
    eff = {j: 1.0 for j in J}
    t = 0.0
    for _ in range(20):
        t += DT
        assert guard.feed(t, pose(0.0), pose(0.0), True, efforts=eff) is None
    # two samples over the threshold, then one below: the count restarts
    for e in (25.0, 25.0, 5.0, 25.0, 25.0):
        t += DT
        assert guard.feed(t, pose(0.0), pose(0.0), True, efforts={**eff, "arm_joint2": e}) is None
    t += DT
    hit = guard.feed(t, pose(0.0), pose(0.0), True, efforts={**eff, "arm_joint2": -26.0})
    assert hit is not None and hit.kind == "effort" and hit.joint == "arm_joint2" and hit.value == -26.0
    # a single threshold applies to every joint, including joint1, which had none in the dict above
    g2 = ImpactGuard(J, effort_abs=20.0, effort_n=3)
    hits = [g2.feed(k * DT, pose(0.0), pose(0.0), True, efforts={**eff, "arm_joint1": 30.0}) for k in range(1, 4)]
    assert hits[:2] == [None, None] and hits[2] is not None and hits[2].joint == "arm_joint1"
    g3 = ImpactGuard(J, effort_abs=20.0)
    assert all(g3.feed(k * DT, pose(0.0), pose(0.0), True, efforts={"arm_joint1": 1.0}) is None
               for k in range(1, 10))


def test_samples_with_nothing_sent_or_stale_measurements_are_ignored():
    guard = ImpactGuard(J, effort_abs=20.0)
    for k in range(1, 60):                                                       # 1.2 s: command far, joint still
        t = k * DT
        assert guard.feed(t, {}, pose(0.0), True, efforts={"arm_joint1": 50.0}) is None
        assert guard.feed(t + DT / 2, pose(40.0), pose(0.0), False, efforts={"arm_joint1": 50.0}) is None
    assert guard.good_samples() == []
    # backend velocities replace sample differences: the joint is "moving", so the lag isn't a block
    for k in range(60, 120):
        assert guard.feed(k * DT, pose(40.0), pose(0.0), True, velocities={"arm_joint1": 30.0}) is None


def test_backoff_selection_rules():
    # 1) the newest good sample >= backoff_s before the hit and >= 3 deg from the measurement
    g = ImpactGuard(J, effort_abs=20.0, backoff_s=0.4)
    t = 0.0
    for k in range(100):                                                         # 0 -> 20 deg in 2 s
        t = (k + 1) * DT
        g.feed(t, pose(k * 0.2 + 0.2), pose(k * 0.2), True)
    for k in range(3):
        t += DT
        hit = g.feed(t, pose(20.0), pose(19.8), True, efforts={"arm_joint1": 30.0})
    assert hit is not None
    bp = hit.backoff_pose["arm_joint1"]
    assert bp <= 19.8 - 3.0 + 1e-9
    good_t = dict((round(p["arm_joint1"], 6), ts) for ts, p in g.good_samples())
    assert good_t[round(bp, 6)] <= hit.t - 0.4
    assert bp == pytest.approx(16.4)                                             # the newest such sample (t = 1.66)

    # 1b) slow move: the sample 0.4 s back is too close, so an older one 3 deg from the measurement wins
    g = ImpactGuard(J, effort_abs=20.0, backoff_s=0.4)
    for k in range(200):                                                         # 0 -> 10 deg in 4 s
        g.feed((k + 1) * DT, pose(k * 0.05 + 0.05), pose(k * 0.05), True)
    t = 200 * DT
    for k in range(3):
        t += DT
        hit = g.feed(t, pose(10.0), pose(9.95), True, efforts={"arm_joint1": 30.0})
    assert hit is not None
    bp = hit.backoff_pose["arm_joint1"]
    assert 9.95 - bp >= 3.0 and bp >= 9.95 - 3.0 - 0.05 - 1e-9                  # the first one >= 3 deg away

    # 2) all good samples too close (the arm was standing) -> the oldest good sample
    g = ImpactGuard(J, effort_abs=20.0)
    for k in range(1, 50):
        g.feed(k * DT, pose(10.0 + k * 0.01), pose(10.0 + k * 0.01), True)
    first = g.good_samples()[0][1]
    for k in range(50, 53):
        hit = g.feed(k * DT, pose(10.5), pose(10.5), True, efforts={"arm_joint1": 30.0})
    assert hit is not None and hit.backoff_pose == first

    # 3) no good samples -> the measurement
    g = ImpactGuard(J, effort_abs=20.0, effort_n=1)
    hit = g.feed(0.1, pose(5.0), pose(4.0, arm_joint2=7.0), True, efforts={"arm_joint1": 30.0})
    assert hit is not None and hit.backoff_pose == pose(4.0, arm_joint2=7.0)

    # 4) the buffer keeps only keep_s; reset clears it
    g = ImpactGuard(J, keep_s=1.0)
    for k in range(1, 200):
        g.feed(k * DT, pose(0.0), pose(0.0), True)
    samples = g.good_samples()
    assert samples[-1][0] - samples[0][0] <= 1.0 + 1e-9
    g.reset()
    assert g.good_samples() == []


def test_the_guard_is_deterministic():
    def run() -> Impact | None:
        g = ImpactGuard(J)
        return ramp(g, 0.0, 0.0, 80.0, speed=20.0, lead=1.0, wall=30.0)[1]

    a, b = run(), run()
    assert a == b and a is not None
    assert a.t == pytest.approx(b.t)
