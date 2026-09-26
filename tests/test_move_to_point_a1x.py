from __future__ import annotations

import sys

import numpy as np

import move_to_point_a1x as M


class FakeArm:
    instances = []

    def __init__(self, iface, dry_run=True, tx=False):
        self.iface = iface
        self.dry_run = dry_run
        self.tx = tx
        self.closed = False
        FakeArm.instances.append(self)

    def close(self):
        self.closed = True

    def drain(self):
        pass


class FakeRobot:
    wait_calls = []

    def __init__(self, arm, cam, speed, rate, track_tol, kp, kd, verbose):
        self.arm = arm
        self.cam = cam
        self.chain = type("Chain", (), {"fk": lambda self, q: np.eye(4)})()
        self.aborted = None
        self._waited = False

    def wait_fresh(self, required=False, timeout=3.0):
        FakeRobot.wait_calls.append((required, timeout))
        self._waited = True
        return True

    def q(self):
        if not self._waited:
            raise AssertionError("q() called before wait_fresh()")
        return np.zeros(6)

    def move(self, q_goal, dur=1.5):
        return None


def test_dry_run_waits_for_fresh_feedback_before_q(monkeypatch, capsys):
    FakeArm.instances = []
    FakeRobot.wait_calls = []

    monkeypatch.setattr(M, "A1XArm", FakeArm)
    monkeypatch.setattr(M, "RealRobot", FakeRobot)
    monkeypatch.setattr(M, "Webcam", lambda *args, **kwargs: None)
    monkeypatch.setattr(M, "ik", lambda chain, T_goal, q_start, iters=200, q_bias=None: (q_start, T_goal))
    monkeypatch.setattr(M, "pose_error", lambda T_got, T_goal: np.zeros(6))
    monkeypatch.setitem(sys.modules, "cv2", object())
    monkeypatch.setattr(M, "read_pose", lambda arm, secs: (np.zeros(6), 200.0))
    monkeypatch.setattr(M.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(sys, "argv", ["move_to_point_a1x.py", "--iface", "can0", "--dry-run", "--dx", "0.03"])

    M.main()

    assert [arm.dry_run for arm in FakeArm.instances] == [True, True]
    assert FakeRobot.wait_calls == [(True, 3.0)]
    out = capsys.readouterr().out
    assert "move goal:" in out
