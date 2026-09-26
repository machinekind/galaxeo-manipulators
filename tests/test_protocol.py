"""galaxeo.protocol against the vendor constants and against the ROS 2 driver's encoder.

The ROS 2 package builds in a container and keeps its own protocol.py; it is loaded
here straight from its file (stdlib only) so the two encoders cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import pytest

from galaxeo import protocol as P

ROS_PROTOCOL = (Path(__file__).resolve().parents[1]
                / "ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver/protocol.py")


@pytest.fixture(scope="module")
def ros():
    spec = importlib.util.spec_from_file_location("ros_a1xy_protocol", ROS_PROTOCOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ros_cmd(ros, p, v=0.0, kp=0.0, kd=0.0, tff=0.0):
    def six(x):
        return list(x) if isinstance(x, (list, tuple)) else [float(x)] * 6
    return ros.ArmCommand(p_des=six(p), v_des=six(v), kp=six(kp), kd=six(kd), t_ff=six(tff))


# The vectors of the ROS driver's _selftest (and of this package's selftest).
SELFTEST_VECTORS = [
    dict(p=[7.0] * 6),                 # p_des above +6.5: 0x7756
    dict(p=[0.0] * 6, v=99.0),         # v_des saturates: 0x7530
    dict(p=[0.0] * 6, kd=999.0),       # kd saturates: 0x7530
    dict(p=[0.0] * 6, tff=999.0),      # t_ff saturates: 0x7530
    dict(p=[0.0] * 6),                 # zero torque: 60 x 0x00
]


@pytest.mark.parametrize("vec", SELFTEST_VECTORS)
def test_package_encoder_is_byte_identical_to_the_ros_encoder_on_the_selftest_vectors(ros, vec):
    assert P.encode_command(**vec) == ros.encode_command(_ros_cmd(ros, **vec))


def test_ros_selftest_itself_passes(ros, capsys):
    ros._selftest()


def test_package_encoder_matches_the_ros_encoder_on_random_commands(ros):
    rng = random.Random(7)
    for _ in range(300):
        cols = {k: [rng.uniform(lo * 1.3, hi * 1.3) for _ in range(6)]
                for k, (_n, lo, hi, _s) in zip(("p", "v", "kp", "kd", "tff"), P.FIELDS)}
        assert P.encode_command(**cols) == ros.encode_command(_ros_cmd(ros, **cols))


def test_encode_arm_equals_the_ros_command_with_zero_v_and_tff(ros):
    p = [0.1, 1.2, -0.5, 0.3, -0.2, 2.9]
    assert P.encode_arm(p, 20.0, 1.0) == ros.encode_command(_ros_cmd(ros, p, 0.0, 20.0, 1.0, 0.0))


def test_constants_match_the_ros_driver(ros):
    assert (P.CMD_ID, P.FB_ID, P.FF_ID, P.FB_LEN, P.CMD_LEN) == (ros.CMD_CAN_ID, ros.FB_CAN_ID, ros.FF_CAN_ID,
                                                                ros.FB_LEN, ros.CMD_LEN)
    assert (P.POS_SCALE, P.VEL_SCALE, P.EFF_SCALE) == (ros.POS_DIV, ros.VEL_DIV, ros.EFF_DIV)
    assert P.FIELDS == ros.FIELDS
    assert P.FB_FOOTER == ros.FOOTER
    assert (P.FF_ENABLE, P.FF_RELEASE, P.FF_CLEAR, P.FF_ENABLE2) == (ros.FF_ENABLE, ros.FF_RELEASE,
                                                                      ros.FF_CLEAR, ros.FF_ENABLE2)


def test_decoder_matches_the_ros_decoder(ros):
    payload = P.encode_feedback([0.1, 1.0, -1.5, 0.2, -0.3, 2.0, 0.7], [0.5] * 7, [-1.2] * 7)
    a, b = P.decode_feedback(payload), ros.decode_feedback(payload)
    assert list(a.pos) == b.position and list(a.vel) == b.velocity and list(a.eff) == b.effort


def test_selftest_bytes():
    P.selftest()
    b = P.encode_command([7.0] * 6)
    assert b[0:2] == bytes([0x77, 0x56]) and len(b) == 60
    assert P.encode_command([0.0] * 6, v=99.0)[2:4] == bytes([0x75, 0x30])
    assert P.encode_command([0.0] * 6) == bytes(60)


def test_ids_and_enable_sequence():
    assert (P.CMD_ID, P.GRIP_ID, P.FB_ID, P.FF_ID) == (0x050, 0x051, 0x052, 0x053)
    assert (P.STATUS_ID, P.VERSION_ID, P.HEARTBEAT_ID) == (0x054, 0x055, 0x023)
    assert P.ENABLE_SEQUENCE == (1, 5, 6)
    assert P.FF_RELEASE in P.RELEASE_CODES and P.FF_RELEASE not in P.ENABLE_SEQUENCE


def test_gripper_frame_is_ten_bytes_with_the_arm_joint_layout():
    g = P.encode_gripper(-2.0, 25.0, 1.0)
    assert len(g) == P.GRIP_LEN == 10
    assert g == P.encode_group(-2.0, 0.0, 25.0, 1.0, 0.0)
    assert int.from_bytes(g[0:2], "big", signed=True) == -9400        # -2.0 * 4700
    assert int.from_bytes(g[4:6], "big", signed=True) == 1500         # 25 * 60


def test_truncation_toward_zero_and_int16_clamp():
    assert int.from_bytes(P.encode_group(0.00019, 0, 0, 0, 0)[0:2], "big", signed=True) == 0
    assert int.from_bytes(P.encode_group(-0.00049, 0, 0, 0, 0)[0:2], "big", signed=True) == -2
    assert int.from_bytes(P.encode_group(-99.0, 0, 0, 0, 0)[0:2], "big", signed=True) == -30550


def test_kp_is_never_encoded_as_zero():
    with pytest.raises(ValueError):
        P.encode_arm([0.0] * 6, 0.0, 1.0)
    with pytest.raises(ValueError):
        P.encode_arm([0.0] * 6, [20.0] * 5 + [0.0], 1.0)
    with pytest.raises(ValueError):
        P.encode_gripper(0.0, 0.0, 1.0)
    b = P.encode_arm([0.0] * 6, 20.0, 1.0)
    assert all(int.from_bytes(b[j * 10 + 4:j * 10 + 6], "big") == 1200 for j in range(6))


def test_encode_arm_wants_six_joints():
    with pytest.raises(ValueError):
        P.encode_arm([0.0] * 5, 20.0, 1.0)


def test_function_frame_is_one_byte_and_code_zero_is_invalid():
    assert P.encode_ff(1) == b"\x01" and P.encode_ff(6) == b"\x06"
    with pytest.raises(ValueError):
        P.encode_ff(0)


def test_feedback_decode_round_trips_encode():
    pos = [0.1, 1.2, -1.5, 0.0, 0.3, -2.0, 0.6]
    fb = P.decode_feedback(P.encode_feedback(pos, [0.25] * 7, [1.5] * 7))
    assert all(math.isclose(a, b, abs_tol=0.5 / P.POS_SCALE) for a, b in zip(fb.pos, pos))
    assert all(math.isclose(v, 0.25, abs_tol=1e-3) for v in fb.vel)
    assert all(math.isclose(e, 1.5, abs_tol=1e-3) for e in fb.eff)


def test_feedback_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        P.decode_feedback(bytes(42))
    with pytest.raises(ValueError):
        P.decode_feedback(bytes(64))


def test_urdf_limits_are_the_six_documented_pairs():
    assert len(P.URDF_LIMITS_RAD) == 6 and P.JOINT_NAMES[0] == "arm_joint1"
    assert P.URDF_LIMITS_RAD[2] == (-3.316, 0.0)
    assert all(lo < hi for lo, hi in P.URDF_LIMITS_RAD)
