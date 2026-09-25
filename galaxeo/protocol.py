"""Galaxea A1X CAN-FD protocol: ids, field scales, encoders and the feedback decoder.

The single source of truth for every host-side tool in this repo (jog_a1x.py,
so101_bridge.py) and for other projects that drive the arm. Pure stdlib, no I/O.

Recovered from the vendor's own binaries (a1_arm_analysis::arm_encode_impl /
arm_decode_impl in libprotocol_modules.so) and verified on the wire; see
docs/PROTOCOL.md. All fields are big-endian int16. The ROS 2 driver keeps its own
copy (ros2_ws/src/galaxea_a1xy_driver/galaxea_a1xy_driver/protocol.py, built in a
container); tests/test_protocol.py pins the two byte for byte.

Safety rules the encoders enforce or document:

  * Only p_des does anything on this transport. Send kp > 0 anyway: kp 0 has
    been seen to leave the arm deaf (diag/REPORT.md), so encode_arm and
    encode_gripper refuse it. encode_command, the raw vendor encoder, does not.
  * FF_RELEASE (2, and 3, 4) makes the motors go limp and freezes telemetry.
    Nothing here sends it; encode_ff only builds the byte.
  * HEARTBEAT_ID 0x023 is what the vendor HDAS emits. It is not required and is
    documented here only so nobody adds it "for completeness".
"""

from __future__ import annotations

import struct
from typing import NamedTuple, Sequence, Union

# ---- CAN ids (all verified on the wire) ------------------------------------------
CMD_ID = 0x050        # host -> arm, 60 B: joint command, 6 joints x 5 int16
GRIP_ID = 0x051       # host -> arm, 10 B: gripper command, 1 group x 5 int16
FB_ID = 0x052         # arm -> host, 48 B: feedback at 200 Hz, 7 groups x 3 int16
FF_ID = 0x053         # host -> arm, 1 B: function frame (enable / release / clear)
STATUS_ID = 0x054     # arm -> host, 16 B, 1 Hz: status word -- NOT temperatures
VERSION_ID = 0x055    # arm -> host, 64 B, 1 Hz: version / serial
HEARTBEAT_ID = 0x023  # host -> arm, 1 B: vendor HDAS heartbeat. Documented, never sent.

# ---- sizes ---------------------------------------------------------------------------
N_JOINTS = 6          # J1..J6
N_GROUPS = 7          # feedback groups: J1..J6 + gripper
GROUP_LEN = 10        # one joint in a command: 5 x int16
CMD_LEN = N_JOINTS * GROUP_LEN   # 60: the vendor sends 60 bytes, not 64
GRIP_LEN = GROUP_LEN             # 10: the gripper ignores a 12-byte frame
FB_LEN = 48
FB_FOOTER = bytes.fromhex("5c2e0024a827")

# ---- scales ---------------------------------------------------------------------------
POS_SCALE = 4700.0    # raw / 4700 -> rad
VEL_SCALE = 750.0     # raw / 750  -> rad/s
EFF_SCALE = 600.0     # raw / 600  -> effort (N m?)

#: (name, clamp_lo, clamp_hi, scale) in command field order. The clamps cross-check
#: against the scales: 6.5 * 4700 = 30550 = 0x7756; 40 * 750 = 200 * 150 = 50 * 600 = 30000.
FIELDS = (
    ("p_des", -6.5, 6.5, 4700.0),
    ("v_des", -40.0, 40.0, 750.0),
    ("kp", 0.0, 500.0, 60.0),
    ("kd", 0.0, 200.0, 150.0),
    ("t_ff", -50.0, 50.0, 600.0),
)

# ---- function frames on 0x053 ---------------------------------------------------------
FF_ENABLE = 1         # engages the motors, but alone leaves the arm deaf to 0x050
FF_RELEASE = 2        # motors limp AND telemetry freezes (3, 4 act the same)
FF_CLEAR = 5          # clears the DISCONNECT bit; briefly disengages the motors
FF_ENABLE2 = 6
RELEASE_CODES = (2, 3, 4)
#: Enable = 1 -> 5 -> 6, always with p_des = measured streamed before, between and
#: after every code (docs/SAFETY.md: enabling with nothing streamed once swung an arm
#: 76 deg at saturated torque). A power-cycled arm obeys 0x050 without it.
ENABLE_SEQUENCE = (FF_ENABLE, FF_CLEAR, FF_ENABLE2)

# ---- joint limits ------------------------------------------------------------------------
JOINT_NAMES = tuple(f"arm_joint{i}" for i in range(1, N_JOINTS + 1))
#: URDF limits [rad] of arm_joint1..6 (docs/HARDWARE.md). J3 reads about +1.5 deg at
#: rest, just past its upper limit: widen these to include the measured start pose.
URDF_LIMITS_RAD = (
    (-2.880, 2.880),   # J1 base yaw
    (0.0, 3.142),      # J2 shoulder pitch
    (-3.316, 0.0),     # J3 elbow pitch
    (-1.571, 1.571),   # J4 wrist pitch
    (-1.571, 1.571),   # J5 wrist yaw
    (-2.880, 2.880),   # J6 wrist roll
)

Number = Union[int, float]


class Feedback(NamedTuple):
    """One decoded 0x052 frame: 7 groups (J1..J6, then the gripper group)."""

    pos: tuple   # rad; group 7 is NOT the gripper position (docs/PROTOCOL.md)
    vel: tuple   # rad/s
    eff: tuple   # effort units (raw / 600)


def _field(value: float, k: int) -> int:
    _name, lo, hi, scale = FIELDS[k]
    value = lo if value < lo else (hi if value > hi else value)
    raw = int(value * scale)                     # fcvtzs: truncate toward zero
    return max(-32768, min(32767, raw))


def encode_group(p: float, v: float, kp: float, kd: float, tff: float) -> bytes:
    """One joint (or the gripper): 10 bytes, fields p_des, v_des, kp, kd, t_ff."""
    out = bytearray(GROUP_LEN)
    for k, val in enumerate((p, v, kp, kd, tff)):
        raw = _field(val, k)
        out[k * 2] = (raw >> 8) & 0xFF           # high byte first
        out[k * 2 + 1] = raw & 0xFF
    return bytes(out)


def _per_joint(value, name: str) -> list:
    if isinstance(value, (int, float)):
        return [float(value)] * N_JOINTS
    vals = [float(x) for x in value]
    if len(vals) != N_JOINTS:
        raise ValueError(f"{name}: expected {N_JOINTS} values, got {len(vals)}")
    return vals


def encode_command(p: Sequence[float], v: Sequence[float] | Number = 0.0,
                   kp: Sequence[float] | Number = 0.0, kd: Sequence[float] | Number = 0.0,
                   tff: Sequence[float] | Number = 0.0) -> bytes:
    """The raw vendor encoder (arm_encode_impl): 60 bytes, six joints.

    Each argument is six values or one value for all joints. Accepts kp 0 (the
    all-zero "zero torque" frame); use encode_arm to drive an arm.
    """
    cols = [_per_joint(x, n) for x, n in ((p, "p"), (v, "v"), (kp, "kp"), (kd, "kd"), (tff, "tff"))]
    return b"".join(encode_group(*(c[j] for c in cols)) for j in range(N_JOINTS))


def encode_arm(p6: Sequence[float], kp: Sequence[float] | Number, kd: Sequence[float] | Number) -> bytes:
    """0x050 payload, 60 bytes: p_des for J1..J6 [rad], v_des = t_ff = 0.

    Raises ValueError for kp <= 0: a kp-0 stream has been seen to leave the arm deaf.
    """
    kps = _per_joint(kp, "kp")
    if min(kps) <= 0.0:
        raise ValueError("kp must be > 0 (kp 0 leaves the arm deaf, diag/REPORT.md)")
    return encode_command(p6, 0.0, kps, kd, 0.0)


def encode_gripper(p: float, kp: float, kd: float) -> bytes:
    """0x051 payload, exactly 10 bytes: the gripper is a seventh joint with the same fields.

    p_des more negative = more open (about -2.0 open, +0.6 closed on the arms measured).
    """
    if kp <= 0.0:
        raise ValueError("gripper kp must be > 0")
    return encode_group(p, 0.0, kp, kd, 0.0)


def encode_ff(code: int) -> bytes:
    """0x053 payload, one byte. Code 0 is rejected by the vendor ('Invalid command')."""
    if not 1 <= int(code) <= 255:
        raise ValueError("function frame code must be 1..255 (0 is invalid)")
    return bytes([int(code)])


def decode_feedback(payload: bytes) -> Feedback:
    """Decode a 0x052 payload. Rejects anything but exactly 48 bytes."""
    payload = bytes(payload)
    if len(payload) != FB_LEN:
        raise ValueError(f"expected {FB_LEN} bytes, got {len(payload)}")
    r = struct.unpack_from(">21h", payload, 0)
    return Feedback(pos=tuple(r[g * 3] / POS_SCALE for g in range(N_GROUPS)),
                    vel=tuple(r[g * 3 + 1] / VEL_SCALE for g in range(N_GROUPS)),
                    eff=tuple(r[g * 3 + 2] / EFF_SCALE for g in range(N_GROUPS)))


def encode_feedback(pos: Sequence[float], vel: Sequence[float] | None = None,
                    eff: Sequence[float] | None = None) -> bytes:
    """Inverse of decode_feedback (48 bytes, with the constant footer) - for fakes and tests."""
    vel = [0.0] * N_GROUPS if vel is None else list(vel)
    eff = [0.0] * N_GROUPS if eff is None else list(eff)
    if not len(pos) == len(vel) == len(eff) == N_GROUPS:
        raise ValueError(f"expected {N_GROUPS} groups")

    def raw(x: float, scale: float) -> int:
        return max(-32768, min(32767, int(round(x * scale))))

    vals = []
    for g in range(N_GROUPS):
        vals += [raw(pos[g], POS_SCALE), raw(vel[g], VEL_SCALE), raw(eff[g], EFF_SCALE)]
    return struct.pack(">21h", *vals) + FB_FOOTER


def selftest() -> None:
    """The saturation bytes read out of the vendor binary, and the frame sizes."""
    assert encode_command([7.0] * 6)[0:2] == bytes([0x77, 0x56])            # 6.5 * 4700 = 30550
    assert encode_command([0.0] * 6, v=99.0)[2:4] == bytes([0x75, 0x30])     # 40 * 750 = 30000
    assert encode_command([0.0] * 6, kd=999.0)[6:8] == bytes([0x75, 0x30])   # 200 * 150 = 30000
    assert encode_command([0.0] * 6, tff=999.0)[8:10] == bytes([0x75, 0x30]) # 50 * 600 = 30000
    assert encode_command([0.0] * 6) == bytes(CMD_LEN)                       # zero torque = all zeros
    assert len(encode_arm([0.0] * 6, 20.0, 1.0)) == CMD_LEN == 60
    assert len(encode_gripper(0.0, 25.0, 1.0)) == GRIP_LEN == 10
    assert encode_ff(1) == b"\x01"
    fb = decode_feedback(encode_feedback([0.1] * 7, [0.2] * 7, [0.3] * 7))
    assert abs(fb.pos[6] - 0.1) < 1e-3 and abs(fb.eff[0] - 0.3) < 1e-2


if __name__ == "__main__":
    selftest()
    print("selftest: 60-byte arm frame, 10-byte gripper frame, saturation bytes, zero torque, "
          "function frame, feedback round trip OK")
