"""The wrist camera's geometry: projection against training, the policy frame, the drawn dump."""

from __future__ import annotations

import json

import numpy as np
import pytest
from grasp_recording import npz

cv2 = pytest.importorskip("cv2")

from galaxeo.grasp import camera as cam  # noqa: E402
from galaxeo.grasp.contract import (  # noqa: E402
    ARM_BASE_WORLD,
    CX,
    CY,
    FOCAL_PX,
    HEAT_SIGMA_FRAC,
    IMG_H,
    IMG_W,
    NEAR_Z,
    RAW_H,
    RAW_W,
    T_GRIPPER_CAMERA,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)
from galaxeo.grasp.rig import Rig  # noqa: E402

BASE = np.array(ARM_BASE_WORLD)


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def max_pixel_error(got: np.ndarray, want: np.ndarray) -> float:
    known = ~np.isnan(want[:, 0])
    return float(np.abs(got[known] - want[known]).max())


def believed_base(name: str) -> np.ndarray:
    if name == "projection.npz":
        return npz(name, "believed_base")["believed_base"]
    return npz(name, "believed_pos")["believed_pos"] - BASE


# ------------------------------------------------------------------ projection
def test_project_reproduces_the_recorded_training_pixels():
    z = npz("projection.npz", "gl_pos", "gl_R", "believed_base", "uv")
    uv = np.array([cam.project(pose(R, p - BASE), b) for p, R, b in zip(z["gl_pos"], z["gl_R"], z["believed_base"])])
    assert uv.dtype == np.float32
    assert np.array_equal(np.isnan(uv), np.isnan(z["uv"]))
    assert max_pixel_error(uv, z["uv"]) <= 1e-4


def test_the_recorded_pixels_include_no_pixel_and_off_frame_ones():
    uv = npz("projection.npz", "uv")["uv"]
    u = uv[:, 0]
    v = uv[:, 1]
    assert np.isnan(u).any()
    assert ((u < -0.5) | (u > IMG_W - 0.5) | (v < -0.5) | (v > IMG_H - 0.5)).any()


@pytest.mark.parametrize("name", ["projection.npz", "shard.npz"])
def test_believed_uv_at_the_hand_pose_of_the_joints_matches_training(name):
    """`uv_fk` is training's pixel from the gripper_link pose its own model gives at the recorded joints."""
    z = npz(name, "gl_pos_fk", "gl_R_fk", "uv_fk")
    wc = cam.WristCamera()
    uv = np.array([wc.believed_uv(pose(R, p - BASE), b)
                   for p, R, b in zip(z["gl_pos_fk"], z["gl_R_fk"], believed_base(name))])
    assert np.array_equal(np.isnan(uv), np.isnan(z["uv_fk"]))
    assert max_pixel_error(uv, z["uv_fk"]) <= 1e-4


@pytest.mark.parametrize("name", ["projection.npz", "shard.npz"])
def test_training_pixels_lag_the_joints_by_about_a_pixel(name):
    """Training's own dot is computed from poses one 2 ms substep older than its joints.

    Checked within 640 px of the image centre, where the recording measured 1.06 px. A few
    projection samples lie further out (u about -1100, shallow depth): there the same 0.3 mm
    lag moves the pixel by up to 11 px, far beyond anything the heat channel shows.
    """
    z = npz(name, "uv", "uv_fk")
    uv = np.asarray(z["uv"], float)
    uv_fk = np.asarray(z["uv_fk"], float)
    known = ~np.isnan(uv[:, 0])
    near = known & (np.hypot(uv[:, 0] - CX, uv[:, 1] - CY) <= 640.0)
    reach = 3 * HEAT_SIGMA_FRAC * IMG_W
    seen = (np.abs(uv[:, 0] - CX) < IMG_W / 2 + reach) & (np.abs(uv[:, 1] - CY) < IMG_H / 2 + reach)
    assert near.sum() >= 0.95 * known.sum()
    assert seen.sum() > 50
    assert (near & seen).sum() == seen.sum()
    assert np.abs(uv[near] - uv_fk[near]).max() <= 1.2


@pytest.mark.parametrize("depth", [-0.10, 0.0, 0.9 * NEAR_Z])
def test_project_has_no_pixel_at_or_behind_the_camera_plane(depth):
    p = (T_GRIPPER_CAMERA @ np.array([0.01, 0.02, depth, 1.0]))[:3]
    uv = cam.project(np.eye(4), p)
    assert uv.dtype == np.float32
    assert uv.shape == (2,)
    assert np.isnan(uv).all()


def test_project_is_the_pinhole_in_front_of_the_camera():
    p = (T_GRIPPER_CAMERA @ np.array([0.01, -0.02, 0.20, 1.0]))[:3]
    assert cam.project(np.eye(4), p) == pytest.approx([FOCAL_PX * 0.05 + CX, -FOCAL_PX * 0.1 + CY], abs=1e-4)


def test_a_point_that_is_not_a_number_has_no_pixel():
    assert np.isnan(cam.project(np.eye(4), np.array([np.nan, 0.0, 0.0]))).all()


def test_cube_top_uv_is_the_projection_of_the_top_face_centre():
    wc = cam.WristCamera()
    z = npz("shard.npz", "gl_pos_fk", "gl_R_fk")
    hand = pose(z["gl_R_fk"][0], z["gl_pos_fk"][0] - BASE)
    top = np.array([0.02, 0.31, 0.03])
    got = wc.cube_top_uv(hand, top[:2], float(top[2]))
    assert np.array_equal(got, cam.project(hand, top), equal_nan=True)


def test_the_camera_projects_through_its_own_mount():
    mount = T_GRIPPER_CAMERA.copy()
    mount[0, 3] += 0.01
    p = (T_GRIPPER_CAMERA @ np.array([0.0, 0.0, 0.20, 1.0]))[:3]
    assert cam.WristCamera().believed_uv(np.eye(4), p) == pytest.approx([CX, CY], abs=1e-4)
    assert cam.WristCamera(T_cam2hand=mount).believed_uv(np.eye(4), p) != pytest.approx([CX, CY], abs=1.0)


# ------------------------------------------------------------------ policy frame
@pytest.fixture(scope="module")
def upright_dots() -> tuple[np.ndarray, np.ndarray]:
    """An analytic dot pattern seen through the default lens: the UPRIGHT raw frame and the dots.

    Every raw pixel is taken back to the ideal pinhole by the inverse of the lens model; the
    brightness there is a sum of Gaussian dots placed at known policy-frame pixels.
    """
    K = WRIST_K_2026_09_26
    dist = WRIST_DIST_2026_09_26
    ys, xs = np.mgrid[0:RAW_H, 0:RAW_W].astype(np.float64)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    ideal = cv2.undistortPoints(pts, K, dist, None, np.eye(3), np.eye(3), crit).reshape(RAW_H, RAW_W, 2)
    u = FOCAL_PX * ideal[..., 0] + CX
    v = FOCAL_PX * ideal[..., 1] + CY

    dots = np.array([(du, dv) for dv in np.linspace(12.3, 107.7, 5) for du in np.linspace(14.2, 145.9, 6)])
    sigma = 1.5
    img = np.zeros((RAW_H, RAW_W))
    for du, dv in dots:
        img += np.exp(-((u - du) ** 2 + (v - dv) ** 2) / (2 * sigma ** 2))
    grey = np.clip(np.round(255.0 * img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(grey[:, :, None], 3, axis=2)), dots


def centroids(img: np.ndarray, dots: np.ndarray, half: int = 6) -> np.ndarray:
    grey = img[:, :, 1].astype(float)
    out = []
    for du, dv in dots:
        u0 = int(round(du))
        v0 = int(round(dv))
        win = grey[v0 - half:v0 + half + 1, u0 - half:u0 + half + 1]
        yy, xx = np.mgrid[v0 - half:v0 + half + 1, u0 - half:u0 + half + 1]
        out.append(((win * xx).sum() / win.sum(), (win * yy).sum() / win.sum()))
    return np.array(out)


def raw_of(upright: np.ndarray, upside_down: bool) -> np.ndarray:
    return np.ascontiguousarray(upright[::-1, ::-1]) if upside_down else upright


@pytest.mark.parametrize("upside_down", [True, False])
def test_the_policy_frame_undoes_the_lens_and_the_upside_down_mount(upright_dots, upside_down):
    upright, dots = upright_dots
    wc = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26, upside_down=upside_down))
    img = wc.policy_frame(raw_of(upright, upside_down))
    assert img.shape == (IMG_H, IMG_W, 3)
    assert img.dtype == np.uint8
    assert np.hypot(*(centroids(img, dots) - dots).T).max() <= 0.25


@pytest.mark.parametrize("upside_down", [True, False])
def test_the_same_frame_read_with_the_rotation_left_out_or_added_misses_the_dots(upright_dots, upside_down):
    upright, dots = upright_dots
    raw = raw_of(upright, upside_down)
    right = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26,
                                                upside_down=upside_down)).policy_frame(raw)
    wrong = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26,
                                                upside_down=not upside_down)).policy_frame(raw)
    at = np.round(dots).astype(int)
    assert right[at[:, 1], at[:, 0], 1].mean() > 200
    assert wrong[at[:, 1], at[:, 0], 1].mean() < 100


def test_the_default_wrist_camera_uses_the_calibrated_intrinsics_of_the_upside_down_mount(upright_dots):
    upright, _ = upright_dots
    raw = raw_of(upright, True)
    default = cam.WristCamera()
    explicit = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26))
    assert default.intr.upside_down
    assert default.intr.width == RAW_W
    assert default.intr.height == RAW_H
    assert np.array_equal(default.policy_frame(raw), explicit.policy_frame(raw))


@pytest.mark.parametrize("bad", [np.zeros((480, 640, 3), np.uint8), np.zeros((RAW_H, RAW_W, 3), np.float32),
                                 np.zeros((RAW_H, RAW_W), np.uint8), np.zeros((RAW_W, RAW_H, 3), np.uint8)],
                         ids=["640x480", "float", "grey", "transposed"])
def test_a_raw_frame_of_the_wrong_size_or_type_raises(bad):
    with pytest.raises(cam.CameraError):
        cam.WristCamera().policy_frame(bad)


def test_a_black_raw_frame_gives_a_policy_frame():
    assert cam.WristCamera().policy_frame(np.zeros((RAW_H, RAW_W, 3), np.uint8)).shape == (IMG_H, IMG_W, 3)


@pytest.mark.parametrize("upside_down", [True, False])
def test_the_calibrated_intrinsics_are_accepted_either_way_up(upside_down):
    wc = cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, WRIST_DIST_2026_09_26, upside_down=upside_down))
    assert wc.intr.upside_down is upside_down


def test_intrinsics_that_look_past_the_sensor_are_refused():
    # The calibrated K without its barrel distortion shows a strip above the sensor's top row;
    # cv2.remap would fill it black without a word.
    outside = (r"5120 of 640x480 px \(1\.67 %\) of the policy frame lie outside the 1920x1080 sensor "
               r"\(top edge")
    with pytest.raises(cam.CameraError, match=outside):
        cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, np.zeros(5)))


def test_a_lens_model_that_folds_is_refused():
    # A negative k2 turns the barrel back inward towards the corners, inside the sensor.
    with pytest.raises(cam.CameraError, match="folds"):
        cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, np.array([-0.648, -0.5, 0.0, 0.0, 0.0])))


def test_a_lens_model_that_gives_no_pixel_is_refused():
    with pytest.raises(cam.CameraError, match="no pixel"):
        cam.WristCamera(cam.WristIntrinsics(WRIST_K_2026_09_26, np.array([np.nan, 0.0, 0.0, 0.0, 0.0])))


def test_intrinsics_a_few_pixels_past_the_sensor_are_accepted_with_a_warning(caplog):
    # The calibrated view's top row samples the upright raw frame at y = 36; a principal point
    # 38 px higher puts a strip of 146 samples at 640x480 above the sensor.
    K = WRIST_K_2026_09_26.copy()
    K[1, 2] -= 38.0
    with caplog.at_level("WARNING", logger=cam.__name__):
        wc = cam.WristCamera(cam.WristIntrinsics(K, WRIST_DIST_2026_09_26))
    assert "146 of 640x480 px" in caplog.text
    assert "top edge" in caplog.text
    img = wc.policy_frame(np.full((RAW_H, RAW_W, 3), 200, np.uint8))
    assert (img[0] < 200).any()
    assert (img[2:] == 200).all()


def test_the_calibrated_intrinsics_give_no_warning(caplog):
    with caplog.at_level("WARNING", logger=cam.__name__):
        cam.WristCamera()
    assert "outside" not in caplog.text


def test_wrist_intrinsics_carry_upside_down_from_the_rig_file_with_the_default_lens(tmp_path):
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps({"version": 1, "wrist": {"upside_down": False}}), encoding="utf-8")
    intr = cam.wrist_intrinsics(Rig.load(path))
    assert intr.upside_down is False
    assert np.array_equal(intr.K, WRIST_K_2026_09_26)
    assert np.array_equal(intr.dist, WRIST_DIST_2026_09_26)


def test_wrist_intrinsics_carry_the_rig_files_own_lens(tmp_path):
    K = [[1200.0, 0.0, 950.0], [0.0, 1210.0, 530.0], [0.0, 0.0, 1.0]]
    dist = [-0.5, 0.3, 0.0, 0.0, -0.1]
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps({"version": 1, "wrist": {"K": K, "dist": dist}}), encoding="utf-8")
    intr = cam.wrist_intrinsics(Rig.load(path))
    assert intr.upside_down is True
    assert np.array_equal(intr.K, np.array(K))
    assert np.array_equal(intr.dist, np.array(dist))


def test_wrist_intrinsics_of_a_rig_without_a_file_are_the_upside_down_default(tmp_path):
    assert cam.wrist_intrinsics(Rig.load(tmp_path / "none.json")).upside_down is True


# ------------------------------------------------------------------ one frame from the twin
def test_a_frame_at_the_policy_size_is_the_simulated_camera_and_is_used_as_it_is():
    frame = np.random.default_rng(5).integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
    assert cam.WristCamera().policy_image(frame) is frame


def test_a_raw_frame_is_re_projected(upright_dots):
    upright, _ = upright_dots
    raw = raw_of(upright, True)
    wc = cam.WristCamera()
    assert np.array_equal(wc.policy_image(raw), wc.policy_frame(raw))


@pytest.mark.parametrize("bad", [np.zeros((480, 640, 3), np.uint8), np.zeros((IMG_H, IMG_W), np.uint8),
                                 np.zeros((IMG_H, IMG_W, 4), np.uint8), np.zeros((RAW_W, RAW_H, 3), np.uint8)],
                         ids=["640x480", "grey", "rgba", "transposed"])
def test_a_frame_of_any_other_size_names_both_sizes_it_takes(bad):
    with pytest.raises(cam.CameraError, match=r"\(120, 160, 3\).*\(1080, 1920, 3\)"):
        cam.WristCamera().policy_image(bad)


@pytest.mark.parametrize("shape", [(IMG_H, IMG_W, 3), (RAW_H, RAW_W, 3)], ids=["sim", "real"])
def test_a_frame_that_is_not_uint8_is_refused(shape):
    with pytest.raises(cam.CameraError, match="uint8"):
        cam.WristCamera().policy_image(np.zeros(shape, np.float32))


# ------------------------------------------------------------------ dump
@pytest.fixture
def dump_image() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 60, (IMG_H, IMG_W, 3), dtype=np.uint8)


def green_of(out: np.ndarray) -> np.ndarray:
    return (out[:, :, 1] > 150) & (out[:, :, 0] < 100)


def test_draw_dump_upscales_without_changing_the_frame(dump_image):
    before = dump_image.copy()
    out = cam.draw_dump(dump_image, np.array([40.0, 70.0], np.float32))
    assert out.shape == (4 * IMG_H, 4 * IMG_W, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(dump_image, before)
    # Nearest-neighbour upscale away from the marks.
    assert np.array_equal(out[400:, 400:], np.repeat(np.repeat(dump_image, 4, 0), 4, 1)[400:, 400:])


def test_draw_dump_marks_the_pixel_its_dot_limit_and_the_extra_points(dump_image):
    out = cam.draw_dump(dump_image, np.array([40.0, 70.0], np.float32), extra={"cube top": np.array([120.0, 30.0])})
    cu = int(4 * 40.5 - 0.5)
    cv = int(4 * 70.5 - 0.5)
    green = green_of(out)
    assert green[cv, cu - 6:cu + 7].any()                                  # cross
    assert green[cv - 2:cv + 3, cu + 32 - 2:cu + 32 + 3].any()             # the 8 px circle
    magenta = (out[:, :, 0] > 150) & (out[:, :, 2] > 150) & (out[:, :, 1] < 100)
    assert magenta[4 * 30 - 8:4 * 30 + 9, 4 * 120 - 8:4 * 120 + 9].any()


def test_draw_dump_says_when_there_is_no_pixel(dump_image):
    none = cam.draw_dump(dump_image, np.array([np.nan, np.nan], np.float32))
    white = (none > 200).all(axis=2)
    assert not green_of(none).any()
    assert white[:30, :120].any()
    assert not white[30:].any()
