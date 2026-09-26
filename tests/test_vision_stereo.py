"""Stereo reconstruction and GraspGenX frame/protocol helpers. No cameras, no GPU."""

from __future__ import annotations

import json
import threading

import pytest

np = pytest.importorskip("numpy")

from galaxeo.vision.camera import Camera, invert, load_session  # noqa: E402
from galaxeo.vision.frames import GRASP_TO_TOOL, grasp_to_tool, transform_points  # noqa: E402
from galaxeo.vision.stereo import relative_pose, workspace_mask  # noqa: E402


def _cam(t_base, K=None):
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]]) if K is None else K
    T = np.eye(4)
    T[:3, 3] = t_base
    return Camera(K=K, dist=np.zeros(5), T_cam2base=T, size=(320, 240))


def test_load_session_roundtrip(tmp_path):
    cam = _cam([0.1, 0.0, 0.8])
    p = tmp_path / "env.json"
    p.write_text(json.dumps({
        "T_cam2base": cam.T_cam2base.tolist(),
        "K": cam.K.tolist(),
        "dist": cam.dist.tolist(),
        "trusted": True,
    }))
    got = load_session(str(p))
    assert np.allclose(got.K, cam.K)
    assert np.allclose(got.T_cam2base, cam.T_cam2base)


def test_relative_pose_is_the_baseline_in_the_left_camera():
    left = _cam([0, 0, 0])
    right = _cam([0.12, 0, 0])
    R, t = relative_pose(left, right)
    assert np.allclose(R, np.eye(3))
    # OpenCV: P_right = R P_left + t. Left origin in the right camera is -0.12 m.
    assert np.allclose(t, [-0.12, 0, 0])


def test_invert_roundtrip():
    T = np.eye(4)
    T[:3, 3] = [0.3, -0.1, 0.4]
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    assert np.allclose(invert(invert(T)), T)


def test_workspace_mask_keeps_the_table_in_front_of_the_arm():
    xyz = np.array([
        [0.35, 0.0, 0.08],
        [1.5, 0.0, 0.08],
        [0.35, 0.0, -0.02],
    ])
    m = workspace_mask(xyz)
    assert list(m) == [True, False, False]


def test_grasp_to_tool_sends_approach_along_tool_x():
    G = np.eye(4)
    G[:3, 3] = [0.35, 0.02, 0.10]
    T = grasp_to_tool(G)
    assert np.allclose(T[:3, 3], G[:3, 3])
    assert np.allclose(T[:3, 0], G[:3, 2])   # tool approach = grasp +Z
    assert np.allclose(T[:3, 1], G[:3, 0])   # tool close    = grasp +X
    det = np.linalg.det(T[:3, :3])
    assert det == pytest.approx(1.0, abs=1e-6)
    batch = grasp_to_tool(np.stack([G, G]))
    assert batch.shape == (2, 4, 4)
    assert np.allclose(batch[0], T)
    # columns of GRASP_TO_TOOL are orthonormal
    assert np.allclose(GRASP_TO_TOOL.T @ GRASP_TO_TOOL, np.eye(3), atol=1e-9)


def test_transform_points_matches_homogeneous():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(transform_points(p, T), [[1, 2, 3], [2, 2, 3]])


def test_reconstruct_a_fronto_parallel_plane():
    """Known disparity of a textured plane at 0.80 m, 12 cm baseline."""
    cv2 = pytest.importorskip("cv2")
    from galaxeo.vision.stereo import reconstruct

    Z, B = 0.80, 0.12
    h, w = 240, 320
    K = np.array([[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1]])
    left = Camera(K=K, dist=np.zeros(5), T_cam2base=np.eye(4))
    T_r = np.eye(4)
    T_r[:3, 3] = [B, 0, 0]
    right = Camera(K=K, dist=np.zeros(5), T_cam2base=T_r)
    rng = np.random.default_rng(0)
    speckle = rng.integers(0, 256, (h, w), dtype=np.uint8)
    d = K[0, 0] * B / Z
    # shift left → right by -d (right camera is to the +x of left)
    M = np.array([[1, 0, -d], [0, 1, 0]], np.float32)
    right_gray = cv2.warpAffine(speckle, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    L = cv2.cvtColor(speckle, cv2.COLOR_GRAY2BGR)
    R = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    xyz = reconstruct(L, R, left, right, min_depth=0.3, max_depth=1.4, stride=1,
                      num_disparities=80)
    assert len(xyz) > 500
    z = xyz[:, 2]
    assert np.median(z) == pytest.approx(Z, abs=0.08)
    assert np.median(np.abs(z - Z)) < 0.12
