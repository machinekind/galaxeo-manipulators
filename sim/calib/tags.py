"""AprilTag (36h11) detection and single-tag pose, via OpenCV's aruco module.

    from calib.tags import detect, tag_pose, corners_in_tag
    seen = detect(rgb)                       # {id: (4, 2) pixel corners}
    T_tag2cam = tag_pose(seen[0], size, K, dist)

Tag frame: origin at the centre, x right, y up, z out of the face toward the
viewer (OpenCV's marker convention). Corners are ordered top-left, top-right,
bottom-right, bottom-left in the tag's own image, which is what `detect`
returns and what `corners_in_tag` assumes.
"""
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_DET = None


def detector():
    global _DET
    if _DET is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        p.minMarkerPerimeterRate = 0.01
        _DET = cv2.aruco.ArucoDetector(DICT, p)
    return _DET


def detect(rgb):
    """{tag id: (4, 2) float32 pixel corners} for every tag found in an RGB image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    corners, ids, _ = detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}


def corners_in_tag(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def tag_pose(corners, size, K, dist=None):
    """4x4 transform of the tag frame in the camera frame from its four corners."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(corners_in_tag(size), np.asarray(corners, float), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.ravel()
    return T


def project(points_cam, K, dist=None):
    """Pixel coordinates of (N, 3) points already in the camera frame."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    px, _ = cv2.projectPoints(np.asarray(points_cam, float), np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1, 2)
