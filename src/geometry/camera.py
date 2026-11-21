"""Camera model for SPEED+ (pinhole + OpenCV Brown-Conrady distortion)."""
import json
from pathlib import Path

import cv2
import numpy as np


def load_camera(path):
    """Return K (3x3) and dist (5,) from camera.json.

    dist is in OpenCV order [k1, k2, p1, p2, k3], so it can be passed straight
    to cv2.projectPoints / cv2.solvePnP*.
    """
    cam = json.loads(Path(path).read_text())
    K = np.array(cam["cameraMatrix"], dtype=np.float64)
    dist = np.array(cam["distCoeffs"], dtype=np.float64).reshape(-1)
    assert K.shape == (3, 3) and dist.shape == (5,)
    return K, dist


def project(pts_body, R, t, K, dist):
    """Project body-frame 3D points (N, 3) to distorted pixels (N, 2).

    R, t map body -> camera:  p_cam = R @ p_body + t.
    """
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    uv, _ = cv2.projectPoints(np.asarray(pts_body, dtype=np.float64).reshape(-1, 1, 3),
                              rvec, np.asarray(t, dtype=np.float64), K, dist)
    return uv.reshape(-1, 2)
