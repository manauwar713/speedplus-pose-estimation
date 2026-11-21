"""Resizing convention: K scaled by scripts/resize_dataset.scale_camera must match cv2.resize."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.resize_dataset import scale_camera  # noqa: E402
from src.geometry.camera import project  # noqa: E402
from src.geometry.keypoints import TANGO_KEYPOINTS  # noqa: E402
from src.geometry.rotations import quat_to_R  # noqa: E402

CAM = {"Nu": 1920, "Nv": 1200,
       "cameraMatrix": [[2988.58, 0, 960.0], [0, 2988.34, 600.0], [0, 0, 1]],
       "distCoeffs": [-0.2238, 0.5141, -6.650e-4, -2.140e-4, -0.1312]}


def test_scaled_K_matches_pixel_centre_rule():
    rng = np.random.default_rng(0)
    K, dist = np.array(CAM["cameraMatrix"]), np.array(CAM["distCoeffs"])
    for s in (0.5, 0.4):
        K2 = np.array(scale_camera(CAM, s)["cameraMatrix"])
        for _ in range(20):
            q = rng.normal(size=4)
            t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(3, 10)])
            uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, dist)
            uv2 = project(TANGO_KEYPOINTS, quat_to_R(q), t, K2, dist)  # SAME dist
            assert np.allclose(uv2, s * (uv + 0.5) - 0.5, atol=1e-9)


def test_cv2_resize_follows_pixel_centre_rule():
    # Empirical check of the rule itself: a blob at a known sub-pixel spot, resized with INTER_AREA,
    # must land where s*(u+0.5)-0.5 says (centroid), not at s*u.
    u0, v0, s = 801.3, 433.7, 0.5
    ys, xs = np.mgrid[0:1200, 0:1920]
    img = np.exp(-((xs - u0) ** 2 + (ys - v0) ** 2) / (2 * 6.0 ** 2))
    small = cv2.resize(img, (960, 600), interpolation=cv2.INTER_AREA)
    ys2, xs2 = np.mgrid[0:600, 0:960]
    cu, cv = (small * xs2).sum() / small.sum(), (small * ys2).sum() / small.sum()
    assert abs(cu - (s * (u0 + 0.5) - 0.5)) < 0.01 and abs(cv - (s * (v0 + 0.5) - 0.5)) < 0.01
    assert abs(cu - s * u0) > 0.2  # the naive rule is measurably wrong


def test_scale_one_is_identity():
    assert np.allclose(scale_camera(CAM, 1.0)["cameraMatrix"], CAM["cameraMatrix"])
