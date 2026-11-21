"""Tests for src/geometry/pnp.py and R_to_quat. Synthetic poses only (no images needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.heatmaps import decode_heatmaps, make_heatmaps  # noqa: E402
from src.datasets.speedplus import apply_affine, box_to_affine, square_box  # noqa: E402
from src.geometry.camera import project  # noqa: E402
from src.geometry.keypoints import TANGO_KEYPOINTS  # noqa: E402
from src.geometry.pnp import DUMMY_Q, crop_to_original, pose_from_crop, solve_pose  # noqa: E402
from src.geometry.rotations import R_to_quat, quat_to_R  # noqa: E402
from src.metrics.pose import err_q, err_t, spec_score  # noqa: E402

# Camera from CLAUDE.md Table 5 (tests must not depend on the data folder)
K = np.array([[2988.58, 0, 960.0], [0, 2988.34, 600.0], [0, 0, 1]])
DIST = np.array([-0.2238, 0.5141, -6.650e-4, -2.140e-4, -0.1312])


def random_pose(rng, z=None):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    z = rng.uniform(2.5, 10) if z is None else z
    t = np.array([rng.uniform(-0.05, 0.05) * z, rng.uniform(-0.05, 0.05) * z, z])
    return q, t


def test_R_to_quat_roundtrip_and_sign():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q, _ = random_pose(rng)
        q2 = R_to_quat(quat_to_R(q))
        assert q2[0] >= 0
        assert np.isclose(np.linalg.norm(q2), 1)
        assert np.allclose(q2, q if q[0] >= 0 else -q, atol=1e-9)


def test_crop_to_original_inverts_affine():
    A = box_to_affine(1000.0, 550.0, 400.0, 256)
    uv = np.array([[900.0, 500.0], [1100.0, 640.0], [1000.0, 550.0]])
    assert np.allclose(crop_to_original(apply_affine(A, uv), A), uv)


def test_exact_points_recover_pose():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q, t = random_pose(rng)
        uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
        out = solve_pose(uv, TANGO_KEYPOINTS, K, DIST)
        assert out["success"] and out["inliers"].all()
        assert err_t(out["t"], t) < 1e-6 * np.linalg.norm(t) + 1e-9
        assert err_q(out["q"], q) < 1e-6
        assert out["reproj_rmse_px"] < 1e-4  # cv2 iterative undistort leaves ~1e-6 px


def test_distortion_matters():
    # Same distorted points, PnP told there is NO distortion -> measurable bias. Guards against
    # someone "simplifying" dist away.
    rng = np.random.default_rng(2)
    q, t = random_pose(rng, z=3.0)
    t[:2] = [0.4, 0.25]  # push target off-centre where distortion is larger
    uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
    good = solve_pose(uv, TANGO_KEYPOINTS, K, DIST)
    bad = solve_pose(uv, TANGO_KEYPOINTS, K, np.zeros(5), reproj_thresh_px=50)
    assert spec_score(good["q"], good["t"], q, t) == 0
    assert err_t(bad["t"], t) > 100 * err_t(good["t"], t)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q, t = random_pose(rng)
        uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
        uv[[2, 9]] += [60.0, -45.0]  # two badly wrong keypoints
        out = solve_pose(uv, TANGO_KEYPOINTS, K, DIST, reproj_thresh_px=5)
        assert out["success"]
        assert not out["inliers"][2] and not out["inliers"][9]
        assert err_q(out["q"], q) < 1e-5


def test_low_confidence_points_are_ignored():
    rng = np.random.default_rng(4)
    q, t = random_pose(rng)
    uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
    conf = np.ones(11)
    uv[:3] = [[1.5, 1.5]] * 3  # what decode returns for an all-zero (vis=0) map...
    conf[:3] = 0.0             # ...with peak 0
    out = solve_pose(uv, TANGO_KEYPOINTS, K, DIST, conf=conf)
    assert out["success"] and out["n_used"] == 8 and not out["inliers"][:3].any()
    assert err_q(out["q"], q) < 1e-6


def test_too_few_points_returns_dummy():
    rng = np.random.default_rng(5)
    q, t = random_pose(rng)
    uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
    conf = np.zeros(11)
    conf[:3] = 1.0
    out = solve_pose(uv, TANGO_KEYPOINTS, K, DIST, conf=conf)
    assert not out["success"] and np.allclose(out["q"], DUMMY_Q)


def test_full_roundtrip_through_heatmaps():
    # GT pose -> project -> crop -> heatmaps -> decode -> inverse crop -> PnP -> metrics.
    # This is the representation floor: a PERFECT network still pays this error.
    rng = np.random.default_rng(6)
    scores = []
    for _ in range(50):
        q, t = random_pose(rng)
        uv = project(TANGO_KEYPOINTS, quat_to_R(q), t, K, DIST)
        A = box_to_affine(*square_box(uv, 0.2), 256)
        kc = apply_affine(A, uv)
        vis = ((kc >= 0) & (kc <= 255)).all(1).astype(float)
        kd, peaks = decode_heatmaps(make_heatmaps(kc, vis))
        out = pose_from_crop(kd, peaks, A, TANGO_KEYPOINTS, K, DIST)
        assert out["success"]
        scores.append(spec_score(out["q"], out["t"], q, t))
    assert np.mean(scores) < 0.01
