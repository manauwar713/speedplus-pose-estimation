import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry.camera import project  # noqa: E402
from src.geometry.rotations import pose_to_Rt, quat_to_R  # noqa: E402


def test_identity():
    assert np.allclose(quat_to_R([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(quat_to_R([0, 0, 0, 1], scalar_first=False), np.eye(3))


def test_matches_scipy_and_is_rotation():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_R(q, scalar_first=True)
        R_ref = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy is scalar-last
        assert np.allclose(R, R_ref)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_q_and_minus_q_same_rotation():
    q = np.array([0.3, -0.5, 0.1, 0.8])
    assert np.allclose(quat_to_R(q), quat_to_R(-q))


def test_pose_to_Rt_locked_convention():
    # Regression guard for the overlay-verified convention: scalar-first, no transpose.
    q = np.array([0.2799, -0.9071, -0.0787, -0.3042])
    R, t = pose_to_Rt(q, [0.0, 0.0, 5.0])
    assert np.allclose(R, quat_to_R(q, scalar_first=True))
    assert not np.allclose(R, R.T)  # make sure this test sample would catch a transpose
    assert np.allclose(t, [0, 0, 5])


def test_project_matches_pinhole_without_distortion():
    K = np.array([[3000.0, 0, 960], [0, 3000.0, 600], [0, 0, 1]])
    pts = np.array([[0.0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    R, t = np.eye(3), np.array([0.1, -0.2, 5.0])
    uv = project(pts, R, t, K, np.zeros(5))
    p_cam = pts @ R.T + t
    expected = (K @ p_cam.T).T
    expected = expected[:, :2] / expected[:, 2:]
    assert np.allclose(uv, expected)
