import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.pose import (  # noqa: E402
    SPEC_Q_THRESH_DEG, SPEC_T_THRESH_REL, err_q, err_t, pose_score, spec_score,
)

Q = np.array([0.2799, -0.9071, -0.0787, -0.3042])  # a real label, scalar-first
T = np.array([0.1, -0.2, 5.0])


def rotate_quat(q, axis, deg):
    """Return q composed with an extra rotation of `deg` about `axis` (scalar-first in/out)."""
    r = Rotation.from_rotvec(np.radians(deg) * np.asarray(axis) / np.linalg.norm(axis))
    x, y, z, w = (r * Rotation.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
    return np.array([w, x, y, z])


def test_perfect_prediction_is_zero():
    # TODO (you): err_t, err_q, pose_score, spec_score of (Q, T) vs (Q, T) are all 0
    assert np.isclose(err_t(T, T), 0)
    assert np.isclose(err_q(Q, Q), 0)
    assert np.isclose(pose_score(Q, T, Q, T), 0)
    assert np.isclose(spec_score(Q, T, Q, T), 0)
    
    


def test_sign_ambiguity():
    # TODO (you): err_q(-Q, Q) == 0
    assert np.isclose(err_q(-Q, Q), 0)
    


def test_known_rotation():
    # TODO (you): for a few axes, err_q(rotate_quat(Q, axis, 10), Q) ~= radians(10)
    axes = [[1, 0, 0], [0, 1, 0],  [1, 1, 0]]
    assert all(np.isclose(err_q(rotate_quat(Q, axis, 10), Q), np.radians(10)) for axis in axes)
    


def test_known_translation():
    # TODO (you): err_t(T + [0.1, 0, 0], T) ~= 0.1
    assert np.isclose(err_t(T + [0.1, 0, 0], T), 0.1)
    


def test_unnormalised_quaternion():
    # TODO (you): err_q(3 * Q, Q) == 0 and is not NaN
    Q_  = 0.95 * Q
    assert np.isclose(err_q(Q_, Q), 0)
    assert np.isclose(err_q(3 * Q, Q), 0)
    


def test_spec_threshold():
    # TODO (you): an error just BELOW both thresholds -> spec_score == 0;
    #             rotation just ABOVE (translation still below) -> spec_score == pose_score.
    #             Hint: rotate_quat(Q, [0,0,1], 0.9 * SPEC_Q_THRESH_DEG),
    #                   T + [0, 0, 0.9 * SPEC_T_THRESH_REL * np.linalg.norm(T)]
    assert np.isclose(spec_score(rotate_quat(Q, [0, 0, 1], 0.9 * SPEC_Q_THRESH_DEG),
                                 T + [0, 0, 0.9 * SPEC_T_THRESH_REL *np.linalg.norm(T)], Q, T), 0)
    assert np.isclose(spec_score(rotate_quat(Q, [0, 0, 1], 1.1 * SPEC_Q_THRESH_DEG),
                                 T + [0, 0, 0.9 * SPEC_T_THRESH_REL * np.linalg.norm(T)], Q, T),
                      pose_score(rotate_quat(Q, [0, 0, 1], 1.1 * SPEC_Q_THRESH_DEG), T + [0, 0, 0.9 * SPEC_T_THRESH_REL * np.linalg.norm(T)], Q, T))    
    
