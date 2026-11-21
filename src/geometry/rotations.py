"""Quaternion / rotation-matrix conversions."""
import numpy as np


def quat_to_R(q, scalar_first=True):
    """Unit quaternion (4,) -> rotation matrix (3, 3).

    scalar_first=True reads q as [w, x, y, z]; False reads [x, y, z, w].
    Uses the Hamilton convention: R @ v rotates v by q (active rotation).
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q if scalar_first else (q[3], q[0], q[1], q[2])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_Rt(q_vbs2tango, r_Vo2To_vbs):
    """SPEED+ label -> (R, t) mapping Tango body frame to camera (VBS) frame.

    p_cam = R @ p_body + t

    Convention VERIFIED by axis overlays on synthetic, lightbox and sunlamp
    (outputs/overlays, 2026-11-07): q is scalar-first [w, x, y, z] and
    quat_to_R(q) is used directly (no transpose), despite the "vbs2tango" name.
    """
    return quat_to_R(q_vbs2tango, scalar_first=True), np.asarray(r_Vo2To_vbs, dtype=np.float64)


def R_to_quat(R):
    """Rotation matrix (3, 3) -> unit quaternion, scalar-first [w, x, y, z], sign fixed to w >= 0.

    Inverse of quat_to_R(q, scalar_first=True) (up to q == -q). Uses scipy, which is
    scalar-LAST internally -> reordered here; tested in tests/test_pnp.py.
    """
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q
