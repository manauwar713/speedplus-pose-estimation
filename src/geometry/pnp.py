"""2D keypoints -> 6-DoF pose with RANSAC PnP + Levenberg-Marquardt refinement.

Pipeline position:
  decode_heatmaps -> kpts (crop px) + peaks
    -> crop_to_original (inverse affine)       -> distorted ORIGINAL pixels
    -> solve_pose (RANSAC EPnP + RefineLM)     -> R, t, q   (p_cam = R p_body + t)

Convention notes (LOUD):
  * Image points are DISTORTED pixels of the full 1920x1200 image (the dataset applies the affine to
    the output of cv2.projectPoints, which includes distortion). So `dist` MUST be passed to PnP;
    OpenCV undistorts the points internally. Passing dist=None would bias the pose, worst near the
    image border.
  * OpenCV returns rvec/tvec mapping object (body) -> camera, i.e. exactly our p_cam = R p_body + t.
    No transpose. q comes out scalar-first via R_to_quat.
  * RANSAC threshold is given in CROP pixels and converted to original pixels with the crop scale,
    so it means the same thing for a close (large) and a far (small) spacecraft.
"""
import cv2
import numpy as np

from src.geometry.rotations import R_to_quat

# Pose returned when PnP cannot run (too few confident points) or fails. The HigherHRNet entry in
# the SPEED+ paper does the same: a failure must still be scored, not silently dropped.
DUMMY_Q = np.array([1.0, 0.0, 0.0, 0.0])
DUMMY_T = np.array([0.0, 0.0, 6.0])  # roughly the middle of the 2-10 m depth range


def crop_to_original(kpts_crop, A):
    """Crop pixels (N, 2) -> original (distorted) pixels (N, 2), using A (2x3 affine original->crop)."""
    A_inv = cv2.invertAffineTransform(np.asarray(A, dtype=np.float64))
    return np.asarray(kpts_crop, dtype=np.float64) @ A_inv[:, :2].T + A_inv[:, 2]


def solve_pose(uv, pts3d, K, dist, conf=None, min_conf=0.1, reproj_thresh_px=8.0,
               min_points=4, refine=True):
    """Estimate the pose from 2D-3D correspondences.

    uv        : (N, 2) distorted pixels in the ORIGINAL image
    pts3d     : (N, 3) body-frame keypoints [m]
    conf      : (N,) per-keypoint confidence (heatmap peak) or None (= all used)
    min_conf  : points with conf < min_conf are dropped before RANSAC (e.g. vis=0 maps -> peak 0)
    reproj_thresh_px : RANSAC inlier threshold in ORIGINAL pixels

    Returns dict: success, q (4,), t (3,), R (3,3), inliers (N,) bool, n_used, reproj_rmse_px
    (RMSE over inliers, original px; a stage-6 confidence signal). On failure: dummy pose.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
    N = len(uv)
    used = np.ones(N, bool) if conf is None else (np.asarray(conf) >= min_conf)
    used &= np.isfinite(uv).all(1)

    fail = {"success": False, "q": DUMMY_Q.copy(), "t": DUMMY_T.copy(), "R": np.eye(3),
            "inliers": np.zeros(N, bool), "n_used": int(used.sum()), "reproj_rmse_px": np.inf}
    if used.sum() < min_points:
        return fail

    idx = np.flatnonzero(used)
    obj, img = pts3d[idx], uv[idx]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP, reprojectionError=reproj_thresh_px,
        iterationsCount=200, confidence=0.999)
    if not ok or inl is None or len(inl) < min_points:
        return fail
    inl = inl.reshape(-1)

    if refine:  # LM on inliers only: minimises reprojection error starting from the EPnP solution
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    if t[2] <= 0:  # spacecraft behind the camera: geometrically invalid solution
        return fail

    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img[inl]) ** 2, axis=1))))

    inliers = np.zeros(N, bool)
    inliers[idx[inl]] = True
    return {"success": True, "q": R_to_quat(R), "t": t, "R": R, "inliers": inliers,
            "n_used": int(used.sum()), "reproj_rmse_px": rmse}


def pose_from_crop(kpts_crop, peaks, A, pts3d, K, dist, min_conf=0.1, reproj_thresh_crop_px=8.0,
                   **kw):
    """Convenience: decoded crop keypoints -> pose. Threshold given in CROP px (8 = 2 heatmap px).

    Crop scale s = A[0, 0] (crop px per original px), so thresh_original = thresh_crop / s.
    """
    uv = crop_to_original(kpts_crop, A)
    s = float(np.asarray(A)[0, 0])
    return solve_pose(uv, pts3d, K, dist, conf=peaks, min_conf=min_conf,
                      reproj_thresh_px=reproj_thresh_crop_px / s, **kw)
