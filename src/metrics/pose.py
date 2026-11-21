"""SPEED+ / SPEC2021 pose metrics.

All functions take ONE sample (q: (4,), t: (3,)). Quaternions are scalar-first
[w, x, y, z] like the labels. Angles are returned in RADIANS.
"""
import numpy as np

# SPEC2021 "within calibration error" thresholds (Park et al. 2021)
SPEC_Q_THRESH_DEG = 0.169      # rotation threshold, DEGREES
SPEC_T_THRESH_REL = 2.173e-3   # translation threshold, relative to ||t||


def err_t(t_hat, t):
    """Translation error E_t = ||t_hat - t||_2  [m]."""
    error = np.linalg.norm(t_hat - t)
    return error


def err_q(q_hat, q):
    """Rotation error E_q = 2 * arccos(|<q_hat, q>|)  [rad].

    Must: normalise both quaternions; take abs (q and -q are the same rotation);
    clip the dot product to <= 1.0 before arccos (float noise -> NaN otherwise).
    """
    q_hat = q_hat / np.linalg.norm(q_hat)
    q = q / np.linalg.norm(q)
    E_q = 2 * np.arccos(np.clip(np.abs(np.sum(q_hat * q)), -1.0, 1.0))
    return E_q


def pose_score(q_hat, t_hat, q, t):
    """E_pose = E_q + E_t / ||t||   (unitless: rad + relative translation)."""
    # TODO (you): 1-2 lines, reuse err_q / err_t
    return err_q(q_hat, q) + err_t(t_hat, t) / np.linalg.norm(t)


def spec_score(q_hat, t_hat, q, t):
    """SPEC2021 score E*_pose.

    0 if E_q < 0.169 deg AND E_t/||t|| < 2.173e-3, else E_pose.
    Careful: err_q returns radians, the threshold is in degrees.
    """
    # TODO (you): ~5 lines
    e_q = err_q(q_hat, q)
    e_t = err_t(t_hat, t) / np.linalg.norm(t) if np.linalg.norm(t) != 0 else 0
    if e_q < np.radians(SPEC_Q_THRESH_DEG) and e_t < SPEC_T_THRESH_REL:
        return 0
    else:
        return pose_score(q_hat, t_hat, q, t)


def summarize(per_sample, fail_thresh_deg=10.0):
    """Aggregate a list of dicts {'e_t', 'e_q', 'e_pose', 'e_spec'} over one domain.

    Returns mean/median E_t [m], mean/median E_q [deg], mean E*_pose, and
    failure rate = fraction of samples with E_q > fail_thresh_deg.
    """
    e_t = np.array([s["e_t"] for s in per_sample])
    e_q = np.degrees([s["e_q"] for s in per_sample])
    e_spec = np.array([s["e_spec"] for s in per_sample])
    return {
        "n": len(per_sample),
        "mean_e_t_m": float(e_t.mean()), "median_e_t_m": float(np.median(e_t)),
        "mean_e_q_deg": float(e_q.mean()), "median_e_q_deg": float(np.median(e_q)),
        "mean_e_spec": float(e_spec.mean()),
        "fail_rate": float((e_q > fail_thresh_deg).mean()),
    }
