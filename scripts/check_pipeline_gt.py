"""End-to-end pipeline check with GT keypoints (no network) on the local subset.

For every sample of each split, through the real Dataset (real crops, real affine A):
  mode "exact"   : GT crop keypoints                                  -> PnP -> metrics
  mode "heatmap" : GT crop keypoints -> make_heatmaps -> decode       -> PnP -> metrics
Expected: "exact" E*_pose ~ 0 everywhere (any bug in crop/affine/PnP/convention shows up here);
"heatmap" = representation floor (error a PERFECT network would still have).
Use --jitter to use training-time box jitter (some keypoints fall outside the crop -> vis=0).

Usage:  .venv\\Scripts\\python scripts\\check_pipeline_gt.py [--root data/speedplusv2] [--jitter]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.heatmaps import decode_heatmaps, make_heatmaps  # noqa: E402
from src.datasets.speedplus import SpeedPlusKeypoints  # noqa: E402
from src.geometry.keypoints import TANGO_KEYPOINTS  # noqa: E402
from src.geometry.pnp import pose_from_crop  # noqa: E402
from src.metrics.pose import err_q, err_t, spec_score, summarize  # noqa: E402


def run(ds, mode):
    rows, n_fail, n_vis = [], 0, []
    for i in range(len(ds)):
        s = ds[i]
        kpts, vis, A = s["kpts"].numpy(), s["vis"].numpy(), s["A"].numpy()
        q, t = s["q"].numpy(), s["t"].numpy()
        if mode == "exact":
            kd, peaks = kpts, vis  # vis as confidence: out-of-crop points are dropped
        else:
            kd, peaks = decode_heatmaps(make_heatmaps(kpts, vis))
        out = pose_from_crop(kd, peaks, A, TANGO_KEYPOINTS, ds.K, ds.dist)
        n_fail += not out["success"]
        n_vis.append(vis.sum())
        rows.append({"e_t": err_t(out["t"], t), "e_q": err_q(out["q"], q),
                     "e_pose": 0.0, "e_spec": spec_score(out["q"], out["t"], q, t)})
    return summarize(rows), n_fail, float(np.mean(n_vis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/speedplusv2")
    ap.add_argument("--splits", nargs="+", default=["validation", "lightbox", "sunlamp"])
    ap.add_argument("--jitter", action="store_true", help="use train-time box jitter")
    ap.add_argument("--max", type=int, default=None, help="max samples per split")
    args = ap.parse_args()

    print(f"{'split':<11}{'mode':<9}{'n':>5}{'mean Et[m]':>12}{'mean Eq[deg]':>14}"
          f"{'median Eq':>12}{'mean E*':>10}{'PnP fail':>10}{'vis/11':>8}")
    for split in args.splits:
        ds = SpeedPlusKeypoints(args.root, split, train=args.jitter, max_samples=args.max)
        for mode in ["exact", "heatmap"]:
            m, n_fail, nv = run(ds, mode)
            print(f"{split:<11}{mode:<9}{m['n']:>5}{m['mean_e_t_m']:>12.2e}{m['mean_e_q_deg']:>14.2e}"
                  f"{m['median_e_q_deg']:>12.2e}{m['mean_e_spec']:>10.2e}{n_fail:>10}{nv:>8.2f}")


if __name__ == "__main__":
    main()
