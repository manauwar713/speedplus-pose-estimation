"""Evaluate a heatmap model: image -> heatmaps -> decode -> PnP -> SPEED+ metrics, per domain.

Used by train.py (validation every few epochs) and from the command line:
  python -m src.evaluate --config configs/smoke_cpu.yaml --ckpt outputs/runs/<run>/best.pt \
      --splits validation lightbox sunlamp
Stage 2: crops come from the GT box (Dataset with train=False) -> NOT a full end-to-end number yet.
"""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.heatmaps import decode_heatmaps, make_heatmaps
from src.datasets.speedplus import SpeedPlusKeypoints
from src.geometry.keypoints import TANGO_KEYPOINTS
from src.geometry.pnp import pose_from_crop
from src.metrics.pose import err_q, err_t, pose_score, spec_score, summarize
from src.models.loss import HeatmapMSELoss
from src.models.pose_net import PoseNet


def make_target_fn(cfg):
    S = cfg["data"]["input_size"]
    # partial (not lambda) so it can be pickled into DataLoader worker processes on Windows
    return partial(make_heatmaps, input_size=S, heatmap_size=S // 4, sigma=cfg["loss"]["sigma"])


def make_dataset(cfg, split, train, max_samples=None):
    d = cfg["data"]
    return SpeedPlusKeypoints(d["root"], split, input_size=d["input_size"], margin=d["margin"],
                              train=train, target_fn=make_target_fn(cfg), seed=cfg["seed"],
                              max_samples=max_samples)


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    return PoseNet(m["backbone"], m["num_keypoints"],
                   pretrained=m["pretrained"] if pretrained is None else pretrained)


@torch.no_grad()
def evaluate(model, loader, K, dist, device, cfg, loss_fn=None):
    """Returns summary dict: summarize() fields + val_loss + pnp_fail_rate + mean_peak."""
    model.eval()
    S, p = cfg["data"]["input_size"], cfg["pnp"]
    rows, losses, n_fail, peaks_all = [], [], 0, []
    for batch in loader:
        hm = model(batch["image"].to(device)).float().cpu()
        if loss_fn is not None:
            losses.append(loss_fn(hm, batch["target"], batch["vis"]).item() * len(hm))
        for b in range(len(hm)):
            kd, peaks = decode_heatmaps(hm[b].numpy(), input_size=S)
            out = pose_from_crop(kd, peaks, batch["A"][b].numpy(), TANGO_KEYPOINTS, K, dist,
                                 min_conf=p["min_conf"], reproj_thresh_crop_px=p["reproj_thresh_crop_px"])
            q, t = batch["q"][b].numpy(), batch["t"][b].numpy()
            n_fail += not out["success"]
            peaks_all.append(peaks)
            rows.append({"e_t": err_t(out["t"], t), "e_q": err_q(out["q"], q),
                         "e_pose": pose_score(out["q"], out["t"], q, t),
                         "e_spec": spec_score(out["q"], out["t"], q, t)})
    res = summarize(rows)
    res["pnp_fail_rate"] = n_fail / max(len(rows), 1)
    res["mean_peak"] = float(np.mean(peaks_all))
    if loss_fn is not None:
        res["val_loss"] = sum(losses) / max(len(rows), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "lightbox", "sunlamp"])
    ap.add_argument("--max", type=int, default=None, help="max samples per split")
    ap.add_argument("--out", default=None, help="write results json here")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    loss_fn = HeatmapMSELoss(cfg["loss"]["invisible"])

    results = {}
    for split in args.splits:
        ds = make_dataset(cfg, split, train=False, max_samples=args.max)
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"])
        results[split] = evaluate(model, dl, ds.K, ds.dist, device, cfg, loss_fn)
        r = results[split]
        print(f"{split:<11} n={r['n']:<5} E_t={r['mean_e_t_m']:.3f} m  E_q={r['mean_e_q_deg']:.2f} deg "
              f"(median {r['median_e_q_deg']:.2f})  E*={r['mean_e_spec']:.3f}  "
              f"fail(>10deg)={r['fail_rate']:.2%}  PnP fail={r['pnp_fail_rate']:.2%}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
