"""Heatmap loss: per-keypoint MSE, optionally masked by visibility.

pred, target : (B, K, H, W)   network output / make_heatmaps() targets
vis          : (B, K)         1 = keypoint inside the crop, 0 = outside (its target map is all zeros)

Two choices for out-of-crop keypoints (config: loss.invisible) -- YOUR experiment to decide:
  "mask" : weight 0 -> the net gets NO signal for those maps (SimpleBaseline's target_weight).
           Their output is untrained, so the peak may look confident at the crop border.
  "zero" : weight 1 against the all-zero target -> the net learns "outside the crop = flat map",
           which makes the heatmap peak usable as a confidence for those points.

Scale: per-keypoint mean over the H*W pixels, then averaged over the weighted keypoints, so the loss
does not change with batch size or with how many keypoints are visible.
Sanity value: an all-zero prediction gives loss ~ pi * sigma^2 / (H*W)  (sum of a squared Gaussian
of height 1 is pi*sigma^2). For sigma=2, 64x64: ~3.1e-3. This is the loss at initialisation.
"""
import torch
import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    def __init__(self, invisible="mask"):
        super().__init__()
        assert invisible in ("mask", "zero"), invisible
        self.invisible = invisible

    def forward(self, pred, target, vis):
        per_kpt = ((pred - target) ** 2).mean(dim=(2, 3))            # (B, K)
        w = vis.float() if self.invisible == "mask" else torch.ones_like(per_kpt)
        return (per_kpt * w).sum() / w.sum().clamp(min=1.0)
