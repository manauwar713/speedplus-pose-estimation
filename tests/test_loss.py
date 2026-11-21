import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.heatmaps import make_heatmaps  # noqa: E402
from src.models.loss import HeatmapMSELoss  # noqa: E402


def _targets(B=2, K=11, seed=0):
    rng = np.random.default_rng(seed)
    kpts = rng.uniform(40, 216, size=(B, K, 2))
    vis = np.ones((B, K))
    vis[:, :2] = 0  # two out-of-crop keypoints per sample
    tgt = np.stack([make_heatmaps(k, v) for k, v in zip(kpts, vis)])
    return torch.tensor(tgt, dtype=torch.float32), torch.tensor(vis, dtype=torch.float32)


def test_zero_when_perfect():
    tgt, vis = _targets()
    for mode in ("mask", "zero"):
        assert HeatmapMSELoss(mode)(tgt.clone(), tgt, vis).item() == 0.0


def test_initial_loss_matches_formula():
    tgt, vis = _targets()
    loss = HeatmapMSELoss("mask")(torch.zeros_like(tgt), tgt, vis).item()
    assert math.isclose(loss, math.pi * 2.0 ** 2 / 64 ** 2, rel_tol=0.02)  # ~3.07e-3


def test_mask_ignores_invisible_maps_zero_does_not():
    tgt, vis = _targets()
    pred = tgt.clone()
    pred[:, :2] = 5.0  # garbage in the out-of-crop maps
    assert HeatmapMSELoss("mask")(pred, tgt, vis).item() == 0.0
    assert HeatmapMSELoss("zero")(pred, tgt, vis).item() > 1.0


def test_gradient_reaches_only_weighted_maps():
    tgt, vis = _targets()
    pred = torch.zeros_like(tgt, requires_grad=True)
    HeatmapMSELoss("mask")(pred, tgt, vis).backward()
    assert pred.grad[:, :2].abs().sum() == 0
    assert pred.grad[:, 2:].abs().sum() > 0


def test_all_invisible_no_nan():
    tgt, vis = _targets()
    loss = HeatmapMSELoss("mask")(torch.randn_like(tgt), tgt, torch.zeros_like(vis))
    assert torch.isfinite(loss) and loss.item() == 0.0
