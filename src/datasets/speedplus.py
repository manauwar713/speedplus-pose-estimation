"""PyTorch dataset: SPEED+ image -> square crop around the spacecraft + 2D keypoints.

Stage 2 uses the GT box (from projected keypoints); stage 5 replaces it with a detector.

Coordinate frames (keep these straight):
  original  : distorted pixels of the full 1920x1200 image (what projectPoints returns)
  crop      : pixels of the input_size x input_size network input
  A (2x3)   : affine original -> crop.  To go back for PnP: invert it (cv2.invertAffineTransform).
Keypoints stay in DISTORTED pixel coords, so PnP later must get the distortion coeffs.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.labels import load_labels
from src.geometry.camera import load_camera, project
from src.geometry.keypoints import TANGO_KEYPOINTS
from src.geometry.rotations import pose_to_Rt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def square_box(uv, margin=0.2):
    """Square box (cx, cy, side) around points uv (N, 2), side enlarged by `margin` on each side."""
    lo, hi = uv.min(0), uv.max(0)
    cx, cy = (lo + hi) / 2
    side = max(hi - lo) * (1 + 2 * margin)
    return cx, cy, side


def jitter_box(cx, cy, side, rng, shift=0.1, scale=0.2):
    """Random shift (fraction of side) and scale of the box: train-time robustness to a sloppy detector."""
    cx += rng.uniform(-shift, shift) * side
    cy += rng.uniform(-shift, shift) * side
    side *= 1 + rng.uniform(-scale, scale)
    return cx, cy, side


def box_to_affine(cx, cy, side, out_size):
    """2x3 affine that maps the box (original pixels) onto [0, out_size)^2 (crop pixels)."""
    s = out_size / side
    return np.array([[s, 0, out_size / 2 - s * cx],
                     [0, s, out_size / 2 - s * cy]], dtype=np.float64)


def apply_affine(A, uv):
    return uv @ A[:, :2].T + A[:, 2]


class SpeedPlusKeypoints(Dataset):
    """Returns dict:
      image   : float32 tensor (3, S, S), ImageNet-normalised (gray replicated to 3 channels)
      kpts    : float32 (11, 2) keypoints in crop pixels
      vis     : float32 (11,)  1 if keypoint lies inside the crop, else 0 (use as loss mask)
      target  : heatmaps from target_fn(kpts, vis) if given (your heatmap code), else absent
      A       : float64 (2, 3) affine original -> crop
      q, t    : GT pose (scalar-first quaternion, metres)
    """

    def __init__(self, root, split, input_size=256, margin=0.2, train=False,
                 target_fn=None, aug=None, seed=0, max_samples=None):
        self.root = Path(root)
        self.samples = load_labels(root, split)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.K, self.dist = load_camera(self.root / "camera.json")
        self.S, self.margin, self.train = input_size, margin, train
        self.target_fn, self.aug = target_fn, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        e = self.samples[i]
        img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(e["path"])
        R, t = pose_to_Rt(e["q"], e["r"])
        uv = project(TANGO_KEYPOINTS, R, t, self.K, self.dist)

        box = square_box(uv, self.margin)
        if self.train:
            box = jitter_box(*box, self.rng)
        A = box_to_affine(*box, self.S)
        crop = cv2.warpAffine(img, A, (self.S, self.S), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kpts = apply_affine(A, uv)
        if self.aug is not None:  # photometric only for now (stage 3); geometric aug must also move kpts
            crop = self.aug(image=crop)["image"]

        vis = ((kpts >= 0) & (kpts <= self.S - 1)).all(1)

        x = np.repeat(crop[..., None], 3, axis=2).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out = {
            "image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
            "kpts": torch.from_numpy(kpts.astype(np.float32)),
            "vis": torch.from_numpy(vis.astype(np.float32)),
            "A": torch.from_numpy(A),
            "q": torch.from_numpy(e["q"]),
            "t": torch.from_numpy(e["r"]),
            "index": i,
        }
        if self.target_fn is not None:
            out["target"] = torch.from_numpy(self.target_fn(kpts, vis).astype(np.float32))
        return out
