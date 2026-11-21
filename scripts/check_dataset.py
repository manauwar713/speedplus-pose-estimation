"""Visual check of SpeedPlusKeypoints: draws crop-space keypoints on network inputs.

Usage: python scripts/check_dataset.py [--split train] [--train]   -> outputs/dataset/<split>.jpg
Keypoints outside the crop fall off the image (vis = 0 for those).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.datasets.speedplus import IMAGENET_MEAN, IMAGENET_STD, SpeedPlusKeypoints
from src.utils.vis import draw_keypoints, make_grid

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/speedplusv2")
ap.add_argument("--split", default="train")
ap.add_argument("--train", action="store_true", help="enable box jitter")
ap.add_argument("--n", type=int, default=8)
args = ap.parse_args()

ds = SpeedPlusKeypoints(args.root, args.split, train=args.train)
tiles = []
for i in np.random.default_rng(0).choice(len(ds), args.n, replace=False):
    s = ds[i]
    x = s["image"].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
    tiles.append(draw_keypoints(img, s["kpts"].numpy(), radius=2))  # out-of-crop points are clipped by cv2
out = Path("outputs/dataset"); out.mkdir(parents=True, exist_ok=True)
path = out / f"{args.split}{'_jitter' if args.train else ''}.jpg"
cv2.imwrite(str(path), make_grid(tiles, ncols=4, scale=2.0))
print("saved", path, "| visible fraction:", np.mean([ds[i]["vis"].mean().item() for i in range(min(50, len(ds)))]))
