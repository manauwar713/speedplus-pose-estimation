"""Project the 11 Tango keypoints with GT pose onto images of each domain.

Usage:  python scripts/overlay_keypoints.py [--root data/speedplusv2] [--n 4]
Writes outputs/keypoints/<split>_grid.jpg: full image (left) + zoomed crop (right) per sample.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.datasets.labels import load_labels
from src.geometry.camera import load_camera, project
from src.geometry.keypoints import TANGO_KEYPOINTS
from src.geometry.rotations import pose_to_Rt
from src.utils.vis import crop_around, draw_keypoints, make_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/speedplusv2")
    ap.add_argument("--n", type=int, default=4, help="samples per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/keypoints")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    K, dist = load_camera(Path(args.root) / "camera.json")
    rng = np.random.default_rng(args.seed)

    for split in ["train", "lightbox", "sunlamp"]:
        samples = load_labels(args.root, split)
        tiles = []
        for i in rng.choice(len(samples), size=min(args.n, len(samples)), replace=False):
            e = samples[i]
            img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
            R, t = pose_to_Rt(e["q"], e["r"])
            uv = project(TANGO_KEYPOINTS, R, t, K, dist)
            vis = draw_keypoints(img, uv, f"{split} {e['path'].stem} z={t[2]:.1f}m")
            crop = crop_around(vis, uv)
            crop = cv2.resize(crop, (img.shape[0], img.shape[0]))  # square, image height
            pad = np.zeros((img.shape[0], img.shape[1] - img.shape[0], 3), np.uint8)
            tiles += [vis, cv2.hconcat([crop, pad])]
        path = out_dir / f"{split}_grid.jpg"
        cv2.imwrite(str(path), make_grid(tiles, ncols=2, scale=0.4))
        print("saved", path)


if __name__ == "__main__":
    main()
