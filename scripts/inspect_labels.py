"""Sanity-check camera + labels: counts, one sample, image shape, pose statistics."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.labels import SPLITS, load_labels  # noqa: E402
from src.geometry.camera import load_camera  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/speedplusv2")
args = p.parse_args()

K, dist = load_camera(Path(args.root) / "camera.json")
np.set_printoptions(precision=4, suppress=True)
print("K =\n", K, "\ndist =", dist)

for split in SPLITS:
    full = load_labels(args.root, split, only_existing=False)
    local = load_labels(args.root, split)
    e = local[0]
    img = cv2.imread(str(e["path"]), cv2.IMREAD_UNCHANGED)
    r = np.stack([x["r"] for x in full])
    qn = np.linalg.norm(np.stack([x["q"] for x in full]), axis=1)
    print(f"\n[{split}] {len(full)} labels, {len(local)} images on disk")
    print(f"  first: {e['path'].name} q={e['q']} r={e['r']} img={img.shape} {img.dtype}")
    print(f"  depth z: min {r[:, 2].min():.2f}  median {np.median(r[:, 2]):.2f}  max {r[:, 2].max():.2f} m")
    print(f"  |q| range: [{qn.min():.6f}, {qn.max():.6f}]")
