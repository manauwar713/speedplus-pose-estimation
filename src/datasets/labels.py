"""Reading SPEED+ label files.

Each entry: filename, q_vbs2tango_true (4,), r_Vo2To_vbs_true (3,) [m].
The quaternion convention is NOT assumed here; it is verified by the overlay check.
"""
import json
from pathlib import Path

import numpy as np

# split name -> (label file, image dir), relative to the dataset root
SPLITS = {
    "train": ("synthetic/train.json", "synthetic/images"),
    "validation": ("synthetic/validation.json", "synthetic/images"),
    "lightbox": ("lightbox/test.json", "lightbox/images"),
    "sunlamp": ("sunlamp/test.json", "sunlamp/images"),
}


def load_labels(root, split, only_existing=True):
    """Return a list of dicts {path, q, r} for one split.

    only_existing=True drops entries whose image is not on disk. The label
    files cover the full dataset, but locally only a subset is downloaded.
    """
    root = Path(root)
    label_file, img_dir = SPLITS[split]
    entries = json.loads((root / label_file).read_text())
    out = []
    for e in entries:
        path = root / img_dir / e["filename"]
        if only_existing and not path.exists():
            continue
        out.append({
            "path": path,
            "q": np.array(e["q_vbs2tango_true"], dtype=np.float64),
            "r": np.array(e["r_Vo2To_vbs_true"], dtype=np.float64),
        })
    return out
