"""Download a small, reproducible subset of SPEED+ directly from the Zenodo zip.

Uses HTTP range requests (remotezip), so only the requested members are fetched,
not the whole ~16 GB archive. All label files are always downloaded in full.

Usage:
    python scripts/download_subset.py --out data/speedplusv2 \
        --n-train 1000 --n-val 200 --n-lightbox 300 --n-sunlamp 300
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteIOError, RemoteZip
from tqdm import tqdm

URL = "https://zenodo.org/api/records/5588480/files/speedplusv2.zip/content"
ROOT = "speedplusv2/"
META_FILES = [
    "LICENSE.md",
    "camera.json",
    "synthetic/train.json",
    "synthetic/validation.json",
    "lightbox/test.json",
    "sunlamp/test.json",
]
# (label file, image dir, CLI arg name)
SPLITS = [
    ("synthetic/train.json", "synthetic/images", "n_train"),
    ("synthetic/validation.json", "synthetic/images", "n_val"),
    ("lightbox/test.json", "lightbox/images", "n_lightbox"),
    ("sunlamp/test.json", "sunlamp/images", "n_sunlamp"),
]


_local = threading.local()


def fetch(member, dst):
    """Download one zip member; each thread keeps its own RemoteZip connection."""
    for attempt in range(8):
        try:
            if not hasattr(_local, "zip"):
                _local.zip = RemoteZip(URL)
            data = _local.zip.read(member)
            break
        except RemoteIOError:  # Zenodo rate-limits (HTTP 429): back off and retry
            time.sleep(min(2 ** attempt, 60))
    else:
        print(f"giving up on {member}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)  # atomic: an interrupted run never leaves a truncated image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/speedplusv2")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-lightbox", type=int, default=300)
    p.add_argument("--n-sunlamp", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)

    with RemoteZip(URL) as z:
        names = set(z.namelist())

        for f in META_FILES:
            dst = out / f
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(ROOT + f))
        print("labels + camera.json downloaded")

        manifest = {}
        for label_file, img_dir, arg in SPLITS:
            labels = json.loads((out / label_file).read_text())
            n = min(getattr(args, arg), len(labels))
            chosen = sorted(rng.sample([l["filename"] for l in labels], n))
            manifest[label_file] = chosen

            todo = [fn for fn in chosen
                    if not (out / img_dir / fn).exists() and f"{ROOT}{img_dir}/{fn}" in names]
            (out / img_dir).mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(args.workers) as ex:
                list(tqdm(ex.map(lambda fn: fetch(f"{ROOT}{img_dir}/{fn}", out / img_dir / fn), todo),
                          total=len(todo), desc=label_file))

        (out / "subset_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done -> {out.resolve()}")


if __name__ == "__main__":
    main()
