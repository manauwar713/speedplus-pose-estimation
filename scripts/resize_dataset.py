"""Make a resized copy of SPEED+ (images + labels + a SCALED camera.json).

Intended for Kaggle/Colab: download the Zenodo zip there, run this, upload --dst as a private dataset.
The source can be the extracted folder OR the .zip itself (read member by member, no extraction,
so you never need disk space for both the extracted 1920x1200 set and the output).

  python scripts/resize_dataset.py --src speedplusv2.zip --dst /kaggle/working/speedplus_960 --scale 0.5
  python scripts/resize_dataset.py --src data/speedplusv2 --dst <scratch>/speedplus_960      (local test)

!! CONVENTION (loud): images are resized with cv2.INTER_AREA by an exact factor s. OpenCV's resize
!! maps pixel CENTRES:  u' = s * (u + 0.5) - 0.5. Projection u = fx * x_d + cx  therefore becomes
!!     fx' = s*fx,  fy' = s*fy,  cx' = s*(cx + 0.5) - 0.5,  cy' = s*(cy + 0.5) - 0.5
!! Distortion coefficients are UNCHANGED: they act on normalised coordinates (x/z, y/z), not pixels.
!! Using cx' = s*cx instead would shift every keypoint by 0.25 px at s=0.5 -- small, but a silent bias.
"""
import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

META_FILES = ["LICENSE.md", "camera.json", "synthetic/train.json", "synthetic/validation.json",
              "lightbox/test.json", "sunlamp/test.json"]
IMAGE_DIRS = ["synthetic/images", "lightbox/images", "sunlamp/images"]


def scale_camera(cam, s):
    """camera.json dict -> new dict for images resized by factor s (see module docstring)."""
    K = np.array(cam["cameraMatrix"], dtype=np.float64)
    K2 = K.copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = s * (K[0, 2] + 0.5) - 0.5
    K2[1, 2] = s * (K[1, 2] + 0.5) - 0.5
    out = dict(cam)
    out.update({"cameraMatrix": K2.tolist(), "Nu": round(cam["Nu"] * s), "Nv": round(cam["Nv"] * s),
                "ccx": K2[0, 2], "ccy": K2[1, 2], "resize_scale": s,
                "resize_note": "K scaled for cv2.INTER_AREA resize (pixel-centre convention); "
                               "distCoeffs unchanged; ppx/ppy/fx/fy (metric) are of the ORIGINAL sensor"})
    return out


class Source:
    """Uniform read access to the dataset as a folder or as the Zenodo zip (prefix speedplusv2/)."""

    def __init__(self, src):
        self.src = Path(src)
        self.is_zip = self.src.suffix == ".zip"
        self._zip = None

    def _z(self):
        if self._zip is None:  # opened lazily: one handle per worker process
            self._zip = zipfile.ZipFile(self.src)
            names = self._zip.namelist()
            self.prefix = names[0].split("/")[0] + "/" if names[0].startswith("speedplusv2") else ""
        return self._zip

    def read(self, rel):
        if self.is_zip:
            return self._z().read(self.prefix + rel)
        return (self.src / rel).read_bytes()

    def list_images(self, img_dir):
        if self.is_zip:
            z = self._z()
            pre = self.prefix + img_dir + "/"
            return sorted(n[len(pre):] for n in z.namelist() if n.startswith(pre) and n.endswith(".jpg"))
        return sorted(p.name for p in (self.src / img_dir).glob("*.jpg"))


_SRC = None


def _init(src):
    global _SRC
    _SRC = Source(src)


def _resize_one(job):
    """Worker: read one image from the source, resize, JPEG-encode. Returns (rel_path, jpeg_bytes)."""
    rel, s, quality = job
    img = cv2.imdecode(np.frombuffer(_SRC.read(rel), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, rel
    return rel, enc.tobytes()


class Sink:
    """Write the output either as a folder or as ONE zip file (--dst ending in .zip).

    Zip mode exists for Kaggle: notebook output is size- and FILE-COUNT-limited, so 70k separate
    JPEGs may not be saveable; one stored (uncompressed -- JPEGs don't compress further) zip is.
    Folder mode is resumable (existing images are skipped); zip mode is not.
    """

    def __init__(self, dst):
        self.dst = Path(dst)
        self.zip = None
        if self.dst.suffix == ".zip":
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            self.zip = zipfile.ZipFile(self.dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True)

    def exists(self, rel):
        return self.zip is None and (self.dst / rel).exists()

    def write(self, rel, data):
        if self.zip is not None:
            self.zip.writestr(rel, data)
            return
        out = self.dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(out)  # atomic: an interrupted run never leaves a truncated image

    def close(self):
        if self.zip is not None:
            self.zip.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset folder or speedplusv2.zip")
    ap.add_argument("--dst", required=True, help="output folder, or a .zip file")
    ap.add_argument("--scale", type=float, default=0.5, help="0.5 -> 960x600")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    s = args.scale

    src, sink = Source(args.src), Sink(args.dst)
    for f in META_FILES:
        data = src.read(f)
        if f == "camera.json":
            data = json.dumps(scale_camera(json.loads(data), s), indent=2).encode()
        sink.write(f, data)
    print(f"labels + scaled camera.json written to {args.dst}")

    jobs = [(f"{d}/{fn}", s, args.quality) for d in IMAGE_DIRS for fn in src.list_images(d)]
    todo = [j for j in jobs if not sink.exists(j[0])]
    print(f"{len(jobs)} images, {len(todo)} to resize (x{s})")

    nbytes = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.src,)) as ex:
        for i, (rel, data) in enumerate(ex.map(_resize_one, todo, chunksize=64)):
            sink.write(rel, data)  # only the main process writes -> zip file stays consistent
            nbytes += len(data)
            if i % 2000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    sink.close()
    print(f"done: {len(todo)} images, {nbytes / 1e9:.2f} GB of JPEG -> {args.dst}")


if __name__ == "__main__":
    main()
