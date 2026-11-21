# Satellite pose estimation on SPEED+ (synthetic → real)

Estimate the 6-DoF pose (rotation **q**, translation **t**) of the Tango spacecraft from a single
grayscale image, training **only on synthetic images** and testing on real hardware-in-the-loop
images (`lightbox`, `sunlamp`) of the SPEED+ dataset. The core problem is the synthetic → real domain gap.

**Status: work in progress** — data, geometry, heatmap targets, model, PnP, metrics and training
loop are implemented and tested; the first baseline training run is next.

## Approach

```
image (1920x1200, gray)
  -> square crop around the spacecraft, resized to 256x256   (stage 2: GT box; stage 5: detector)
  -> ResNet-50 + 3 deconv layers (SimpleBaseline)            -> 11 keypoint heatmaps, 64x64
  -> argmax + sub-pixel decode, peak value = confidence
  -> inverse crop transform -> RANSAC EPnP + Levenberg-Marquardt refinement (with lens distortion)
  -> q, t
```

Heatmap keypoints + PnP are used because they were the most robust to the domain gap in the
SPEED+ paper's baselines. Planned stages: baseline → augmentation for the domain gap → style
randomisation → detector (full pipeline) → confidence / failure detection → ONNX export + latency.

## Results

Metrics as defined for SPEC2021: E_t = ‖t̂ − t‖ [m], E_q = 2·arccos|⟨q̂, q⟩|, E_pose = E_q + E_t/‖t‖,
E*_pose = 0 when within calibration accuracy (E_q < 0.169°, E_t/‖t‖ < 2.173·10⁻³).

| Experiment | synthetic val E*_pose | lightbox E*_pose | sunlamp E*_pose |
|---|---|---|---|
| *(to be filled)* | | | |

Pipeline sanity check (no network, GT keypoints → heatmaps → decode → PnP): E_q ≈ 0.02°, E_t ≈ 1 mm on
all three domains, i.e. the 64×64 heatmap representation itself is not the accuracy bottleneck.

## Repository layout

```
configs/            YAML experiment configs (smoke_cpu.yaml, baseline_r50.yaml)
src/datasets/       labels, crop dataset, heatmap targets + decoding
src/geometry/       camera model, rotations, keypoints, PnP
src/models/         PoseNet (ResNet + deconv head), heatmap loss
src/metrics/        SPEED+ / SPEC2021 metrics
src/train.py        training loop      src/evaluate.py   per-domain evaluation
scripts/            download subset, resize dataset, overlays, pipeline checks
tests/              unit tests (geometry, heatmaps, model, PnP, loss, metrics, resize)
```

## Usage

```bash
pip install torch torchvision          # CPU or CUDA build for your machine
pip install -r requirements.txt
python -m pytest -q tests

# small local subset (HTTP range requests, no full download)
python scripts/download_subset.py --out data/speedplusv2
python -m src.train --config configs/smoke_cpu.yaml

# full dataset, resized (e.g. in a cloud notebook): scales camera.json to match
python scripts/resize_dataset.py --src speedplusv2.zip --dst speedplus_960 --scale 0.5
python -m src.train --config configs/baseline_r50.yaml
python -m src.evaluate --config configs/baseline_r50.yaml --ckpt outputs/runs/<run>/best.pt
```

Conventions: quaternions are scalar-first; `p_cam = R(q)·p_body + t` (verified with projection
overlays on all three domains); keypoints are in distorted pixel coordinates, so the distortion
coefficients are always passed to OpenCV.

## Dataset and citation

Uses the SPEED+ dataset (CC BY-NC-SA 4.0, non-commercial). The dataset is **not** included in this repository.

> T. H. Park, M. Märtens, G. Lecuyer, D. Izzo, S. D'Amico, "SPEED+: Next-Generation Dataset for
> Spacecraft Pose Estimation across Domain Gap", IEEE Aerospace Conference, 2022. arXiv:2110.03101.

The 11 Tango keypoint coordinates are taken from the authors' public baseline
(github.com/tpark94/speedplusbaseline).
