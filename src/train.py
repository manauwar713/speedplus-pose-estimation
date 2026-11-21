"""Train the heatmap keypoint network.

  python -m src.train --config configs/smoke_cpu.yaml            (laptop CPU smoke test)
  python -m src.train --config configs/baseline_r50.yaml         (Kaggle GPU)
  python -m src.train --config ... --resume outputs/runs/<run>/last.pt
  python -m src.train --config configs/baseline_r50.yaml --set data.root=/kaggle/temp/sp960 train.epochs=1

Each run writes outputs/runs/<name>_<timestamp>/:
  config.yaml (incl. git hash), metrics.csv (one row per epoch), last.pt, best.pt (lowest val E*_pose)
MLflow logging is used if mlflow is installed (optional; CSV is always written).
"""
import argparse
import csv
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.evaluate import build_model, evaluate, load_config, make_dataset
from src.models.loss import HeatmapMSELoss

try:
    import mlflow
except ImportError:
    mlflow = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init_fn(worker_id):
    # Each DataLoader worker gets a COPY of the dataset, including its numpy RNG. Without reseeding,
    # every worker would produce the same box-jitter sequence. torch gives each worker (and each
    # epoch) a different base seed, so derive the numpy RNG from it.
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.root=/x train.epochs=1")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["git_hash"] = git_hash()
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, dc = cfg["train"], cfg["data"]

    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"run dir: {run_dir}   device: {device}   git: {cfg['git_hash']}")

    # ---- data ----
    train_ds = make_dataset(cfg, dc["train_split"], train=True, max_samples=dc.get("train_max_samples"))
    val_ds = make_dataset(cfg, dc["val_split"], train=False, max_samples=dc.get("val_max_samples"))
    loader_kw = dict(num_workers=dc["num_workers"], pin_memory=device.type == "cuda",
                     persistent_workers=dc["num_workers"] > 0)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                          worker_init_fn=worker_init_fn if dc["num_workers"] > 0 else None, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, **loader_kw)
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs, {len(train_dl)} steps/epoch")

    # ---- model / optimisation ----
    model = build_model(cfg).to(device)
    loss_fn = HeatmapMSELoss(cfg["loss"]["invisible"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)  # stepped per batch
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    if mlflow:
        mlflow.set_experiment("speedplus")
        mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params({"git_hash": cfg["git_hash"], **{f"{k}.{kk}": vv for k, v in cfg.items()
                                                            if isinstance(v, dict) for kk, vv in v.items()}})

    csv_path = run_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "val_loss", "mean_e_t_m", "mean_e_q_deg", "median_e_q_deg",
              "mean_e_spec", "fail_rate", "pnp_fail_rate", "mean_peak", "epoch_s"]
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(fields)

    # ---- loop ----
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, batch in enumerate(train_dl):
            x = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
            loss = loss_fn(pred.float(), tgt, vis)  # loss in fp32 for precision (values are ~1e-3)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * len(x); n += len(x)
            if step % tc["log_every"] == 0:
                print(f"  ep {epoch} step {step}/{len(train_dl)} loss {loss.item():.3e} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": run_loss / n}
        last_epoch = epoch == tc["epochs"] - 1
        if (epoch + 1) % tc["val_every"] == 0 or last_epoch:
            row.update(evaluate(model, val_dl, val_ds.K, val_ds.dist, device, cfg, loss_fn))
        row["epoch_s"] = time.time() - t0

        print(f"epoch {epoch}: train {row['train_loss']:.3e}"
              + (f" | val loss {row['val_loss']:.3e}  E_t {row['mean_e_t_m']:.3f} m  "
                 f"E_q {row['mean_e_q_deg']:.2f} deg (med {row['median_e_q_deg']:.2f})  "
                 f"E* {row['mean_e_spec']:.3f}  PnP fail {row['pnp_fail_rate']:.1%}  "
                 f"peak {row['mean_peak']:.2f}" if "val_loss" in row else "")
              + f"  [{row['epoch_s']:.0f}s]", flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in fields])
        if mlflow:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "config": cfg}
        if "mean_e_spec" in row and row["mean_e_spec"] < best:
            best = ck["best"] = row["mean_e_spec"]
            torch.save(ck, run_dir / "best.pt")
        torch.save(ck, run_dir / "last.pt")

    if mlflow:
        mlflow.end_run()
    print(f"done. best val E*_pose = {best:.4f}   ({run_dir})")


if __name__ == "__main__":
    main()
