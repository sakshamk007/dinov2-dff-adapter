#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dff_adapter.data import DeepfakeBenchManifest
from dff_adapter.model import DFFAdapterDetector


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_loader(path, cfg, train):
    ds = DeepfakeBenchManifest(path, cfg["image_size"], train=train)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train,
                      num_workers=cfg["train"]["num_workers"], pin_memory=True,
                      persistent_workers=cfg["train"]["num_workers"] > 0)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); ys, scores, vids = [], [], []
    for batch in tqdm(loader, desc="validate", leave=False):
        logits = model(batch["pixel_values"].to(device, non_blocking=True), task_id=0)
        ys.extend(batch["label"].tolist()); scores.extend(torch.sigmoid(logits).cpu().tolist()); vids.extend(batch["video_id"])
    frame_auc = roc_auc_score(ys, scores) if len(set(ys)) == 2 else float("nan")
    per_video = {}
    for y, s, v in zip(ys, scores, vids):
        per_video.setdefault(v, [y, []])[1].append(s)
    vy = [v[0] for v in per_video.values()]
    vs = [float(np.mean(v[1])) for v in per_video.values()]
    return {"frame_auc": frame_auc, "video_auc": roc_auc_score(vy, vs) if len(set(vy)) == 2 else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--resume")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("This training recipe expects a CUDA GPU.")
    train_loader = get_loader(cfg["data"]["train_manifest"], cfg, True)
    val_loader = get_loader(cfg["data"]["val_manifest"], cfg, False)
    model = DFFAdapterDetector(cfg["model_name"], cfg["num_forgery_types"], cfg["adapter"]).to(device)
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    updates_per_epoch = math.ceil(len(train_loader) / cfg["train"]["grad_accum_steps"])
    total_steps = cfg["train"]["epochs"] * updates_per_epoch
    warmup = cfg["train"]["warmup_epochs"] * updates_per_epoch
    scheduler = LambdaLR(opt, lambda step: min(1., (step + 1) / max(1, warmup)) * max(0., 1 - step / total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])
    out = Path(cfg["data"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    start_epoch, best_auc = 0, -1.
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device); model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_auc = ckpt["epoch"] + 1, ckpt["best_auc"]
    accum = cfg["train"]["grad_accum_steps"]
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train(); opt.zero_grad(set_to_none=True); running = 0.
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}")):
            x = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            types = batch["type_label"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=cfg["train"]["amp"]):
                # Two task-wise passes on the exact same mini-batch (paper Eq. 10).
                binary_loss = F.binary_cross_entropy_with_logits(model(x, task_id=0), labels)
                fake = types >= 0
                type_logits = model(x, task_id=1)
                type_loss = F.cross_entropy(type_logits[fake], types[fake]) if fake.any() else binary_loss.new_zeros(())
                loss = (cfg["train"]["lambda_binary"] * binary_loss + cfg["train"]["lambda_type"] * type_loss) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); scheduler.step()
            running += loss.item() * accum
        metrics = evaluate(model, val_loader, device)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_auc": best_auc, "config": cfg, "metrics": metrics}
        torch.save(state, out / "last.pt")
        if metrics["video_auc"] > best_auc:
            best_auc = metrics["video_auc"]; state["best_auc"] = best_auc; torch.save(state, out / "best.pt")
        report = {"epoch": epoch + 1, "train_loss": running / len(train_loader), **metrics, "best_video_auc": best_auc}
        print(json.dumps(report)); (out / "history.jsonl").open("a").write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
