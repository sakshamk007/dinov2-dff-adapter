"""
DFF-Adapter (re-implementation from the paper description) on DINOv2-with-registers-BASE.
Single-file script for Kaggle T4. UNTESTED against the authors' code (none is referenced in the paper).

Usage (in a Kaggle cell):
  !python dff_adapter_kaggle.py train \
      --json /kaggle/input/<ds>/dataset_json/FaceForensics++.json \
      --root /kaggle/input/<ds>/FaceForensics++ \
      --frames_train 8 --frames_eval 16 --epochs 12 --bs 24

  !python dff_adapter_kaggle.py eval --ckpt /kaggle/working/best.pt \
      --json /kaggle/input/<ds2>/dataset_json/Celeb-DF-v2.json \
      --root /kaggle/input/<ds2>/Celeb-DF-v2 --anchor Celeb-DF-v2 --split test
"""
import os, json, math, random, argparse, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from transformers import Dinov2WithRegistersModel

TYPE_MAP = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}  # DeepfakeBench label names
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- model
class ExpertPath(nn.Module):
    """Channel axis split into h slices. Each slice is routed over a pool of N LoRA experts
    (d_h -> r_h -> d_h). Task paths: input-dependent router + top-k. Shared path: dense softmax
    over a global logit vector (Eq. 5). Slice outputs are concatenated back to width d."""

    def __init__(self, d, h, N, r_h, topk=3, dense=False):
        super().__init__()
        self.h, self.N, self.topk, self.dense = h, N, topk, dense
        dh = d // h
        self.A = nn.Parameter(torch.randn(N, dh, r_h) / math.sqrt(dh))
        self.B = nn.Parameter(torch.zeros(N, r_h, dh))  # zero-init => adapter starts as identity
        if dense:
            self.Z = nn.Parameter(torch.zeros(N))
        else:
            self.Z = nn.Parameter(torch.zeros(h, N))
            self.Wr = nn.Parameter(torch.randn(h, dh, N) * 0.02)

    def forward(self, x):
        Bsz, L, d = x.shape
        xs = x.reshape(Bsz, L, self.h, d // self.h)
        if self.dense:
            w = torch.softmax(self.Z.float(), -1).view(1, 1, 1, -1).expand(Bsz, L, self.h, -1)
        else:
            logits = torch.einsum("blhd,hdn->blhn", xs.float(), self.Wr.float()) + self.Z.float()
            p = torch.softmax(logits, -1)
            top = p.topk(self.topk, -1)
            w = torch.zeros_like(p).scatter(-1, top.indices, top.values)
            w = w / w.sum(-1, keepdim=True)  # renormalise over top-k (Eq. 2)
        z = torch.einsum("blhd,ndr->blhnr", xs, self.A.to(xs.dtype))
        z = z * w.unsqueeze(-1).to(z.dtype)
        out = torch.einsum("blhnr,nrd->blhd", z, self.B.to(z.dtype))
        return out.reshape(Bsz, L, d)


class DFFLinear(nn.Module):
    """Frozen Linear + (task-specific path + shared path). ctx['task']: 0=authenticity, 1=forgery type."""

    def __init__(self, base, ctx, r=16, alpha=32, h=4, N=6, topk=3):
        super().__init__()
        assert base.in_features == base.out_features
        d = base.in_features
        self.base, self.ctx, self.beta = base, ctx, alpha / r
        r_h = r // h
        self.auth = ExpertPath(d, h, N, r_h, topk)
        self.ftype = ExpertPath(d, h, N, r_h, topk)
        self.shared = ExpertPath(d, h, N, r_h, dense=True)

    def forward(self, x):
        path = self.auth if self.ctx["task"] == 0 else self.ftype
        return self.base(x) + self.beta * (path(x) + self.shared(x))


def inject(backbone, ctx, **kw):
    targets = {"query", "value", "q_proj", "v_proj", "dense", "o_proj"}
    n = 0
    for name, mod in list(backbone.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets and "attention" in f"{name}.{cname}":
                setattr(mod, cname, DFFLinear(child, ctx, **kw))
                n += 1
    return n


class DFFDino(nn.Module):
    def __init__(self, name="facebook/dinov2-with-registers-base", n_types=5, **kw):
        super().__init__()
        self.ctx = {"task": 0}
        self.backbone = Dinov2WithRegistersModel.from_pretrained(name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.n_injected = inject(self.backbone, self.ctx, **kw)
        d = self.backbone.config.hidden_size
        self.head_auth = nn.Linear(d, 1)
        self.head_type = nn.Linear(d, n_types)

    def forward(self, x, task):
        self.ctx["task"] = task
        cls = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head_auth(cls).squeeze(-1) if task == 0 else self.head_type(cls)

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items()
                if any(s in k for s in (".auth.", ".ftype.", ".shared.", "head_auth", "head_type"))}


# ----------------------------------------------------------------- data
class DFBenchDataset(Dataset):
    def __init__(self, json_path, root, split, frames_per_video, train, anchor="FaceForensics++",
                 ff_types=True, size=224):
        data = json.load(open(json_path))
        data = data[next(iter(data))]  # {label: {split: {video: {label, frames}}}}
        self.items = []
        for label, splits in data.items():
            if ff_types and label not in TYPE_MAP:
                continue
            if split not in splits:
                continue
            is_real = "real" in label.lower()
            for vid, info in splits[split].items():
                fr = sorted(info["frames"])
                idx = np.unique(np.linspace(0, len(fr) - 1, min(frames_per_video, len(fr))).astype(int))
                for i in idx:
                    p = fr[i].replace("\\", "/")
                    p = os.path.join(root, p[p.find(anchor) + len(anchor):].lstrip("/"))
                    self.items.append((p, 0 if is_real else 1, TYPE_MAP.get(label, -1), f"{label}/{vid}"))
        if train:
            self.tf = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(),
                                 T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, yb, yt, vid = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), yb, yt, vid


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    fs, fl, vs, vl = [], [], {}, {}
    for x, yb, _, vid in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x.to(dev, non_blocking=True), 0).float()).cpu().numpy()
        for pi, yi, v in zip(p, yb.numpy(), vid):
            fs.append(pi); fl.append(yi); vs.setdefault(v, []).append(pi); vl[v] = yi
    keys = list(vs)
    vauc = roc_auc_score([vl[k] for k in keys], [np.mean(vs[k]) for k in keys])
    fauc = roc_auc_score(fl, fs)
    return vauc, fauc


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "sanity"])
    ap.add_argument("--json"); ap.add_argument("--root"); ap.add_argument("--anchor", default="FaceForensics++")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="facebook/dinov2-with-registers-base")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--lam0", type=float, default=10.0); ap.add_argument("--lam1", type=float, default=2.0)
    ap.add_argument("--frames_train", type=int, default=8); ap.add_argument("--frames_eval", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()
    dev = "cuda"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    model = DFFDino(a.model).to(dev)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"injected linears: {model.n_injected} | trainable params: {ntr/1e6:.2f}M")

    if a.mode == "sanity":
        x = torch.randn(2, 3, 224, 224, device=dev)
        print("auth", model(x, 0).shape, "type", model(x, 1).shape)
        return

    if a.mode == "eval":
        sd = torch.load(a.ckpt, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
        ds = DFBenchDataset(a.json, a.root, a.split, a.frames_eval, False, a.anchor, ff_types=False)
        dl = DataLoader(ds, 64, num_workers=a.workers, pin_memory=True)
        print("video AUC / frame AUC:", evaluate(model, dl, dev))
        return

    tr = DFBenchDataset(a.json, a.root, "train", a.frames_train, True, a.anchor)
    va = DFBenchDataset(a.json, a.root, "val", a.frames_eval, False, a.anchor)
    print(f"train frames: {len(tr)} | val frames: {len(va)}")
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(va, 64, num_workers=a.workers, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.cuda.amp.GradScaler()

    start, best = 0, 0.0
    last = os.path.join(a.out, "last.pt")
    if a.ckpt is None and os.path.exists(last):
        a.ckpt = last
    if a.ckpt and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"]); scaler.load_state_dict(sd["scaler"])
        start, best = sd["epoch"] + 1, sd["best"]
        print(f"resumed from {a.ckpt} at epoch {start}")

    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); run = [0.0, 0.0]
        for it, (x, yb, yt, _) in enumerate(tl):
            x, yb, yt = x.to(dev, non_blocking=True), yb.float().to(dev), yt.to(dev)
            opt.zero_grad(set_to_none=True)
            # pass 1: authenticity (task flag 0). Backward immediately to free activations;
            # gradients of (l0 + l1) == sum of separate backward calls.
            with torch.autocast("cuda", dtype=torch.float16):
                l0 = a.lam0 * F.binary_cross_entropy_with_logits(model(x, 0).float(), yb)
            scaler.scale(l0).backward()
            # pass 2: forgery type (task flag 1)
            with torch.autocast("cuda", dtype=torch.float16):
                l1 = a.lam1 * F.cross_entropy(model(x, 1).float(), yt)
            scaler.scale(l1).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run[0] += l0.item(); run[1] += l1.item()
            if it % 100 == 0:
                print(f"ep{ep} it{it}/{len(tl)} L_bce*λ0={run[0]/(it+1):.3f} L_ftc*λ1={run[1]/(it+1):.3f} "
                      f"{(time.time()-t0)/(it+1):.2f}s/it", flush=True)
        vauc, fauc = evaluate(model, vl, dev)
        print(f"== epoch {ep}: val video AUC {vauc:.4f} frame AUC {fauc:.4f} ({(time.time()-t0)/60:.1f} min)")
        if vauc > best:
            best = vauc
            torch.save({"model": model.trainable_state()}, os.path.join(a.out, "best.pt"))
        torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last)


if __name__ == "__main__":
    main()
