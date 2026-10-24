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
            with torch.autocast("cuda", enabled=False):  # keep routing in fp32
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
    def __init__(self, name="facebook/dinov2-with-registers-large", n_types=5, **kw):
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
FF_FAKES = {"FF-DF": "Deepfakes", "FF-F2F": "Face2Face", "FF-FS": "FaceSwap", "FF-NT": "NeuralTextures"}
IMG_EXT = (".png", ".jpg", ".jpeg")


def make_tf(train, size=224):
    if train:
        return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


SKIP = ("frames", "landmarks", "masks")


def find_root(root):
    """Dir containing manipulated_sequences/ and original_sequences/. If `root` is wrong, scans /kaggle/input."""
    starts = ([root] if os.path.isdir(root) else []) + ["/kaggle/input"]
    for st in starts:
        base = st.rstrip(os.sep).count(os.sep)
        for r, dirs, _ in os.walk(st, followlinks=True):
            if "manipulated_sequences" in dirs and "original_sequences" in dirs:
                return r
            dirs[:] = [d for d in dirs if d not in SKIP]  # never walk into the huge frame folders
            if r.count(os.sep) - base >= 6:
                dirs[:] = []
    raise FileNotFoundError("manipulated_sequences/ + original_sequences/ not found; "
                            "run: find /kaggle/input -maxdepth 6 -type d -name manipulated_sequences")


def find_split_file(base, split):
    p1 = os.path.dirname(base)
    for start in (base, p1, os.path.dirname(p1)):
        if start in ("", "/", "/kaggle", "/kaggle/input"):
            continue
        b = start.rstrip(os.sep).count(os.sep)
        for r, dirs, files in os.walk(start, followlinks=True):
            dirs[:] = [d for d in dirs if d not in SKIP]
            if r.count(os.sep) - b >= 3:
                dirs[:] = []
            if f"{split}.json" in files:
                return os.path.join(r, f"{split}.json")
    raise FileNotFoundError(f"{split}.json not found near {base}")


def load_pairs(fp):
    d = json.load(open(fp))
    pairs = []
    for e in d:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            pairs.append((str(e[0]).zfill(3), str(e[1]).zfill(3)))
        elif isinstance(e, str) and "_" in e:
            a, b = e.split("_"); pairs.append((a.zfill(3), b.zfill(3)))
        else:
            raise ValueError(f"unexpected entry in {fp}: {e!r}")
    return pairs


class FFppDataset(Dataset):
    """Official FF++ split (pairs [a,b]) -> real videos a,b ; fake videos a_b, b_a for each of the 4 methods."""

    def __init__(self, root, split, frames_per_video, train, size=224):
        self.root = find_root(root)
        sf = find_split_file(self.root, split)
        pairs = load_pairs(sf)
        sub = {"FF-real": "original_sequences/youtube/c23/frames"}
        sub.update({k: f"manipulated_sequences/{v}/c23/frames" for k, v in FF_FAKES.items()})
        self.items, stats = [], {}
        for label, rel in sub.items():
            names = [n for a, b in pairs for n in ((a, b) if label == "FF-real" else (f"{a}_{b}", f"{b}_{a}"))]
            ok = 0
            for n in names:
                d = os.path.join(self.root, rel, n)
                if not os.path.isdir(d):
                    continue
                fr = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
                if not fr:
                    continue
                idx = np.unique(np.linspace(0, len(fr) - 1, min(frames_per_video, len(fr))).astype(int))
                for i in idx:
                    self.items.append((os.path.join(d, fr[i]), 0 if label == "FF-real" else 1,
                                       TYPE_MAP[label], f"{label}/{n}"))
                ok += 1
            stats[label] = f"{ok}/{len(names)}"
            print(f"  [{split}] scanned {label}: {ok}/{len(names)} videos", flush=True)
        print(f"[{split}] root={self.root} | {sf} | {len(pairs)} pairs | videos found: {stats} | frames: {len(self.items)}")
        self.tf = make_tf(train, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, yb, yt, vid = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), yb, yt, vid


class DFBenchDataset(Dataset):  # for DeepfakeBench JSON datasets (Celeb-DF, DFDC, ...)
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
        self.tf = make_tf(train, size)

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


def load_trainable(model, sd):
    r = model.load_state_dict(sd["model"], strict=False)
    missing = set(model.trainable_state()) - set(sd["model"])
    assert not r.unexpected_keys and not missing, f"checkpoint mismatch: unexpected={r.unexpected_keys[:3]} missing={list(missing)[:3]}"


def init_wandb(a, run_id=None, job="train"):
    """Optional Weights & Biases. Key from env WANDB_API_KEY or Kaggle Secret 'WANDB_API_KEY'; falls back to offline."""
    if not a.wandb:
        return None
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            key = None
    if key:
        wandb.login(key=key)
    else:
        print("[wandb] no API key found -> running OFFLINE (add Kaggle secret WANDB_API_KEY to sync)")
        os.environ["WANDB_MODE"] = "offline"
    wb = wandb.init(project=a.wandb_project, name=a.wandb_name, id=run_id, resume="allow",
                    job_type=job, config=vars(a), dir=a.out)
    wb.define_metric("global_step")
    for pat in ("train/*", "val/*", "sys/*", "time/*"):
        wb.define_metric(pat, step_metric="global_step")
    return wb


def make_ds(a, split, nfr, train):
    if a.data == "ffpp":
        return FFppDataset(a.root, split, nfr, train)
    return DFBenchDataset(a.json, a.root, split, nfr, train, a.anchor, ff_types=False)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "sanity", "check"])
    ap.add_argument("--data", choices=["ffpp", "json"], default="ffpp")
    ap.add_argument("--json"); ap.add_argument("--root", default="./FaceForensics",
                    help="Path to the extracted FaceForensics++ directory")
    ap.add_argument("--anchor", default="FaceForensics++")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="facebook/dinov2-with-registers-large")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--lam0", type=float, default=10.0); ap.add_argument("--lam1", type=float, default=2.0)
    ap.add_argument("--frames_train", type=int, default=8); ap.add_argument("--frames_eval", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="./outputs",
                    help="Directory for checkpoints and logs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb_project", default="dff-adapter"); ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--log_every", type=int, default=20, help="wandb logging interval (iterations)")
    a = ap.parse_args()
    a.root = os.path.abspath(os.path.expanduser(a.root))
    a.out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(a.out, exist_ok=True)
    if a.json:
        a.json = os.path.abspath(os.path.expanduser(a.json))
    import sys; sys.stdout.reconfigure(line_buffering=True)
    dev = "cuda"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    if a.mode == "check":  # no GPU/model needed: verify paths, counts, split files
        for sp, n in (("train", a.frames_train), ("val", a.frames_eval), ("test", a.frames_eval)):
            ds = FFppDataset(a.root, sp, n, False)
            samp = random.Random(0).sample(ds.items, min(300, len(ds.items)))
            miss = [p for p, *_ in samp if not os.path.exists(p)]
            print(f"  -> {sp}: {len(ds)} frames, missing in 300-sample: {len(miss)} {miss[:2]}")
        return

    model = DFFDino(a.model).to(dev)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"injected linears: {model.n_injected} | trainable params: {ntr/1e6:.2f}M")

    if a.mode == "sanity":
        x = torch.randn(2, 3, 224, 224, device=dev)
        print("auth", model(x, 0).shape, "type", model(x, 1).shape)
        return

    if a.mode == "eval":
        sd = torch.load(a.ckpt, map_location="cpu")
        load_trainable(model, sd)
        ds = make_ds(a, a.split, a.frames_eval, False)
        dl = DataLoader(ds, 64, num_workers=a.workers, pin_memory=True)
        vauc, fauc = evaluate(model, dl, dev)
        print("video AUC / frame AUC:", (vauc, fauc))
        wb = init_wandb(a, None, job="eval")
        if wb:
            wb.summary[f"eval/{a.split}_video_auc"] = vauc
            wb.summary[f"eval/{a.split}_frame_auc"] = fauc
            wb.finish()
        return

    tr = FFppDataset(a.root, "train", a.frames_train, True)
    va = FFppDataset(a.root, "val", a.frames_eval, False)
    print(f"train frames: {len(tr)} | val frames: {len(va)}")
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, 64, num_workers=a.workers, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda")

    start, best, rid = 0, 0.0, None
    last = os.path.join(a.out, "last.pt")
    if a.ckpt is None and os.path.exists(last):
        a.ckpt = last
    if a.ckpt and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location="cpu")
        load_trainable(model, sd)
        rid = sd.get("wandb_id")
        if "opt" in sd:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"]); scaler.load_state_dict(sd["scaler"])
            start, best = sd["epoch"] + 1, sd["best"]
            print(f"resumed from {a.ckpt} at epoch {start}")
        else:
            print(f"loaded weights only from {a.ckpt}; optimizer/schedule restart at epoch 0")

    wb = init_wandb(a, rid, job="train")
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
            if wb and it % a.log_every == 0:
                wb.log({"global_step": ep * len(tl) + it, "epoch": ep + it / len(tl),
                        "train/loss_bce": l0.item() / a.lam0, "train/loss_ftc": l1.item() / a.lam1,
                        "train/loss_bce_weighted": l0.item(), "train/loss_ftc_weighted": l1.item(),
                        "train/lr": sched.get_last_lr()[0], "train/s_per_it": (time.time() - t0) / (it + 1),
                        "sys/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
            if it % 100 == 0:
                print(f"ep{ep} it{it}/{len(tl)} L_bce*λ0={run[0]/(it+1):.3f} L_ftc*λ1={run[1]/(it+1):.3f} "
                      f"{(time.time()-t0)/(it+1):.2f}s/it", flush=True)
        vauc, fauc = evaluate(model, vl, dev)
        print(f"== epoch {ep}: val video AUC {vauc:.4f} frame AUC {fauc:.4f} ({(time.time()-t0)/60:.1f} min)")
        if wb:
            wb.log({"global_step": (ep + 1) * len(tl), "epoch": ep + 1, "val/video_auc": vauc,
                    "val/frame_auc": fauc, "val/best_video_auc": max(best, vauc),
                    "time/epoch_min": (time.time() - t0) / 60})
        if vauc > best:
            best = vauc
            if wb:
                wb.summary["best_val_video_auc"] = best; wb.summary["best_epoch"] = ep
            torch.save({"model": model.trainable_state()}, os.path.join(a.out, "best.pt"))
        torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                    "wandb_id": wb.id if wb else rid}, last)
    if wb:
        bp = os.path.join(a.out, "best.pt")
        if os.path.exists(bp):
            import wandb
            art = wandb.Artifact("dff-adapter-best", type="model"); art.add_file(bp); wb.log_artifact(art)
        wb.finish()


if __name__ == "__main__":
    main()

