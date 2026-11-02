# DFF-Adapter for DINOv2-with-registers: a Reproduction

An independent PyTorch re-implementation of **"Fine-Grained DINO Tuning with Dual
Supervision for Face Forgery Detection"** (Zhang et al., AAAI 2026,
[arXiv:2511.12107](https://arxiv.org/abs/2511.12107)).

The authors have not released their code, so we rebuilt the DeepFake Fine-Grained
Adapter (DFF-Adapter) from the paper alone, prepared FaceForensics++ (c23) with the
DeepfakeBench pipeline, trained on the official splits, and compared our numbers
with the paper's. The full write-up is in the PDF in this repository
(*Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection - A
Reproduction Study*).

**Authors (IIT Bhilai):** Rohan Sen (M26DS010), Saksham Kohli (M26DS011),
Gobinda Prasad Bag (M26DS016)



## 1. The idea in brief

- **Frozen backbone.** `facebook/dinov2-with-registers-large` is never updated.
- **Routed LoRA experts in every block.** The query, value and attention-output
  projections of every Transformer block are wrapped with a DFF-Adapter module.
  Each module has N = 6 low-rank experts (rank 16, alpha 32). The channels are split
  into h = 4 slices, and a router picks the top-3 experts per slice.
- **Dual supervision.** The model learns two tasks: is the face real or fake
  (binary cross-entropy), and which method forged it (cross-entropy over 5 classes:
  real, Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **Shared branch.** Every module has three paths: authenticity, forgery-type and
  shared. The shared path is always on and is trained by both tasks, so fine-grained
  forgery cues learned from the type task reach the authenticity task.
- **Loss:** `L = 10 * L_bce + 2 * L_ftc`.
  Each mini-batch goes through the backbone twice (task flag 0, then 1). At
  inference only the real/fake pass is run.
- **Size.** 3.55M trainable parameters for ViT-L (1.33M for ViT-B), against about
  300M frozen backbone weights.

## 2. Results

All models are trained on FF++ (c23) only. Metric: video-level AUC (frame scores
averaged per video).

| Setting | Paper | Ours |
|---|---|---|
| FF++ test, ViT-L, 50 epochs | 99.56 | **99.62** (frame-level 98.97) |
| FF++ validation (best epoch), ViT-L, 50 epochs | n/a | 99.69 |
| FF++ validation (best epoch), ViT-B, 12 epochs | n/a (paper ViT-B test: 99.61) | 99.55 |
| Trainable parameters, ViT-L | 3.5M | 3.549M |
| Cross-dataset average (CDF-v2, DFDC, CDF-v1, DFDCP) | 93.23 | 88.00 |
| Cross-manipulation average (12 DF40 methods) | 95.20 | 94.16 |

Cross-dataset detail (video-level AUC):

| | CDF-v2 | DFDC | CDF-v1 | DFDCP | Avg. |
|---|---|---|---|---|---|
| Paper | 95.26 | 89.96 | 96.14 | 91.57 | 93.23 |
| Ours | 90.11 | 83.09 | 88.20 | 90.59 | 88.00 |

**Summary.** The in-domain result reproduces fully. The cross-manipulation result
comes close, and we beat the paper on InSwapper, UniFace, Wav2Lip and FOMM.
Face reenactment is our weak spot, mainly HyperReenact (75.78 vs 90.28). Part of the
cross-dataset generalisation is not reproduced yet (5.24 points short on average).
Our cross-dataset numbers sit closer to the paper's single-task variant (no type
loss) than to its full model. We list possible causes in the report, but this is a
hypothesis, not a result.

## 3. Differences from the paper

The paper leaves some details open, so we made these choices:

| Aspect | Paper | Ours |
|---|---|---|
| Task routing | static table of routing logits, top-3 | per-token router plus learned bias, top-3 |
| Shared routing | global softmax over N experts | same |
| Fusion of shared and task updates | concatenation at the CLS token | residual sum inside each wrapped layer |
| Wrapped layers | query, value, dense | query, value, attention output |
| Forgery-type classes | forgery methods | 4 methods + real (5 classes) |
| LR schedule | not stated | cosine decay to 0 |
| Precision | not stated | fp16 AMP, routing kept in fp32 |

## 4. Repository layout

```text
model.py                 model, datasets, training, evaluation (single entry point)
dff-adapter.ipynb        the Kaggle notebook we ran
requirements.txt
datasets/
  readme.md              data-preparation notes
  faceforensics_download_v4.py   official FF++ download script
  extract_compressed_videos.py   video -> frames
  preprocess.py          DeepfakeBench face alignment / cropping
  rearrange.py           writes the DeepfakeBench JSON files (used for cross-dataset tests)
scripts/download_gdrive.py
checkpoints/
  Epochs12/              ViT-B, 12 epochs  (best.pt, last.pt)
  Epochs50/              ViT-L, 50 epochs  (best.pt, last.pt, wandb logs)
papers/                  base paper (AAAI 2026 and arXiv versions)
```

Checkpoints store only the trainable tensors (adapters and heads), about 5 MB
(ViT-B) and 14 MB (ViT-L) for `best.pt`. `last.pt` also stores the optimiser state.

## 5. Setup

```bash
git clone https://github.com/sakshamk007/dinov2-dff-adapter.git
cd dinov2-dff-adapter
pip install -r requirements.txt
```

Needs a CUDA GPU (the code uses `cuda` directly). Internet is needed on the first run
to download the 1.2 GB+ DINOv2 checkpoint from Hugging Face. On Kaggle, turn Internet
on for the first run, then save `/root/.cache/huggingface` as a dataset.

## 6. Data preparation

1. **Get FaceForensics++.** Request access through the official
   [FaceForensics form](https://github.com/ondyari/FaceForensics) and download the
   c23 videos (H.264, CRF 23) with the script you receive, using
   `-d all -c c23 -t videos`.
2. **Crop faces with DeepfakeBench.** Run `datasets/preprocess.py` (based on the
   DeepfakeBench script). It detects the largest face with dlib, aligns it using five
   keypoints, and saves a 256x256 crop (1.3 scale margin) into a `frames/` folder
   next to each video.
3. **(Optional) Upload to Kaggle** as a private dataset so everyone trains from the
   same copy.
4. **Check the layout.** The loader expects:

```text
<root>/
  original_sequences/youtube/c23/frames/<video-id>/<frame>.png
  manipulated_sequences/Deepfakes/c23/frames/<video-id>/<frame>.png
  manipulated_sequences/Face2Face/c23/frames/<video-id>/<frame>.png
  manipulated_sequences/FaceSwap/c23/frames/<video-id>/<frame>.png
  manipulated_sequences/NeuralTextures/c23/frames/<video-id>/<frame>.png
  train.json  val.json  test.json     # official FF++ video-pair lists, found automatically
```

5. **Verify before training** (no GPU or model needed). It prints how many videos and
   frames were found per class and split, and checks that a sample of frames exists:

```bash
python model.py check --root /path/to/FaceForensics
```

**Splits.** The loader follows the official FF++ pair lists (360 / 70 / 70 pairs). A
pair (a, b) means real videos a and b and, for each method, fakes `a_b` and `b_a`.
The split is therefore by video and by identity, so no frame leaks between splits.

**Frame sampling** (evenly spaced frames per video):

| Split | Frames per video | Frames used |
|---|---|---|
| Train | 8 | 28,768 |
| Val | 16 | 11,186 |
| Test | 32 | 22,388 |

The paper does not state its frame counts; these are our choices. One video per class
had no usable face crops in the training set, which is why the totals are slightly
below `videos x frames`. The training and evaluation frame counts can be changed with
`--frames_train` and `--frames_eval`.

## 7. Training

Quick architecture check (random input, prints parameter count and output shapes):

```bash
python model.py sanity --root /path/to/FaceForensics
```

ViT-L, 50 epochs (the paper's setting):

```bash
python model.py train \
  --root /path/to/FaceForensics \
  --model facebook/dinov2-with-registers-large \
  --epochs 50 --bs 24 --lr 2e-4 --wd 1e-5 --lam0 10 --lam1 2 \
  --frames_train 8 --frames_eval 16 \
  --out ./outputs/vitl
```

ViT-B, 12 epochs (cheaper; about 14 minutes per epoch on a Kaggle GPU):

```bash
python model.py train --root /path/to/FaceForensics \
  --model facebook/dinov2-with-registers-base --epochs 12 --out ./outputs/vitb
```

What the training loop does:

- Adam, cosine decay, fp16 mixed precision, gradient clipping at norm 5, seed 0.
- Training augmentation: random horizontal flip and colour jitter (0.1); images are
  resized to 224x224 and normalised with ImageNet statistics.
- Per batch: pass 1 with task flag 0 (real/fake), back-propagate `lam0 * BCE`; pass 2
  with task flag 1 (forgery type), back-propagate `lam1 * CE`; then one optimiser
  step. This gives the same gradients as one combined loss but holds only one pass of
  activations in memory.
- B matrices start at zero, so training begins from the exact frozen DINOv2.
- After each epoch it evaluates on the validation split and keeps the checkpoint with
  the best **video-level AUC**.

Outputs in `--out`:

- `last.pt`: resumable state (adapters, heads, optimiser, scheduler, epoch).
- `best.pt`: best validation video-level AUC (adapters and heads only).

Training resumes automatically from `<out>/last.pt` if it exists, or pass
`--ckpt <path>` to load weights from a specific file. Optional logging to Weights &
Biases: add `--wandb` (set `WANDB_API_KEY` or a Kaggle secret; otherwise it runs
offline).

## 8. Evaluation

**FF++ test split** (the report used 32 frames per test video):

```bash
python model.py eval --root /path/to/FaceForensics \
  --model facebook/dinov2-with-registers-large \
  --ckpt checkpoints/Epochs50/best.pt \
  --split test --frames_eval 32
```

It prints `video AUC / frame AUC`. Use the same `--model` that the checkpoint was
trained with (`-large` for `Epochs50`, `-base` for `Epochs12`).

**Cross-dataset (Celeb-DF v1/v2, DFDCP, DFDC) and DF40.** Preprocess the dataset with
the DeepfakeBench scripts, create its JSON file with `datasets/rearrange.py`, then:

```bash
python model.py eval --data json \
  --json /path/to/CelebDF-v2.json --root /path/to/datasets \
  --ckpt checkpoints/Epochs50/best.pt \
  --split test --frames_eval 32
```

`--anchor` (default `FaceForensics++`) is the folder name used to rewrite the frame
paths stored in the JSON file.

**Metric.** Video-level AUC: average the frame probabilities of each video, then
compute ROC AUC over videos (as in the paper and DeepfakeBench). Frame-level AUC is
also printed.

## 9. Parameter count (ViT-L)

| Component | Parameters |
|---|---|
| Authenticity paths (72 wrapped layers) | 1,328,832 |
| Forgery-type paths | 1,328,832 |
| Shared paths | 885,168 |
| Authenticity head | 1,025 |
| Forgery-type head | 5,125 |
| **Total trainable** | **3,548,982** |

The paper reports 3.5M. For ViT-B (36 wrapped layers) the total is 1,333,662.

## 10. Limitations and future work

- We have not yet run our own component ablations. The ones we plan to run first:
  `lam1 = 0` (no type loss), static routing, no shared path, a type head without the
  "real" class, top-k in {1, 3, 6} and h in {1, 4, 8}.
- Results are from a single run with a single seed. Repeat runs and EER are still to
  do.
- Only the 12 DF40 methods in the paper's main table are tested, not all 40.
- No identity-constrained training, t-SNE or Grad-CAM visualisations yet.
- Exact paper scores depend on the same face crops, official splits, GPU and software
  versions, and preprocessing.

## 11. Acknowledgements and reference

We thank the FaceForensics++ authors for dataset access, the DeepfakeBench team for
the preprocessing and evaluation protocol, and Kaggle for GPU time.

> T. Zhang, P. Yu, Z. Xia, L. Dai, X. Zhou, H. Gao. *Fine-Grained DINO Tuning with
> Dual Supervision for Face Forgery Detection.* arXiv:2511.12107, 2025.

This is our own implementation of the described method, not the authors' official
code.
