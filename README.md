# DFF-Adapter for DINOv2-with-registers

This is a Kaggle-oriented implementation of *Fine-Grained DINO Tuning with Dual
Supervision for Face Forgery Detection* (arXiv:2511.12107). It freezes
`facebook/dinov2-with-registers-large`, then puts routed low-rank adapters in
the query, value and attention-output projections of every ViT block.

The two passes in each minibatch reproduce the paper's dual supervision:
authenticity (`real` / `fake`) and forgery-method classification. The latter is
computed only for fake frames; real frames use `type_label: -1` and are ignored
by that loss.

## 1. Create the Kaggle notebook

1. Create a new Kaggle notebook, select **GPU** (T4/P100 works with the
   default effective batch size 24; a 24 GB GPU can set `batch_size: 24` and
   `grad_accum_steps: 1`), and turn **Internet on** for the first run. Internet
   is needed only to download the DINOv2 Hugging Face checkpoint and git repo.
2. Add a Kaggle dataset containing extracted, aligned face frames from
   DeepfakeBench / FaceForensics++ c23. Do not point training at videos: this
   code deliberately trains on frames and averages frame scores per video.
3. In a first notebook cell, copy this repository (uploaded as a Kaggle Dataset
   or cloned from your own GitHub mirror) into `/kaggle/working/dff-adapter`.
   Then run:

```bash
%cd /kaggle/working/dff-adapter
!pip install -q -r requirements.txt
```

## 2. Prepare DeepfakeBench data

Follow DeepfakeBench's documented FF++ preprocessing to extract **face crops**
from the c23 videos. A typical mounted dataset layout is:

```text
/kaggle/input/deepfakebench-ffpp/
  original_sequences/.../c23/<video-id>/<frame>.png
  manipulated_sequences/Deepfakes/.../c23/<video-id>/<frame>.png
  manipulated_sequences/Face2Face/.../c23/<video-id>/<frame>.png
  manipulated_sequences/FaceSwap/.../c23/<video-id>/<frame>.png
  manipulated_sequences/NeuralTextures/.../c23/<video-id>/<frame>.png
```

Build leakage-safe manifests. The script samples up to 32 frames/video and
splits **by video**, never by individual frame:

```bash
!python scripts/build_ffpp_manifest.py \
  --root /kaggle/input/deepfakebench-ffpp \
  --output-dir /kaggle/working/manifests \
  --official-splits /kaggle/input/deepfakebench-ffpp/splits \
  --frames-per-video 32
!head -1 /kaggle/working/manifests/ffpp_train.jsonl
```

`--official-splits` is strongly recommended for reported FF++ results: provide
the directory containing the standard `train.json`, `val.json`, and `test.json`
pair lists. Without it, the script makes a reproducible 80/10/10 split by video
folder (useful for a quick smoke test, but not a DeepfakeBench comparison).

The expected row schema is `path`, `label`, `type_label`, and `video_id`.
If your DeepfakeBench export has different folder names, change only `--types`
or edit `class_for()` in `scripts/build_ffpp_manifest.py`; check the first
manifest row before training.

## 3. Train

The supplied config already targets Kaggle paths. Change only its dataset root
paths if necessary:

```bash
!python train.py --config configs/ffpp_c23.yaml
```

Output appears under `/kaggle/working/dff_adapter_runs/ffpp_c23/`:

- `last.pt` — resumable state after each epoch;
- `best.pt` — highest validation **video-level** AUC;
- `history.jsonl` — one compact metrics line per epoch.

Resume an interrupted Kaggle session:

```bash
!python train.py --config configs/ffpp_c23.yaml \
  --resume /kaggle/working/dff_adapter_runs/ffpp_c23/last.pt
```

## 4. Evaluate

```bash
!python evaluate.py --config configs/ffpp_c23.yaml \
  --checkpoint /kaggle/working/dff_adapter_runs/ffpp_c23/best.pt
```

For a cross-dataset DeepfakeBench evaluation, build a manifest with the same
four fields for CDF-v1/v2, DFDCP, or DFDC and pass it explicitly:

```bash
!python evaluate.py --config configs/ffpp_c23.yaml \
  --checkpoint /kaggle/working/dff_adapter_runs/ffpp_c23/best.pt \
  --manifest /kaggle/working/manifests/cdfv2_test.jsonl
```

## Notes

- First launch downloads the 1.2 GB+ backbone. After a successful run, save
  `/root/.cache/huggingface` as a Kaggle dataset or enable Internet again.
- The stated paper uses batch 24, 50 epochs, Adam at `2e-4`, weight decay
  `1e-5`, rank 16, α 32, six experts, four heads and loss weights 10:2. Those
  settings are in `configs/ffpp_c23.yaml`.
- This reproduces the described architecture, not the authors' unavailable
  official implementation. Exact paper scores depend on the same face crops,
  official DeepfakeBench splits, GPU/software versions, and preprocessing.
