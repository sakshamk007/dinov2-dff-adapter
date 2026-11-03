# Data Preparation for DFF-Adapter (FaceForensics++ and Cross-Dataset Benchmarks)

This folder documents and contains the scripts we used to turn raw FaceForensics++
(FF++) videos, and the unseen cross-dataset benchmarks, into the aligned face crops
and metadata that the model in `model.py` reads. It follows the dataset preparation
and evaluation protocol of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench),
which is the protocol used by the base paper
(*Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection*,
arXiv:2511.12107).


## 1. Pipeline at a glance

```text
FF++ c23 videos
   -> (optional) frame extraction                  extract_compressed_videos.py
   -> face detection, alignment, 256x256 crops     preprocess.py   (DeepfakeBench, dlib)
   -> metadata JSON per dataset                    rearrange.py    (DeepfakeBench)
   -> private Kaggle dataset                       shared by all three team members
   -> frame loader on the official splits          FFppDataset / DFBenchDataset in ../model.py
   -> model -> frame scores -> video-level AUC     evaluate() in ../model.py
```

## 2. Files in this folder

| File | What it does | 
|---|---|
| `faceforensics_download_v4.py` | Downloads FF++ videos after access is approved |
| `extract_compressed_videos.py` | Extracts every frame of every video to PNG |
| `preprocess.py` | Detects, aligns and crops faces; saves frames, landmarks and masks | 
| `rearrange.py` | Builds the JSON metadata (labels, frame lists, splits) | DeepfakeBench |
| `../scripts/download_gdrive.py` | Downloads a large file from Google Drive with `gdown`; resumable | 

Our work was obtaining dataset
access, running and configuring them, checking the outputs, making the folder layout
and paths consistent with the loader in `model.py`, and publishing the result as a
private Kaggle dataset.

## 3. FaceForensics++ dataset

FF++ has 1,000 real YouTube videos and 4,000 forged videos made with four methods:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

FaceShifter and DeepFakeDetection are separate components that we do not use for
training.

### Official dataset access

FF++ is distributed through an access-request process. We filled in the access form in
the official repository and received the download script after approval.

* [Official FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
* [FaceForensics++ dataset access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)

We use the **c23** videos only (H.264 with constant rate factor 23, a medium-quality
setting similar to online video), because the base paper trains on FF++ c23 only. The
other levels are raw (c0) and c40 (heavy compression).

### Official splits: why we split by video pair

FF++ ships official `train.json`, `val.json` and `test.json` files. Each lists **pairs**
of video IDs `[a, b]`. A pair stands for:

* the real videos `a` and `b`, and
* for each method, the fakes `a_b` and `b_a` (one person's face placed on the other's
  video).

We split by pair (360 / 70 / 70 pairs for train / val / test), so the same person never
appears in two splits. Splitting by frame, or even by single video, would let the model
recognise people instead of forgeries and inflate the scores. The split files must sit
in the `FaceForensics++/` root folder (next to `original_sequences/`); `rearrange.py`
and `model.py` look for them there.

### Folder layout

Before preprocessing:

```text
FaceForensics++/
├── train.json  val.json  test.json
├── original_sequences/
│   └── youtube/c23/videos/*.mp4
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/
    ├── Face2Face/c23/videos/
    ├── FaceSwap/c23/videos/
    └── NeuralTextures/c23/videos/
```

After preprocessing, `frames/`, `landmarks/` and (for the four fake methods) `masks/`
appear next to each `videos/` folder:

```text
.../c23/
├── videos/
├── frames/<video-id>/<frame-number>.png     <- the only folder our model reads
├── landmarks/<video-id>/<frame-number>.npy
└── masks/<video-id>/<frame-number>.png      (fakes only, not used by our model)
```

## 4. Software requirements

* Ubuntu Linux, Python 3
* OpenCV, NumPy, Pillow, `tqdm`, `scikit-image`, `imutils`, `pandas`, PyYAML
* FFmpeg with H.264 support
* `dlib`, and the 81-point landmark file `shape_predictor_81_face_landmarks.dat` from
  DeepfakeBench (`preprocess.py` expects it at `./dlib_tools/`)

## 5. Setup

Get DeepfakeBench and work inside its `preprocessing` folder.

```bash
cd ~/Desktop/DeepfakeBench-main
ls preprocessing
python3 -m venv venv
source venv/bin/activate
```

Copy the scripts from this folder into `preprocessing/` if you want to use our
versions.

## 6. Download the raw FF++ c23 videos

```bash
python aceforensics_download_v4.py -h
python aceforensics_download_v4.py /path/to/output \
  -d all \
  -c c23 \
  -t videos
```

* `-d` selects the dataset (`all`, `original`, `Deepfakes`, `Face2Face`, `FaceSwap`,
  `NeuralTextures`, ...).
* `-c` selects the compression level (`raw`, `c23`, `c40`).
* `-t` selects the file type (`videos`, `masks`, `models`).
* `--server EU`, `EU2` or `CA` selects the download server.

The exact options depend on the script supplied after access approval; run `-h` first.

### Verify the downloaded videos

```bash
DATASET="$HOME/Desktop/FaceForensics/ffpp_c23"
find "$DATASET" -maxdepth 4 -type d | head -50
find "$DATASET" -type f -iname "*.mp4" | wc -l
find "$DATASET/original_sequences" -type f -iname "*.mp4" | head
find "$DATASET/manipulated_sequences" -type f -iname "*.mp4" | head
```

These checks confirm the folder structure and the file count. They do not prove that
every video downloaded completely or can be decoded.

## 7. Frame extraction (optional, for inspection)

`extract_compressed_videos.py` reads every frame of each video with OpenCV and writes
`0000.png`, `0001.png`, ... into an `images/` folder next to `videos/`:

```bash
python extract_compressed_videos.py \
  --data_path /path/to/FaceForensics++ \
  --dataset original \
  --compression c23
```

Choices for `--dataset` are `original`, `Deepfakes`, `Face2Face`, `FaceSwap` and `all`
(NeuralTextures is not in this script's list). **This `images/` output is not used by
the model.** It is useful for looking at the raw data. The training frames come from the
face-preprocessing step below.

## 8. Face detection, alignment and cropping (`preprocess.py`)

A video frame shows a whole scene, but the detector only needs the face, and every face
should sit in the same position. For each video, `preprocess.py`:

1. **Chooses frames.** Mode `fixed_num_frames` takes evenly spaced frames
   (`np.linspace`); mode `fixed_stride` takes every n-th frame. Both are set in
   `config.yaml`.
2. **Detects faces** with dlib's frontal face detector and keeps only the **largest**
   face in the frame.
3. **Finds landmarks** with the 81-point model and keeps five: both eyes, the nose tip
   and both mouth corners.
4. **Aligns and crops.** A similarity transform (rotation, scale and shift, with no
   distortion) maps the five points onto a standard face template, giving a
   **256x256 crop with a 1.3x margin**.
5. **Re-checks the crop.** It runs the detector again on the crop to get landmarks for
   it. If no face is found, the frame is skipped.
6. **Saves** `frames/<video>/<frame>.png`, `landmarks/.../*.npy` and, for fakes,
   `masks/.../*.png`.

Videos run in parallel (one thread per CPU core), and progress and errors go to
`logs/<dataset>.log`.


Set these keys under `preprocess`:

| Key | Value |
|---|---|
| `dataset_name` | `FaceForensics++` |
| `dataset_root_path` | the folder that contains `FaceForensics++/` |
| `comp` | `c23` |
| `mode` | `fixed_num_frames` or `fixed_stride` |
| `num_frames` / `stride` | the sampling values for your experiment |

### Run

```bash
python preprocess.py
```

**Folder check.** For FF++, `preprocess.py` lists these folders and **exits if any is
missing**: `original_sequences/youtube`, `original_sequences/actors`,
`manipulated_sequences/Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`,
`FaceShifter` and `DeepFakeDetection`. If you downloaded only the five components we
use, delete the other names from the `sub_dataset_names` list in `preprocess.py`
before running. Do the same for the `actors` and FaceShifter handling in
`rearrange.py` (see below).

### Verify

```bash
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ -type d -name frames
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ \
  -type f -path "*/frames/*" -name "*.png" | wc -l
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ \
  -type d \( -name landmarks -o -name masks \)
```

A few videos may end up with no crops when no face is detected in any sampled frame. In
our training split, one real video and one fake video per method (except FaceSwap) were
missing, which is 719 of 720 videos found, so the effect is negligible. The loader
prints these counts.

## 9. Metadata and splits (`rearrange.py`)

`rearrange.py` scans the `frames/` folders and writes JSON metadata of this form:

```text
{ "<dataset>": { "<label>": { "<split>": { "<compression>": { "<video-id>": {
      "label": "<label>", "frames": ["/path/to/frame1.png", ...] } } } } } }
```

For FF++ it uses the official `train.json`, `val.json` and `test.json` to assign every
video (`a`, `b`, `a_b`, `b_a`) to a split, and writes one JSON file per forgery method
(for example `FF-DF.json`), each holding the real videos plus that method. Labels are
`FF-real`, `FF-DF` (Deepfakes), `FF-F2F` (Face2Face), `FF-FS` (FaceSwap) and `FF-NT`
(NeuralTextures). `model.py` uses the same names for the forgery-type labels.


```bash
cd ~/Desktop/DeepfakeBench-main/preprocessing
python rearrange.py
find dataset_json -maxdepth 2 -type f -name "*.json"
```

`rearrange.py` also reads `original_sequences/actors` for the FF++ run. If you did not
download it, remove that block (or create an empty `actors/c23/frames` folder) before
running.

Validate the JSON syntax:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("dataset_json")
files = list(root.glob("*.json"))
if not files:
    raise SystemExit("No JSON files found. Check the configured output directory.")
for path in files:
    with path.open("r", encoding="utf-8") as file:
        json.load(file)
    print("Valid JSON:", path)
print("JSON syntax validation completed.")
PY
```

**Note:** our FF++ training and evaluation (`FFppDataset` in `model.py`) reads the
`frames/` folders and the `train/val/test.json` pair lists directly. The JSON files from
`rearrange.py` are needed for the **cross-dataset** evaluation (Section 12).

## 10. Private Kaggle dataset

The prepared data is stored in a **private Kaggle dataset**, so all collaborators train
from an identical copy without repeating the download and cropping. It must stay private
because FF++ is shared under an access agreement.

1. Organise the prepared folders (`frames/` and the split files; masks and landmarks are
   optional).
2. Check the package contents.
3. Upload it as a private Kaggle dataset.
4. Attach it to the notebook.
5. Confirm that the images and split files are visible from the notebook.

## 11. How the training loader reads the data

`FFppDataset` in `../model.py`:

* finds the folder that contains `original_sequences/` and `manipulated_sequences/`
  (searching `/kaggle/input` if the given path is wrong);
* finds `train.json`, `val.json` and `test.json` and reads the video pairs;
* builds the video names (real: `a`, `b`; fakes: `a_b`, `b_a`) for each of the five
  classes;
* picks evenly spaced frames from each video's `frames/` folder.

| Split | Pairs | Frames per video | Frames used |
|---|---|---|---|
| Train | 360 | 8 | 28,768 |
| Val | 70 | 16 | 11,186 |
| Test | 70 | 32 | 22,388 |

The paper does not state its frame counts; these are our choices. Images are resized to
224x224 and normalised with ImageNet statistics. Training also uses a random horizontal
flip and colour jitter (0.1).

**Check before training** (no GPU needed). It prints the number of videos and frames
found per class and split, and verifies a random sample of 300 frames:

```bash
python ../model.py check --root /path/to/FaceForensics
```

## 12. Cross-dataset evaluation

The detector is trained on FF++ c23 only and then tested, without retraining, on:

* **Celeb-DF v1 and v2**: real celebrity videos and face-swap fakes.
* **DFDCP**: the preview set of the Deepfake Detection Challenge.
* **DFDC**: the full Deepfake Detection Challenge test set.

These tests measure generalisation to faces, lighting, compression and forgery methods
the model has never seen. Each dataset requires its own access and download.

### Raw layouts expected by `preprocess.py`

```text
Celeb-DF-v1/ and Celeb-DF-v2/   Celeb-real/  Celeb-synthesis/  YouTube-real/  List_of_testing_videos.txt
DFDCP/                          original_videos/  method_A/  method_B/  dataset.json
DFDC/                           test/ (with labels.csv)  train/dfdc_train_part_0 ... 49
```

### Steps for each dataset

1. Run `python preprocess.py` to create the face crops.
2. Run `python rearrange.py` to create `dataset_json/<dataset>.json`. The test split comes
   from `List_of_testing_videos.txt` (Celeb-DF), `dataset.json` (DFDCP) or `labels.csv`
   (DFDC).
3. Evaluate the trained checkpoint:

```bash
python ../model.py eval --data json \
  --json /path/to/dataset_json/Celeb-DF-v2.json \
  --root /path/to/datasets/Celeb-DF-v2 \
  --anchor Celeb-DF-v2 \
  --ckpt ../checkpoints/Epochs50/best.pt \
  --split test --frames_eval 32
```

How this works in `DFBenchDataset`:

* A video is **real** if its label contains "real" (for example `CelebDFv2_real`,
  `DFDC_Real`), and fake otherwise.
* The JSON stores absolute paths from the machine that created it. The loader keeps the
  part of each path after `--anchor` and joins it to `--root`, so the data can be moved
  to Kaggle or another machine. The default anchor is `FaceForensics++`, so for other
  datasets set `--anchor` to a folder name that appears in the stored paths (for
  example `Celeb-DF-v2`).

### Metric

**Video-level AUC.** The model scores each frame (probability of fake), the scores are
averaged per video, and ROC AUC is computed over videos, as in the paper and
DeepfakeBench. Frame-level AUC is also printed. AUC measures how well fakes are ranked
above reals (0.5 is random, 1.0 is perfect) and does not depend on a threshold.

### Our results (ViT-L, trained on FF++ c23 for 50 epochs)

| Test set | Paper | Ours |
|---|---|---|
| FF++ test | 99.56 | 99.62 |
| Celeb-DF v2 | 95.26 | 90.11 |
| DFDC | 89.96 | 83.09 |
| Celeb-DF v1 | 96.14 | 88.20 |
| DFDCP | 91.57 | 90.59 |
| Cross-dataset average | 93.23 | 88.00 |

On twelve DF40 manipulation methods, we average 94.16 against the paper's 95.20. Our
report discusses possible reasons for the cross-dataset gap (our design choices,
checkpoint selection on FF++ validation AUC, and preprocessing or frame-sampling
differences); these are hypotheses that still need ablations.

## 13. References

1. [FaceForensics++ official repository](https://github.com/ondyari/FaceForensics)
2. [FaceForensics++ dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
3. [FaceForensics frame extraction script](https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py)
4. [DeepfakeBench repository](https://github.com/SCLBD/DeepfakeBench)
5. [DeepfakeBench preprocessing script](https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/preprocess.py)
6. [DeepfakeBench preprocessing configuration](https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/config.yaml)
7. T. Zhang et al., *Fine-Grained DINO Tuning with Dual Supervision for Face Forgery
   Detection*, arXiv:2511.12107, 2025.

## 14. Dataset integrity and responsible use

This repository documents the dataset preparation workflow and the code used to
organise the data for research.

FaceForensics++, Celeb-DF, DFDCP, DFDC and other datasets remain subject to their
providers' access terms and usage restrictions. Dataset files are not included in this
public repository, and our prepared copy is kept in a private Kaggle dataset.

For reproducibility, we record the dataset version, compression setting, preprocessing
configuration, frame-sampling method, metadata format, model input size and evaluation
protocol used in each experiment.
