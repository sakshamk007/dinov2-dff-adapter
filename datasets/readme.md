# Deepfake Detection Using Fine-Grained DINOv2 Tuning


## 1. Project Overview

This project focuses on deepfake detection using a vision-transformer-based approach inspired by the research paper *Fine-Grained DINO Tuning*. The implementation follows the dataset preparation and evaluation practices of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

The primary dataset is **FaceForensics++ (FF++) with C23 compression**. Raw videos are obtained through the official FaceForensics dataset access process and prepared using the dataset's extraction utilities and DeepfakeBench's preprocessing pipeline.



## 2. FaceForensics++ Dataset

FaceForensics++ is a benchmark dataset for facial manipulation detection. It includes original videos and manipulated videos generated using different methods.

The original dataset contains manipulation methods including:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

Additional dataset components, such as FaceShifter and DeepFakeDetection, are available through the appropriate dataset access arrangements.


### Official dataset access

FaceForensics++ is distributed through an access-request process. We have filled the acess form present in the official repository and obtained the download script after approval.

* [Official FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
* [FaceForensics++ dataset access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)


The project uses the **C23 compressed videos** for its initial dataset preparation. C23 refers to the H.264 compression setting with a constant rate factor of 23.


### Dataset organization

The raw video dataset is organized using the directory conventions expected by DeepfakeBench.

```text
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    ├── Face2Face/
    │   └── c23/
    │       └── videos/
    ├── FaceSwap/
    │   └── c23/
    │       └── videos/
    ├── NeuralTextures/
    │   └── c23/
    │       └── videos/
    └── FaceShifter/
        └── c23/
            └── videos/
```

The exact folders depend on which dataset components were downloaded. The directory structure should be checked before preprocessing.



## 3. Software Requirements

The workflow uses:

* Ubuntu Linux
* Python 3
* OpenCV
* NumPy
* Pillow
* `tqdm`
* FFmpeg with H.264 support
* Dlib and the facial-landmark predictor required by DeepfakeBench
* PyYAML and other dependencies specified by the selected DeepfakeBench version

The original FaceForensics utilities document the use of OpenCV, FFmpeg, and `tqdm` for frame extraction. DeepfakeBench provides additional face preprocessing and dataset arrangement utilities.



## 4. Repository Setup


```bash
cd ~/Desktop/DeepfakeBench-main
```

Inspect the repository:

```bash
ls
ls preprocessing
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```



## 5. Download Raw FF++ C23 Videos

```bash
python download-FaceForensics.py -h
```

The official FaceForensics documentation provides the following general command for downloading original and manipulated videos at C23 compression:

```bash
python download-FaceForensics.py /path/to/output \
  -d all \
  -c c23 \
  -t videos
```

If supported by the supplied script, a server option can also be provided:

```bash
python download-FaceForensics.py /path/to/output \
  -d all \
  -c c23 \
  -t videos \
  --server EU2
```

Replace `/path/to/output` with the intended dataset location. The exact script filename and available options should be checked against the script supplied after access approval.

The download selection should be restricted to the dataset components needed for the experiment. Check the output to ensure the required original and manipulated C23 videos are present.

### Verify the downloaded videos

Set the dataset directory:

```bash
DATASET="$HOME/Desktop/FaceForensics/ffpp_c23"
```

Inspect its contents:

```bash
find "$DATASET" -maxdepth 4 -type d | head -50
```

Count the downloaded MP4 files:

```bash
find "$DATASET" -type f -iname "*.mp4" | wc -l
```

Inspect a sample of original videos:

```bash
find "$DATASET/original_sequences" \
  -type f -iname "*.mp4" | head
```

Inspect a sample of manipulated videos:

```bash
find "$DATASET/manipulated_sequences" \
  -type f -iname "*.mp4" | head
```

These checks verify the directory structure and provide a file count. They do not by themselves establish that every video downloaded successfully or can be decoded.




## 6. Frame Extraction

Frame extraction converts video files into individual image files. The original FaceForensics repository supplies an extraction utility for this purpose.

Official reference: [FaceForensics frame extraction utility](https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py).

### 6.1 Inspect the extraction utility

Obtain the official utility from the FaceForensics repository, then inspect its options:

```bash
python extract_compressed_videos.py -h
```

The official script uses a dataset root, dataset selection, and compression setting. For example, the command pattern is:

```bash
python extract_compressed_videos.py \
  --data_path /path/to/FaceForensics++ \
  --dataset original \
  --compression c23
```

The original extraction utility supports the dataset categories and compression settings defined in its own source code. Check the available choices before running it on additional manipulation categories.

### 6.2 Extract frames from the selected videos

For categories supported by the utility, the extracted images are written into image directories associated with the input video folders.

The output structure is generally similar to:

```text
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           ├── videos/
│           └── images/
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       ├── videos/
    │       └── images/
    └── Face2Face/
        └── c23/
            ├── videos/
            └── images/
```

Each video's extracted frames are stored in a corresponding image subdirectory.

The extraction utility is useful for producing frame images and inspecting the raw video data. The subsequent DeepfakeBench preprocessing stage performs the face-oriented processing needed by the detector.



## 7. DeepfakeBench Face Preprocessing

DeepfakeBench provides a preprocessing pipeline for face detection, alignment, cropping, and the generation of associated data such as facial landmarks and masks where supported.

Official references:

* [DeepfakeBench preprocessing script](https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/preprocess.py)
* [DeepfakeBench repository](https://github.com/SCLBD/DeepfakeBench)

### 7.1 Configure the preprocessing pipeline

Move into the preprocessing directory:

```bash
cd ~/Desktop/DeepfakeBench-main/preprocessing
```

Open the configuration:

```bash
nano config.yaml
```

Configure the dataset name, dataset root, compression level, and frame-sampling options according to the actual keys in the local configuration file.

The essential settings should correspond to:

* Dataset: FaceForensics++
* Compression: C23
* Dataset root: the directory containing the `FaceForensics++` folder
* Sampling mode: the mode required by the experiment
* Number of frames or sampling stride: the values selected for the experiment


### 7.2 Run the official preprocessing script

After confirming the raw video structure, dependencies, predictor files, and configuration, run:

```bash
cd ~/Desktop/DeepfakeBench-main/preprocessing
python preprocess.py
```

The script processes the dataset according to the configured paths and options. The exact outputs depend on the dataset and preprocessing configuration.

The resulting data include:

* Extracted video frames
* Detected and cropped face images
* Facial landmarks
* Manipulation masks where supported

A representative output layout is:

```text
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           ├── videos/
│           ├── frames/
│           └── landmarks/
└── manipulated_sequences/
    └── Deepfakes/
        └── c23/
            ├── videos/
            ├── frames/
            ├── landmarks/
            └── masks/
```



### 7.3 Verify generated outputs

Inspect the generated frame directories:

```bash
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ \
  -type d -name frames
```

Count generated PNG files:

```bash
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ \
  -type f -path "*/frames/*" -name "*.png" | wc -l
```

Inspect landmark and mask directories:

```bash
find ~/Desktop/DeepfakeBench-main/datasets/rgb/FaceForensics++ \
  -type d \( -name landmarks -o -name masks \)
```




## 8. Dataset Metadata and Splits

DeepfakeBench provides a rearrangement script for organizing the processed dataset into the metadata format used by its data loaders.

After preprocessing, inspect the rearrangement configuration in `preprocessing/config.yaml`. Confirm the dataset root, compression setting, and output directory.

Run:

```bash
cd ~/Desktop/DeepfakeBench-main/preprocessing
python rearrange.py
```

The expected output includes JSON metadata describing dataset entries and the splits used by the selected workflow. The exact JSON filenames and fields depend on the configuration and repository version.

Inspect the generated JSON files:

```bash
find dataset_json -maxdepth 2 -type f -name "*.json"
```

Validate JSON syntax:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("dataset_json")
files = list(root.glob("*.json"))

if not files:
    raise SystemExit(
        "No JSON files found. Check the configured output directory."
    )

for path in files:
    with path.open("r", encoding="utf-8") as file:
        json.load(file)
    print("Valid JSON:", path)

print("JSON syntax validation completed.")
PY
```



## 9. Private Kaggle Dataset and Notebook Workflow

The prepared FaceForensics++ dataset is stored in a **private Kaggle dataset** for use in Kaggle notebooks during the later stages of the project.

This allows the authorized project collaborators to access the prepared data from the notebook environment without needing to repeat the full download and preprocessing workflow each time.

The dataset transfer workflow includes:

1. Organizing the prepared data and associated metadata.
2. Verifying the contents of the dataset package.
3. Uploading the prepared dataset to a private Kaggle dataset.
4. Attaching the private dataset to the relevant Kaggle notebook.
5. Confirming that the expected image files and metadata can be accessed from the notebook.
6. Using the data for the subsequent model development and evaluation phases.




## 10. FUTURE WORK (Phase 3): Cross-Dataset Evaluation

The third phase of the project will investigate whether the trained detector generalizes beyond FaceForensics++.

The following datasets are planned for evaluation, subject to obtaining access and preparing the data in the required format.

### Celeb-DF v1

Celeb-DF v1 is a deepfake dataset containing real celebrity videos and synthesized face-swapping videos. It provides an additional benchmark for evaluating cross-dataset generalization.

### Celeb-DF v2

Celeb-DF v2 is an expanded version of Celeb-DF and provides a larger evaluation set for testing the detector on videos produced using a different dataset and manipulation pipeline.

### Deepfake Detection Challenge Preview (DFDCP)

DFDCP is the preview dataset associated with the Deepfake Detection Challenge. It provides additional manipulated and authentic videos for assessing the robustness of a detector outside the main FF++ dataset.

### Deepfake Detection Challenge (DFDC)

DFDC is a larger deepfake detection dataset containing real and manipulated videos. It can be used to investigate performance on a broader set of identities, recording conditions, and manipulation methods.

DeepfakeBench lists Celeb-DF v1, Celeb-DF v2, DFDCP, and DFDC among the supported dataset configurations. Each dataset requires the appropriate access, file organization, metadata, and preprocessing.

Official reference: [DeepfakeBench README](https://github.com/SCLBD/DeepfakeBench).




## 11. References

1. [FaceForensics++ official repository](https://github.com/ondyari/FaceForensics)
2. [FaceForensics++ dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
3. [FaceForensics frame extraction script](https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py)
4. [DeepfakeBench repository](https://github.com/SCLBD/DeepfakeBench)
5. [DeepfakeBench preprocessing script](https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/preprocess.py)
6. [DeepfakeBench preprocessing configuration](https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/config.yaml)
7. *Fine-Grained DINO Tuning* — the research paper used as the methodological reference.




## 12. Dataset Integrity and Responsible Use

This repository documents the dataset preparation workflow and the code used to organize the data for research.

FaceForensics++, Celeb-DF, DFDCP, DFDC, and other datasets remain subject to their respective providers' access terms and usage restrictions. Dataset files are not included in the public source-code repository.

For reproducibility, the project records the dataset version, compression setting, preprocessing configuration, frame-sampling method, metadata format, model input size, and evaluation protocol used in each experiment.
