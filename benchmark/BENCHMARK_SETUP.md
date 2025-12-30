# Traffic Tracker Benchmark Setup Guide

This guide walks you through setting up and running the multi-object tracking (MOT) benchmark system for evaluating different tracking algorithms.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

This benchmark system evaluates various multi-object tracking algorithms (ByteTrack, BoostTrack, BotSort, StrongSort, OcSort, DeepOcSort, HybridSort) using:
- YOLOv7-tiny TensorRT engine for object detection
- OSNet for re-identification
- MOTMetrics for evaluation

**Important**: Due to library dependency conflicts, **two separate conda environments** are required:
1. `traffic` - for running inference
2. `mot-eval` - for running evaluation

---

## Prerequisites

### 1. Ground Truth Annotations

Ground truth data is required for evaluation. You can create annotations using the [video-mot-annotator tool](https://github.com/Razan-S/video-mot-annotator).

The ground truth file should be placed at:
```
benchmark/data/ground-truth/test_30s_output.txt
```

Format: MOT16 format (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)

### 2. Video Frames

Extract frames from your test video and place them in:
```
video/frames/
```

### 3. Model Files

Ensure the following models are available in the `models/` directory:
- `yolov7-tiny.engine` - TensorRT engine for detection
- `osnet_x0_25_msmt17.pt` - ReID model for tracking

---

## Environment Setup

### Environment 1: `traffic` (Inference)

Create and activate the inference environment:

```bash
conda create -n traffic python=3.11
conda activate traffic
```

Install required packages:

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install tensorrt-cu12==10.14.1.48.post1
pip install opencv-python==4.12.0.88
pip install boxmot==16.0.6
pip install numpy==1.26.4
pip install pandas==2.3.3
pip install tqdm==4.67.1
pip install jupyter ipykernel
```

Register the kernel for Jupyter:
```bash
python -m ipykernel install --user --name=traffic --display-name="Python (traffic)"
```

<details>
<summary>Click to see full package list for traffic environment</summary>

```
asttokens==3.0.1
beautifulsoup4==4.14.3
boxmot==16.0.6
certifi==2025.11.12
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
contourpy==1.3.3
cuda-toolkit==12.9.1
cycler==0.12.1
filelock==3.20.0
filterpy==1.4.5
fonttools==4.61.1
fsspec==2025.12.0
ftfy==6.3.1
gdown==5.2.0
idna==3.11
ipykernel==7.1.0
ipython==9.8.0
jinja2==3.1.6
joblib==1.5.3
jupyter_client==8.7.0
jupyter_core==5.9.1
kiwisolver==1.4.9
lapx==0.9.3
loguru==0.7.3
markupsafe==2.1.5
matplotlib==3.10.8
motmetrics==1.4.0
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
nvidia-cuda-runtime==13.1.80
nvidia-cuda-runtime-cu12==12.9.79
opencv-python==4.12.0.88
pandas==2.3.3
pillow==12.0.0
pyparsing==3.3.1
pysocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.2
pyyaml==6.0.3
regex==2024.11.6
requests==2.32.5
scikit-learn==1.8.0
scipy==1.16.3
soupsieve==2.8.1
sympy==1.14.0
tensorrt-cu12==10.14.1.48.post1
tensorrt-cu12-bindings==10.14.1.48.post1
tensorrt-cu12-libs==10.14.1.48.post1
threadpoolctl==3.6.0
torch==2.9.1+cu126
torchvision==0.24.1+cu126
tqdm==4.67.1
urllib3==2.6.2
win32-setctime==1.2.0
xmltodict==1.0.2
yacs==0.1.8
```
</details>

### Environment 2: `mot-eval` (Evaluation)

Create and activate the evaluation environment:

```bash
conda create -n mot-eval python=3.10
conda activate mot-eval
```

Install required packages:

```bash
pip install motmetrics==1.4.0
pip install numpy==1.26.4
pip install pandas==2.1.4
pip install scipy==1.10.1
pip install tqdm==4.67.1
pip install xmltodict==1.0.2
pip install jupyter ipykernel
```

Register the kernel for Jupyter:
```bash
python -m ipykernel install --user --name=mot-eval --display-name="Python (mot-eval)"
```

<details>
<summary>Click to see full package list for mot-eval environment</summary>

```
asttokens==3.0.1
colorama==0.4.6
decorator==5.2.1
executing==2.2.1
ipykernel==7.1.0
ipython==8.37.0
jedi==0.19.2
jupyter_client==8.7.0
jupyter_core==5.9.1
matplotlib-inline==0.2.1
motmetrics==1.4.0
numpy==1.26.4
pandas==2.1.4
parso==0.8.5
prompt-toolkit==3.0.52
pygments==2.19.2
python-dateutil==2.9.0.post0
pytz==2025.2
pyzmq==27.1.0
scipy==1.10.1
tqdm==4.67.1
traitlets==5.14.3
xmltodict==1.0.2
```
</details>

---

## Directory Structure

```
benchmark/
├── BENCHMARK_SETUP.md          # This file
├── inference.ipynb             # Run inference with tracking algorithms
├── evaluate.ipynb              # Evaluate tracking results
├── visualize.ipynb             # Visualize results
└── data/
    ├── config/                 # Stores run configurations (JSON)
    │   └── <run_timestamp>/    # One folder per benchmark run
    │       ├── ByteTrack_round1.json
    │       ├── ByteTrack_round2.json
    │       └── ...
    ├── predicted/              # Tracking results (CSV & TXT)
    │   └── <run_timestamp>/    # One folder per benchmark run
    │       ├── ByteTrack_round1.csv
    │       ├── ByteTrack_round1.txt
    │       └── ...
    ├── evaluate/               # Evaluation results
    │   └── evaluation_results_<timestamp>.csv
    └── ground-truth/           # Ground truth annotations
        └── test_30s_output.txt

models/
├── yolov7-tiny.engine          # TensorRT detection model
└── osnet_x0_25_msmt17.pt       # ReID model

video/
└── frames/                     # Extracted video frames
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

---

## Usage

### Step 1: Run Inference

1. Open `inference.ipynb` in Jupyter
2. **Select the `Python (traffic)` kernel**
3. Update paths if needed:
   - `MODEL_PATH` - Path to TensorRT engine
   - `REID_PATH` - Path to ReID model
   - `FRAMES_DIR` - Path to video frames
4. Configure number of rounds: `NUM_ROUNDS = 10`
5. Run all cells

The notebook will:
- Load the TensorRT model and warm it up
- Run each tracking algorithm for the specified number of rounds
- Save results to `data/predicted/<timestamp>/` (CSV and TXT formats)
- Save configuration metadata to `data/config/<timestamp>/` (JSON)

**Output:**
- Tracking results: `data/predicted/<timestamp>/*.csv` and `*.txt`
- Configuration files: `data/config/<timestamp>/*.json`

### Step 2: Run Evaluation

1. Open `evaluate.ipynb` in Jupyter
2. **Select the `Python (mot-eval)` kernel**
3. Run the cells
4. When prompted, enter the folder name from Step 1 (e.g., `20251229-234204`)

The notebook will:
- Load predicted results and ground truth
- Calculate MOT metrics for each run
- Display results in a DataFrame
- Optionally save results to CSV

**Metrics computed:**
- IDF1, MOTA, MOTP
- Precision, Recall
- FP (False Positives), FN (False Negatives)
- ID Switches, Fragmentations
- Inference time, Tracking time

**Output:**
- Evaluation CSV: `data/evaluate/evaluation_results_<timestamp>.csv`

---

## Troubleshooting

### Issue: TensorRT engine not found
**Solution:** Build the engine from ONNX:
```bash
trtexec --onnx=models/yolov7-tiny.onnx --saveEngine=models/yolov7-tiny.engine --fp16
```

### Issue: CUDA out of memory
**Solution:** 
- Reduce batch size (currently set to 1)
- Close other GPU applications
- Use smaller model or input resolution

### Issue: Wrong kernel selected
**Solution:**
- For `inference.ipynb`: Select `Python (traffic)` kernel
- For `evaluate.ipynb`: Select `Python (mot-eval)` kernel
- In Jupyter: Kernel → Change Kernel → Select appropriate kernel

### Issue: Import errors in inference
**Solution:** Make sure you're using the `traffic` environment:
```bash
conda activate traffic
jupyter notebook
```

### Issue: Import errors in evaluation
**Solution:** Make sure you're using the `mot-eval` environment:
```bash
conda activate mot-eval
jupyter notebook
```

### Issue: Ground truth file not found
**Solution:** 
- Ensure `data/ground-truth/test_30s_output.txt` exists
- Annotate your video using [video-mot-annotator](https://github.com/Razan-S/video-mot-annotator)

### Issue: Frame files not found
**Solution:**
- Extract frames from your video first
- Place frames in `video/frames/` directory
- Ensure frames are named sequentially (e.g., frame_0001.jpg, frame_0002.jpg)

---

## Notes

- **Performance:** TensorRT inference is GPU-accelerated. Ensure CUDA is properly installed.
- **Reproducibility:** Each algorithm runs multiple rounds (default: 10) to measure consistency.
- **ID Mapping:** Track IDs are remapped to sequential integers starting from 1.
- **Coordinate System:** Boxes are scaled from letterboxed 640x640 space back to original frame coordinates.
- **File Formats:**
  - CSV: Includes headers, easier to read
  - TXT: MOT16 format for evaluation tools
  - JSON: Metadata (inference times, tracking times, configuration)

---

## Citation

If you use this benchmark system, please cite the respective tracking algorithms and tools:
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [MOTMetrics](https://github.com/cheind/py-motmetrics)

---

## License

Follow the licenses of the respective dependencies and models used in this benchmark.
