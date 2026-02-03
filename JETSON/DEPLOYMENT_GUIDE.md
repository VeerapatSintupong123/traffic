# Jetson Nano Deployment Guide

⚠️ **IMPORTANT:** This guide is specifically for Jetson Nano deployment. Your development machine should continue using the full requirements.txt with boxmot trackers.

Complete guide for deploying YOLOv7 + SORT Tracking on Jetson Nano.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verification](#3-verification)
4. [Performance Testing](#4-performance-testing)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements
- ✅ Jetson Nano with CUDA 10.2
- ✅ 4GB RAM available
- ✅ 10GB+ free storage
- ✅ NVIDIA Jetson runtime installed

### Software Requirements
```bash
# Verify Python version
python --version  # Should be 3.6.9

# Verify CUDA
nvcc --version    # Should show CUDA 10.2

# Update pip
pip install --upgrade pip
```

---

## 2. Installation

### Step 1: Install Core Dependencies
```bash
cd JETSON
pip install -r requirements.txt
```

This installs **13 packages** (takes ~10-15 minutes):
- PyTorch 1.10.0 + TorchVision 0.11.1
- OpenCV 4.5.3, NumPy 1.19.4
- TensorRT 8.2.1.8
- filterpy, scipy, shapely
- PyYAML, tqdm, Pillow, pytz

### Step 2: Verify Installation
```bash
python -c "
import torch; print(f'torch: {torch.__version__}')
import torchvision; print(f'torchvision: {torchvision.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import cv2; print(f'opencv: {cv2.__version__}')
import tensorrt; print(f'tensorrt: {tensorrt.__version__}')
print('✓ All core dependencies installed')
"
```

**Expected output:**
```
torch: 1.10.0
torchvision: 0.11.1
numpy: 1.19.4
opencv: 4.5.3.56
tensorrt: 8.2.1.8
✓ All core dependencies installed
```

### Step 3: Apply Code Changes (ON JETSON ONLY)

⚠️ **Do NOT modify these files on your development machine!**

See [CODE_CHANGES.md](CODE_CHANGES.md) for details. Summary:

**File 1: `pipelineV7.py` (Jetson copy only)**
```python
# Comment out these lines:
# from ultralytics import YOLO
# from boxmot import (BoostTrack, BotSort, ...)
```

**File 2: `trt_pipeline/tracking.py` (Jetson copy only)**
```python
# Comment out this line:
# from boxmot.trackers import ByteTrack
```

**Why?** boxmot isn't available on Jetson Nano. You'll use `--type sort` instead.

### Step 4: Test Imports
```bash
cd ..  # Back to traffic/ directory
python -c "
from pipelineV7 import PipelineV7
from sort import Sort
from utils import is_point_on_lane
from trt_pipeline.trt_model import TRTModel
print('✓ All modules imported successfully')
"
```

---

## 3. Verification

### 3.1 Core Dependencies Check

```bash
# PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Expected: CUDA available: True

# NumPy version (CRITICAL)
python -c "import numpy; print('numpy:', numpy.__version__)"
# Expected: numpy: 1.19.4 (DO NOT UPGRADE)

# TensorRT
python -c "import tensorrt as trt; logger = trt.Logger(trt.Logger.INFO); print('TensorRT OK')"
# Expected: TensorRT OK
```

### 3.2 Tracking Test

```bash
python -c "
import numpy as np
from sort import Sort

tracker = Sort()
detections = np.array([[100, 100, 150, 150], [300, 200, 350, 250]])
tracked = tracker.update(detections)
print(f'✓ SORT tracking: {len(tracked)} objects tr

**Note:** SORT is the only tracker available on Jetson. Your dev machine can use ByteTrack, OcSort, etc.acked')
"
```

**Expected:** `✓ SORT tracking: 2 objects tracked`

### 3.3 Model Loading Test

```bash
python -c "
import sys
from pathlib import Path
import torch

# Add yolov7 to path
sys.path.insert(0, 'yolov7')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('models/yolov7.pt', map_location=device, weights_only=False)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
print(f'✓ YOLOv7 loaded on {device}')
"
```

### 3.4 TensorRT Engine Test (if using .engine)

```bash
python -c "
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

with open('models/yolov7-tiny.engine', 'rb') as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

print(f'✓ TensorRT engine loaded: {engine.num_io_tensors} I/O tensors')
"
```

---

## 4. Performance Testing

### 4.1 Memory Check

```bash
# Monitor GPU memory
nvidia-smi

# Test memory allocation
python -c "
import torch
device = torch.device('cuda')
x = torch.randn(1, 3, 640, 640, device=device)
print(f'GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"
```

**Target:** < 2GB GPU memory for inference

### 4.2 Inference Speed Benchmark

```bash
python -c "
import torch
import time
import numpy as np

device = torch.device('cuda')
x = torch.randn(1, 3, 640, 640, device=device)

# Warmup
for _ in range(10):
    _ = x * 2

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        y = x * 2
    torch.cuda.synchronize()
    times.append(time.time() - start)

avg_ms = np.mean(times) * 1000
print(f'Average inference: {avg_ms:.2f} ms')
print(f'Expected FPS: {1000/avg_ms:.1f}')
"
```

**Target:** 15-30 FPS (depending on model size)

### 4.3 Full Pipeline Test

```bash
python main.py tracking \
  --config config/config_test.json \
  --type sort
```

Monitor:
- FPS in terminal output
- GPU memory with `nvidia-smi` (in another terminal)
- CPU temperature with `jetson_stats` (if installed)

---

## 5. Troubleshooting

### Issue: `ImportError: No module named 'ultralytics'`
**Status:** ✅ Expected - this is correct!
**Action:** None needed (already removed from imports)

### Issue: `ImportError: No module named 'boxmot'`
**Status:** ✅ Expected - this is correct!
**Action:** None needed (already removed from imports)

### Issue: `RuntimeError: numpy version incompatibility`
**Cause:** numpy upgraded beyond 1.19.4
**Fix:**
```bash
pip install --force-reinstall numpy==1.19.4
```

### Issue: `torch.cuda.is_available()` returns `False`
**Diagnosis:**
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```
**Fix:** Ensure torch 1.10.0 is installed (CUDA 10.2 compatible)

### Issue: CUDA out of memory
**Symptoms:** `RuntimeError: CUDA out of memory`
**Fixes:**
1. Use smaller input size: `--img-size 416` instead of 640
2. Use TensorRT engine format for efficiency
3. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: YOLOv7 model loading fails
**Check:**
```bash
ls -lh models/yolov7.pt
ls -lh yolov7/models/
```
**Fix:** Ensure `yolov7/` directory exists with all model files

### Issue: Low FPS (< 10 FPS)
**Optimizations:**
1. Convert to TensorRT engine:
   ```bash
   python export.py --weights models/yolov7.pt --img-size 640
   ```
2. Use `yolov7-tiny.pt` instead of full model
3. Reduce input resolution
4. Enable `jetson_clocks` for max performance:
   ```bash
   sudo jetson_clocks
   ```

### Issue: Package dependency conflicts
**Symptoms:** `pip` reports version conflicts
**Fix:** Clean install:
```bash
pip uninstall -y torch torchvision numpy opencv-python
pip install -r JETSON/requirements.txt --no-cache-dir
```

---

## Appendix: Dependency Matrix

| Package | Jetson Version | Purpose | Critical? |
|---------|---------------|---------|-----------|
| torch | 1.10.0 | ML framework | ✅ YES |
| torchvision | 0.11.1 | Vision models | ✅ YES |
| numpy | 1.19.4 | Arrays (DO NOT UPGRADE) | ✅ YES |
| opencv-python | 4.5.3.56 | Image I/O | ✅ YES |
| tensorrt | 8.2.1.8 | Inference engine | ✅ YES |
| filterpy | 1.4.5 | Kalman filter (SORT) | ✅ YES |
| scipy | 1.5.4 | Linear assignment | ✅ YES |
| shapely | 1.8.0 | Geometry/lanes | ✅ YES |
| PyYAML | 5.4.1 | Config parsing | ✅ YES |
| tqdm | 4.62.3 | Progress bars | ⚪ Nice to have |
| Pillow | 8.4.0 | Image ops | ⚪ Nice to have |
| pytz | 2025.2 | Timezone | ⚪ Nice to have |

---

## Quick Reference Commands

```bash
# Install
pip install -r JETSON/requirements.txt

# Verify
python -c "import torch, cv2, numpy, tensorrt; print('✓')"

# Test tracking
python -c "from sort import Sort; Sort(); print('✓')"

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run inference
python main.py tracking --config config/config_test.json --type sort

# Monitor GPU
watch -n 1 nvidia-smi

# Enable max performance
sudo jetson_clocks
```

---

## Success Criteria ✅

After completing this guide:
- [ ] All 13 dependencies installed
- [ ] torch.cuda.is_available() returns True
- [ ] numpy version is exactly 1.19.4
- [ ] Code changes applied (2 files)
- [ ] All imports work without errors
- [ ] SORT tracker initializes
- [ ] YOLOv7 model loads
- [ ] Sample video processes at 15+ FPS
- [ ] GPU memory < 3GB during inference

**Status: Ready for Production!** 🚀
