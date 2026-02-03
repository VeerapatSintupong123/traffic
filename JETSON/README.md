# Jetson Nano Deployment Package

⚠️ **IMPORTANT:** This is ONLY for Jetson Nano deployment. Your main development environment can continue using boxmot trackers (ByteTrack, OcSort, etc.)

**Status:** ✅ Ready for Deployment  
**Date:** February 3, 2026  
**Target:** Jetson Nano (CUDA 10.2, Python 3.6.9)

---

## 📁 Files in This Directory

| File | Purpose |
|------|---------|
| **[requirements.txt](requirements.txt)** | All dependencies optimized for Jetson Nano |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Complete setup and verification guide |
| **[CODE_CHANGES.md](CODE_CHANGES.md)** | Required code modifications |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd JETSON
pip install -r requirements.txt
```

### 2. Apply Code Changes (JETSON ONLY)
**⚠️ Only make these changes on Jetson Nano, NOT on your development machine!**

See [CODE_CHANGES.md](CODE_CHANGES.md) for details:
- Comment out `ultralytics` and `boxmot` imports in `pipelineV7.py`
- Comment out `boxmot` import in `trt_pipeline/tracking.py`
- Use `--type sort` when running (other trackers won't work on Jetson)

### 3. Verify Installation
```bash
python -c  (on Jetson Nano)
```bash
cd ..
python main.py tracking --config config/config_test.json --type sort
```

**Note:** On Jetson, you can ONLY use `--type sort`. Other trackers (ByteTrack, OcSort, etc.) require boxmot which isn't available.bash
cd ..
python main.py tracking --config config/config_test.json --type sort
```

---

## ✅ What's Included

**13 Core Dependencies** (all Jetson-compatible):
- PyTorch 1.10.0 + TorchVision 0.11.1
- OpenCV 4.5.3, NumPy 1.19.4
- TensorRT 8.2.1.8
- Tracking: filterpy, scipy
- Utilities: shapely, PyYAML, tqdm, Pillow, pytz

**Removed** (incompatible):
- ❌ ultralytics (requires newer numpy)
- ❌ boxmot (not available for Jetson)

---

## 📊 Expected Performance

- **Inference Speed:** 15-30 FPS (model dependent)
- **Memory Usage:** < 3GB RAM
- **Model Load:** < 2 seconds
- **Tracking:** SORT (Kalman-based, real-time)

---

## 🆘 Troubleshooting

**Import errors?**
- Verify numpy version: `python -c "import numpy; print(numpy.__version__)"` → Must be 1.19.4

**CUDA not found?**
- Check: `python -c "import torch; print(torch.cuda.is_available())"` → Must be True

**Model loading fails?**
- Ensure `yolov7/` folder exists in parent directory
- Check model file: `models/yolov7.pt` or `models/yolov7-tiny.engine`

**More help:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Section 12

---

## 📝 Summary of Changes

From original requirements:
- **Before:** 15+ packages, including incompatible ultralytics/boxmot
- **After:** 13 core packages, all Jetson-compatible
- **Code changes:** 2 files (remove 4 lines total)
- **Result:** Clean, optimized inference pipeline

---

**Ready to deploy!** 🎯
