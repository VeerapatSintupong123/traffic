# Code Changes for Jetson Nano Compatibility

⚠️ **CRITICAL:** Only apply these changes ON THE JETSON NANO, not on your development machine!

**Summary:** Comment out incompatible imports to allow SORT-only tracking on Jetson.

---

## Overview

**Your Current Setup:**
- **Development Machine:** Uses boxmot trackers (ByteTrack, OcSort, BotSort, etc.) ✅ Keep as-is
- **Jetson Nano:** Can only use SORT (local implementation) ⚠️ Apply changes below

The following packages are **NOT compatible** with Jetson Nano:
- ❌ `ultralytics` (requires numpy > 1.19.5)
- ❌ `boxmot` (not available for Jetson platform)

**Solution:** Use local YOLOv7 implementation and SORT tracker (already in codebase).

---

## File 1: `pipelineV7.py`

### Location
`c:\Users\sschw\schwynn\Work\Teacher-Supervised\traffic\pipelineV7.py`

### Change 1: Comment Out Incompatible Imports (Lines ~16-28)

**BEFORE:**
```python
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from clean import process_images
from sort import Sort
from boxmot import (
    BoostTrack, BotSort, HybridSort,
    StrongSort, DeepOcSort, ByteTrack,
    OcSort
)
from utils import (...)
```

**AFTER (Jetson Nano only):**
```python
import torch
import torch.nn as nn
# JETSON: Commented out - not compatible with Jetson Nano
# from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from clean import process_images
from sort import Sort
# JETSON: Commented out - boxmot not available on Jetson Nano
# from boxmot import (
#     BoostTrack, BotSort, HybridSort,
#     StrongSort, DeepOcSort, ByteTrack,
#     OcSort
# )
from utils import (...)
```

### Change 2: Simplify Tracking Algorithm Init (Lines ~46-60)

**BEFORE:**
```python
def _initial_tracking_algorithm(self, algorithm_name: str):
    if algorithm_name.lower() == "sort":
        return Sort()
    elif algorithm_name.lower() == "boosttrack":
        return BoostTrack(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=self.device, half=False)
    elif algorithm_name.lower() == "botsort":
        return BotSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=self.device, half=False)
    # ... more boxmot trackers ...
    else:
        raise ValueError(f"Unsupported tracking algorithm: {algorithm_name}")
```

**AFTER:**
```python
def _initial_tracking_algorithm(self, algorithm_name: str):
    """Initialize tracking algorithm (Jetson Nano compatible)."""
    if algorithm_name.lower() == "sort":
        return Sort()
    else:
        raise ValueError(f"Unsupported tracking algorithm: {algorithm_name}. Use 'sort' for Jetson Nano.")
```

### Change 3: Update Tracker Call (Lines ~235)

**BEFORE:**
```python
# Different interface for SORT vs boxmot trackers
if isinstance(self.tracker, Sort):
    tracker_objects = self.tracker.update(np.array(boxes_only))
else:
    tracker_objects = self.tracker.update(np.array(response) if response else np.empty((0, 6)), img0)
```

**AFTER:**
```python
# SORT tracker interface: numpy array of [x1, y1, x2, y2]
tracker_objects = self.tracker.update(np.array(boxes_only) if boxes_only else np.empty((0, 4)))
```

---

## File 2: `trt_pipeline/tracking.py`

### Location
`c:\Users\sschw\schwynn\Work\Teacher-Supervised\traffic\trt_pipeline\tracking.py`

### Change: Remove boxmot Import (Line ~13)

**BEFORE:**
```python
import torch
import numpy as np
import os
import time
import cv2 as cv
from shapely.geometry import Point
from shapely import contains

from boxmot.trackers import ByteTrack  # ❌ REMOVE

from trt_model import TRTModel
from video_stream import VideoStream, AsyncImageSaver, letterbox
from tools import (...)
```

**AFTER:**
```python
import torch
import numpy as np
import os
import time
import cv2 as cv
from shapely.geometry import Point
from shapely import contains

# REMOVED: from boxmot.trackers import ByteTrack
# Use SORT or custom tracker implementation

from trt_model import TRTModel
from video_stream import VideoStream, AsyncImageSaver, letterbox
from tools import (...)
```

---

## File 3: `sort.py`

### Status
✅ **No changes needed** - Already compatible with Jetson Nano

**Dependencies:**
- numpy (✓)
- filterpy.kalman (✓)
- scipy.optimize (✓)

All available in `JETSON/requirements.txt`

---

## File 4: `utils.py`

### Status
⚪ **Optional changes** - Core functions already compatible

**Current imports:**
```python
import cv2 as cv          # ✓ Compatible
import numpy as np        # ✓ Compatible
import torch              # ✓ Compatible
from shapely.geometry import Point, Polygon  # ✓ Compatible
from PIL import Image     # ✓ Compatible
import pytz               # ✓ Compatible

# Optional (only for advanced features):
from moviepy.video.io.VideoFileClip import VideoFileClip  # ⚠️ Not in requirements
from sklearn import decomposition  # ⚠️ Not in requirements
```

**Action:** Keep as-is. Functions using moviepy/sklearn are:
- `extract_background()` - Only needed during setup
- `anomaly_detect()` - Optional advanced feature

These will fail gracefully if called without dependencies.

---

## Summary

| File | Lines Changed | Complexity | Time |
|------|--------------|------------|------|
| pipelineV7.py | 3 changes | Medium | 5 min |
| trt_pipeline/tracking.py | 1 change | Easy | 1 min |
| sort.py | 0 changes | - | - |
| utils.py | 0 changes | - | - |

**Total effort:** ~10 minutes

---

## Testing After Changes

### 1. Syntax Check
```bash
python -m py_compile pipelineV7.py
python -m py_compile trt_pipeline/tracking.py
```

### 2. Import Test
```bash
python -c "from pipelineV7 import PipelineV7; print('✓')"
python -c "from trt_pipeline.tracking import *; print('✓')"
```

### 3. SORT Tracker Test
```bash
python -c "
from sort import Sort
import numpy as np
tracker = Sort()
boxes = np.array([[100, 100, 200, 200]])
result = tracker.update(boxes)
print(f'✓ Tracked {len(result)} objects')
"
```

### 4. Full Pipeline Test
```bash
python main.py tracking --config config/config_test.json --type sort
```

---

## Backup Original Files

Before making changes:
```bash
cp pipelineV7.py pipelineV7.py.backup
cp trt_pipeline/tracking.py trt_pipeline/tracking.py.backup
```

Restore if needed:
```bash
cp pipelineV7.py.backup pipelineV7.py
cp trt_pipeline/tracking.py.backup trt_pipeline/tracking.py
```

---

## Alternative: Implement Custom Trackers (Advanced)

If you want ByteTrack or OC-Sort without boxmot:

**Option 1: Copy from Reference Implementation**
```bash
# ByteTrack
git clone https://github.com/ifzhang/ByteTrack
cp ByteTrack/yolox/tracker/byte_tracker.py tracking/bytetrack.py

# OC-Sort
git clone https://github.com/noahcao/OC_SORT
cp OC_SORT/trackers/ocsort_tracker.py tracking/ocsort.py
```

**Option 2: Use SORT Variations**
The existing `sort.py` can be extended with:
- Confidence-based association (ByteTrack style)
- Observation-centric updates (OC-Sort style)

Both require only numpy + scipy (already in requirements).

---

## Status: Ready to Apply ✅

All changes identified and documented. Proceed with:
1. Backup original files
2. Apply changes above
3. Run tests
4. Deploy to Jetson Nano

**Questions?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Section 5 (Troubleshooting)
