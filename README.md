# Traffic Monitoring Pipeline

This project is a traffic monitoring pipeline that utilizes YOLO for object detection, SORT for tracking, and various techniques for density estimation and heatmap generation. The system processes video footage to detect, track, and analyze traffic patterns while providing outputs such as tracked objects, density maps, and heatmaps.

The implementation is optimized for real-time performance on Jetson Nano using multi-threading to efficiently handle video processing and analysis.

## Features
- **Object Detection**: Uses YOLO for detecting vehicles (cars, bicycles, motorcycles, buses, trucks).
- **Object Tracking**: Implements SORT algorithm for tracking vehicles across frames.
- **Density Estimation**: Analyzes traffic congestion using histogram-based density calculation.
- **Heatmap Generation**: Generates heatmaps based on object movement patterns.
- **Data Export**: Saves results such as images, CSV files, and tracked object data.

## Installation

### Prerequisites
- Python 3.11.11
- CUDA-capable GPU and matching NVIDIA drivers

### 1) Download YOLOv7 weights
Grab the official `yolov7.pt` (or your preferred checkpoint) and place it at the project root:
```sh
curl -L -o yolov7.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### 2) Clone YOLOv7
Fetch the YOLOv7 repo next to this project (already present in this workspace as `yolov7/`):
```sh
git clone https://github.com/WongKinYiu/yolov7.git
```

### 3) Install dependencies
Create and activate a virtual environment (recommended):
```sh
python -m venv venv
venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate
```

Install project and YOLOv7 requirements:
```sh
pip install -r requirements.txt
pip install -r yolov7/requirements.txt
```

### 4) Create or adjust a config
Use the provided sample at [config/config_test.json](config/config_test.json) and update paths/zones as needed for your video.

### 5) Run the YOLOv7 pipeline
```sh
python main.py tracking --config config/config_test.json
```

## Output Files
- **Tracked Vehicles**: `output_directory/<lane>/<vehicle_class>/`
- **Density Analysis**: `output_directory/density/`
- **Heatmaps**: `output_directory/heatmap/`
- **CSV Reports**: `output_directory/tracking_data.csv`

## Troubleshooting
- Ensure the correct YOLO model is downloaded and compatible with your setup.
- Verify that CUDA is available using `torch.cuda.is_available()`.
- Check for missing dependencies using `pip check`.