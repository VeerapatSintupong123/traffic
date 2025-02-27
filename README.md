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
Ensure you have Python 3.11.11 installed. The project requires CUDA-compatible GPU for optimal performance.

### Step 1: Clone the Repository
```sh
git clone https://github.com/your-repository.git
cd your-repository
```

### Step 2: Install Dependencies
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install required packages:
```sh
pip install -r requirements.txt
```

### Step 3: Export Yolo to TensorRT
Ensure you have the YOLO model engine file placed in the project directory:
```sh
yolo export model=yolo11n.pt format=engine 
```

### Step 4: Run the Pipeline
Prepare a configuration JSON file (`config.json`) with required parameters:
```json
{
  "video": "input_video.mp4",
  "tracking": [ { "name": "lane1", "lane": "[...]", "line": "[...]" } ],
  "density": [ { "name": "zone1", "lane": "[...]" } ],
  "output": "output_directory",
  "scale": 100
}
```

Execute the script:
```sh
python main.py --config config.json
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