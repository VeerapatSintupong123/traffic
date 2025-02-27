import os
import pytz
import csv
import json
import cv2 as cv
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import TripletModel

from PIL import Image, ImageFont, ImageDraw
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import contains
from moviepy.editor import VideoFileClip
from sklearn import decomposition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_point_on_lane(x, y, lane):
    area = Polygon(lane)
    return contains(area, Point(x, y))

def side_of_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def results_to_detection(results, dict_class):
    return [
        [
            *map(int, box.xyxy[0].cpu().numpy()), # x1, y1, x2, y2
            box.conf[0].cpu().item(), # confident
            int(box.cls[0]) # class id
        ]
        for result in results
        for box in result.boxes
        if int(box.cls[0]) in dict_class
    ]

def location_support(cross_map, bbox, iou_threshold=0.4):
    x1, y1, x2, y2 = bbox
    current_area = (x2 - x1) * (y2 - y1)
    
    for item in cross_map:

        item_x1, item_y1, item_x2, item_y2 = item["bbox"]
        item_area = (item_x2 - item_x1) * (item_y2 - item_y1)
        
        x_overlap = max(0, min(x2, item_x2) - max(x1, item_x1))
        y_overlap = max(0, min(y2, item_y2) - max(y1, item_y1))
        intersection_area = x_overlap * y_overlap
        
        union_area = current_area + item_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0

        if iou > iou_threshold:
            return True
            
    return False

def find_closest_class(centroid, class_id_points):
    min_dist = float('inf')
    closest_class = None
    for point, class_id in class_id_points.items():
        dist = np.sqrt((centroid[0] - point[0]) ** 2 + (centroid[1] - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_class = class_id
    return closest_class

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")
    
def create_data_matrix_from_video(clip, scale):
    fps = clip.fps
    duration = clip.duration
    frames = []

    for i in range(int(fps * duration)):
        frame = clip.get_frame(i / float(fps))
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        H, W = gray_frame.shape
        dims = (int(W * (scale/100)), int(H * (scale/100)))

        resized_frame = cv.resize(gray_frame, dims, interpolation=cv.INTER_AREA)
        frames.append(resized_frame.flatten())
    
    return np.array(frames).T

def extract_background(video_path, scale, output_path):
    clip = VideoFileClip(video_path)
    W, H = clip.size

    matrix = create_data_matrix_from_video(clip, scale)    
    u, s, v = decomposition.randomized_svd(matrix, 1)
    low_rank = u @ np.diag(s) @ v
    background =  np.array(low_rank[:, 500]).reshape((int(H * (scale/100)), int(W * (scale/100))))

    cv.imwrite(f"{output_path}/background.jpg", background)

def segment_zone(frame, zone):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    zone = np.array(zone, dtype=np.int32)

    cv.fillPoly(mask, [zone], 255)

    if len(frame.shape) == 3:
        mask = cv.merge([mask] * 3)
    result = cv.bitwise_and(frame, mask)

    return result

def color_density(density):
    if 0.00 <= density < 0.20:
        return (255, 69, 0)   # #1E90FF
    elif 0.20 <= density < 0.40:
        return (255, 165, 0)   # #40E0D0
    elif 0.40 <= density < 0.60:
        return (173, 255, 47)  # #ADFF2F
    elif 0.60 <= density < 0.80:
        return (64, 224, 208)  # #FFA500
    elif 0.80 <= density <= 1.00:
        return (30, 144, 255)  # #FF4500
    else:
        return (0, 0, 0)

def save_crop_image(frame, save_path, class_id, x1, y1, x2, y2, id):
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped_frame = frame[y1:y2, x1:x2]
    resized_frame = cv.resize(cropped_frame, (224, 224), interpolation=cv.INTER_AREA)
    cv.imwrite(f"{save_path}/{class_id}_{id}.jpg", resized_frame)
    
def save_csv(ouput, data):
    columns = data[0].keys()
    with open(ouput, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames= columns)
        writer.writeheader()
        writer.writerows(data)

def save_result(output, video_path, dict_map):
    file_path = f"{output}/result.txt"

    video = cv.VideoCapture(video_path)
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = int(video.get(cv.CAP_PROP_FOURCC))

    thailand_tz = pytz.timezone('Asia/Bangkok')
    thailand_time = datetime.now(pytz.UTC).astimezone(thailand_tz)
    current_time = thailand_time.strftime("%Y-%m-%d %H:%M:%S")

    text = f"""================================
Video Analysis Report
================================
Generated on: {current_time}

---- Video Properties ----
- Frame Width: {frame_width}
- Frame Height: {frame_height}
- Frames per Second (FPS): {fps}
- Total Frames: {frame_count}
- Video Codec (FOURCC): {fourcc}

---- Lane Information ----"""

    sum_vehicle = 0
    for lane_name, vehicles in dict_map.items():
        vehicle_ids = [vehicle["id"] for vehicle in vehicles]

        vehicle_counts = {}
        for vehicle in vehicles:
            path_parts = vehicle["path"].split('/')
            vehicle_type = path_parts[-2]
            
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1
            else:
                vehicle_counts[vehicle_type] = 1
        
        text += f"\nLane: {lane_name}\n"
        text += f"- Vehicle Count: {len(vehicles)}\n"
        text += f"- Vehicle IDs: {', '.join(map(str, vehicle_ids))}\n"
        
        for vehicle_type, count in vehicle_counts.items():
            text += f"  - {vehicle_type.capitalize()}: {count}\n"
        
        sum_vehicle += len(vehicles)

    text += f"\nTotal Vehicle Count: {sum_vehicle}\n"
    text += "=" * 30 + "\n"

    with open(file_path, "w") as file:
        file.write(text)

    print(f"Results saved to {file_path}")

def get_feature_image(image_path, model):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(transformed_image).squeeze()

    return feature

def anomaly_detect(video_path, output_folder, dict_map):
    """
    Detects anomalies in a video based on image features.
    
    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save results.
        dict_map (list): List of dictionaries with object details ({id, frame, angle, bbox, path}).
    """
    anomaly_labels = ["tuktuk", "ambulance"]
    anomaly_images = {item: [] for item in anomaly_labels}

    checkpoint = torch.load("resnet_checkpoint.pth", map_location=device)
    model = TripletModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open("resnet_features.json", mode="r") as file_json:
        dict_features = json.load(file_json)

    labels = list(dict_features.keys())
    features = [torch.tensor(f).to(device) for f in dict_features.values()]

    for _, items in dict_map.items():
        for item in items:
            image_path = item["path"]
            test_feature = get_feature_image(image_path, model)

            similarity_list = [F.cosine_similarity(test_feature, f, dim=0).item() for f in features]
            max_index = np.argmax(similarity_list)
            detected_label = labels[max_index]

            if detected_label in anomaly_labels and similarity_list[max_index] > 0.96:
                anomaly_images[detected_label].append(image_path)

    for label, images in anomaly_images.items():
        print(f"🚨 {label}")
        for image in images: print(f"- {image}")