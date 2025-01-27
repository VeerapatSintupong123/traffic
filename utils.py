import argparse
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import contains
import cv2 as cv
from moviepy.editor import VideoFileClip
from sklearn import decomposition
import os

def is_point_on_lane(x, y, lane):
    area = Polygon(lane)
    return contains(area, Point(x, y))

def side_of_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def find_closest_class(centroid, class_id_points):
    min_dist = float('inf')
    closest_class = None
    for point, class_id in class_id_points.items():
        dist = np.sqrt((centroid[0] - point[0]) ** 2 + (centroid[1] - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_class = class_id
    return closest_class

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
    
def histogram_density(frame, config, parking_zone):
    scale_factor = config["scale"] / 100
    new_width = int(frame.shape[1] * scale_factor)
    new_height = int(frame.shape[0] * scale_factor)
    frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)

    base_frame = cv.imread(f"{config['output']}/background.jpg")
    base_frame = cv.resize(base_frame, (new_width, new_height), interpolation=cv.INTER_AREA)
    diff_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    display_frame = cv.cvtColor(diff_frame, cv.COLOR_GRAY2BGR)

    for parking in parking_zone:
        zone_name = parking["name"]
        
        zone = [(int(item[0] * scale_factor), int(item[1] * scale_factor)) for item in parking["lane"]]

        zone_array = np.array(zone, dtype=np.int32)

        segment_base = segment_zone(base_frame, zone)
        segment_diff = segment_zone(diff_frame, zone)

        hist_base = cv.calcHist([segment_base], [0], None, [256], [1, 255]).flatten()
        hist_diff = cv.calcHist([segment_diff], [0], None, [256], [1, 255]).flatten()

        result = np.maximum(hist_base - hist_diff, 0)
        density = np.sum(result) / np.sum(hist_base) if np.sum(hist_base) > 0 else 0

        cv.fillPoly(display_frame, [zone_array], color_density(density))

        zone_center = tuple(np.mean(zone_array, axis=0).astype(int))
        cv.putText(display_frame, zone_name, zone_center, cv.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv.LINE_AA)
        
    save_path = f"{config['output']}/density"
    len_dir = len(os.listdir(save_path))    
    cv.imwrite(f"{config['output']}/density/frame_{len_dir}.jpg", display_frame)

def save_crop_image(frame, save_path, class_id, x1, y1, x2, y2):
    len_dir = len(os.listdir(save_path))
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped_frame = frame[y1:y2, x1:x2]
    resized_frame = cv.resize(cropped_frame, (224, 224), interpolation=cv.INTER_AREA)
    cv.imwrite(f"{save_path}/{class_id}_{len_dir + 1}.jpg", resized_frame)
    
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))