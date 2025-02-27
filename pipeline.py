import os
import json
import time
import logging
import ast
from threading import Thread, Event
from queue import Queue

import copy
import cv2 as cv
import numpy as np

import torch
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from clean import process_images
from sort import Sort
from utils import (
    results_to_detection, is_point_on_lane, side_of_line,
    find_closest_class, location_support, save_crop_image,
    extract_background, segment_zone, color_density,
    save_csv, save_result, anomaly_detect
)

MODEL_NAME = "yolo11n.engine"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pipeline:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in config file: {config_path}")

        required_keys = ["video", "tracking", "density", "output", "scale"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")

        self.model = YOLO(MODEL_NAME, task= "detect")
        self.tracker = Sort()
        self.dict_class = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.input_frame = Queue(maxsize= 50)
        self.heatmap_frame = Queue(maxsize= 50)
        self.density_frame = Queue(maxsize= 50)
        self.output_frame = Queue(maxsize= 50)
        self.stop_threads = Event()

        self.frame_count = 0
        self.last_processed_frame_info = {"frame": None, "frame_index": 0.0}

        # input video
        self.video = cv.VideoCapture(self.config["video"])
        if not os.path.exists(self.config["video"]):
            raise FileNotFoundError(f"Video file not found: {self.config['video']}")

        # configuration
        self.tracking_zone = self._parse_zones(self.config["tracking"])
        self.density_zone = self._parse_zones(self.config["density"])
        self.lane_data = self._initial_lane_data()

        # heatmap
        self.accum_image = None
        self.first_frame = None
        self.previous_mask = None 

        # density
        self.densities = []

        # tracking
        self.prev_tracking_info = {}
        self.history_tracking = []

        # ouput directory
        self.output = self.config["output"]
        os.makedirs(f"{self.output}/density", exist_ok= True)
        os.makedirs(f"{self.output}/heatmap", exist_ok= True)

        for zone_name in self.lane_data.keys():
            for name in self.dict_class.values():
                os.makedirs(f"{self.output}/{zone_name}/{name}", exist_ok= True)

    def _parse_zones(self, zones):
        parsed_zones = []
        for zone in zones:
            parsed_zone = zone.copy()
            for key in ['lane', 'line']:
                if isinstance(parsed_zone.get(key), str):
                    try:
                        parsed_zone[key] = ast.literal_eval(parsed_zone[key])
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse {key} coordinates for {parsed_zone.get('name', 'Unknown zone')}")
                        continue
            parsed_zones.append(parsed_zone)
        return parsed_zones

    def _initial_lane_data(self):
        return {
            lane["name"]: {
                "cross_ids": set(),
                "cross_map": [],
                "dict_count": {key: 0 for key in self.dict_class.keys()},
            }
            for lane in self.tracking_zone
        }

    def warm_up_model(self):
        dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        self.model(dummy_frame)

    def video_to_frame(self):
        cap = self.video
        fps = cap.get(cv.CAP_PROP_FPS)

        try:
            while cap.isOpened() and not self.stop_threads.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
                self.last_processed_frame_info = {
                    "frame": frame,
                    "frame_index": current_frame_index
                }

                try:
                    if not self.input_frame.full():
                        self.input_frame.put(frame, timeout= 1)
                    if not self.density_frame.full():
                        self.density_frame.put(frame, timeout= 1)
                    if not self.heatmap_frame.full():
                        self.heatmap_frame.put(frame, timeout= 1)
                    self.frame_count += 1
                except Queue.full:
                    continue

                time.sleep(1.0 / fps)
        finally:
            self.stop_threads.set()
            cap.release()
            print("Video processing completed")

    def tracking(self):
        while not self.stop_threads.is_set():
            if not self.input_frame.empty():
                frame = self.input_frame.get()

                results = self.model(frame)
                response = results_to_detection(results, self.dict_class)
                detections = [item[:4] for item in response]
                map_class_id = {
                    ((item[0] + item[2]) // 2, (item[1] + item[3]) // 2): item[5]
                    for item in response
                }

                save_dict = {}
                save_dict["frame"] = self.frame_count
                save_dict["detections"] = []
                tracker_objects = self.tracker.update(np.array(detections))

                for obj in tracker_objects:
                    x1, y1, x2, y2, obj_id = map(int, obj)
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                    vehicle = {}
                    vehicle["id"] = obj_id
                    vehicle["bbox"] = [x1, y1, x2, y2]
                    vehicle["speed"] = 0
                    vehicle["angle"] = 0

                    # speed, direction and angle calculation
                    if obj_id in self.prev_tracking_info:
                        prev_centroid = self.prev_tracking_info[obj_id]
                        speed = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))  
                        vehicle["speed"] = speed

                        direction = (centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1])
                        angle_radians = np.arctan2(direction[1], direction[0])
                        angle_degrees = np.degrees(angle_radians) % 360
                        vehicle["angle"] = angle_degrees

                    save_dict["detections"].append(vehicle)

                    # update previous tracking info
                    self.prev_tracking_info[obj_id] = centroid

                    for tracking_info in self.tracking_zone:
                        try:
                            lane = tracking_info["lane"]
                            line = tracking_info["line"]
                            lane_name = tracking_info["name"]
                            lane_info = self.lane_data[lane_name]

                            if is_point_on_lane(centroid[0], centroid[1], lane):
                                curr_side = side_of_line(centroid, line[0], line[1])
                                same_location = location_support(lane_info["cross_map"], [x1, y1, x2, y2], angle_degrees)

                                if obj_id not in lane_info["cross_ids"] and curr_side < 0 and not same_location:
                                    class_id = find_closest_class(centroid, map_class_id)

                                    lane_info["cross_ids"].add(obj_id)
                                    lane_info["dict_count"][class_id] += 1

                                    save_path = f"{self.output}/{lane_name}/{self.dict_class[class_id]}"
                                    save_crop_image(frame, save_path, self.dict_class[class_id], x1, y1, x2, y2, obj_id)

                                    dict_ids = {
                                        "id": obj_id,
                                        "frame": self.frame_count,
                                        "angle": angle_degrees,
                                        "bbox": [x1, y1, x2, y2],
                                        "path": f"{save_path}/{self.dict_class[class_id]}_{obj_id}.jpg"
                                    }
                                    lane_info["cross_map"].append(dict_ids)

                        except Exception as e:
                            print(f"Error processing tracking info: {e}")
                            continue
                
                self.history_tracking.append(save_dict)

                if not self.output_frame.full():
                    self.output_frame.put((frame, tracker_objects))

    def histogram_density(self):
        while not self.stop_threads.is_set():
            if not self.density_frame.empty():
                frame = self.density_frame.get()

                scale_factor = self.config["scale"] / 100
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)

                base_frame = cv.imread(f"{self.output}/background.jpg")
                base_frame = cv.resize(base_frame, (new_width, new_height), interpolation=cv.INTER_AREA)
                diff_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                display_frame = cv.cvtColor(diff_frame, cv.COLOR_GRAY2BGR)

                save_dict = {}
                save_dict["frame"] = self.frame_count
                for parking in self.density_zone:
                    zone_name = parking["name"]
                    
                    zone = [(int(item[0] * scale_factor), int(item[1] * scale_factor)) for item in parking["lane"]]

                    zone_array = np.array(zone, dtype=np.int32)

                    segment_base = segment_zone(base_frame, zone)
                    segment_diff = segment_zone(diff_frame, zone)

                    hist_base = cv.calcHist([segment_base], [0], None, [256], [1, 255]).flatten()
                    hist_diff = cv.calcHist([segment_diff], [0], None, [256], [1, 255]).flatten()

                    result = np.maximum(hist_base - hist_diff, 0)
                    density = np.sum(result) / np.sum(hist_base) if np.sum(hist_base) > 0 else 0
                    save_dict[zone_name] = density

                    cv.fillPoly(display_frame, [zone_array], color_density(density))

                    zone_center = tuple(np.mean(zone_array, axis=0).astype(int))
                    cv.putText(display_frame, zone_name, zone_center, cv.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv.LINE_AA)
                
                self.densities.append(save_dict)
                if self.frame_count % 5 == 0:
                    cv.imwrite(f"{self.output}/density/frame_{self.frame_count}.jpg", display_frame)

    def heatmap_generator(self):
        background_segment = cv.bgsegm.createBackgroundSubtractorMOG()
        frame_reset_count = 25
        
        while not self.stop_threads.is_set():
            try:
                if not self.heatmap_frame.empty():
                    frame = self.heatmap_frame.get()

                    # Reset accumulator image every 25 frames
                    if self.frame_count % frame_reset_count == 1:
                        self.first_frame = copy.deepcopy(frame)
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        height, width = gray.shape[:2]
                        self.accum_image = np.zeros((height, width), np.uint8)
                        self.previous_mask = None
                    else:
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        current_mask = background_segment.apply(gray)

                        # Only process if the current mask is different from the previous one
                        if self.previous_mask is None or not np.array_equal(current_mask, self.previous_mask):
                            thresh = 2
                            maxValue = 2
                            ret, th1 = cv.threshold(current_mask, thresh, maxValue, cv.THRESH_BINARY)
                            self.accum_image = cv.add(self.accum_image, th1)
                            self.previous_mask = current_mask.copy()

                    color_image = cv.applyColorMap(self.accum_image, cv.COLORMAP_HOT)
                    result = cv.addWeighted(self.first_frame, 0.7, color_image, 0.7, 0)

                    if self.frame_count % frame_reset_count == 0:
                        output_path = f"{self.output}/heatmap/frame_{self.frame_count}.jpg"
                        if not cv.imwrite(output_path, result):
                            logging.error(f"Failed to save heatmap frame: {output_path}")

            except Exception as e:
                logging.error(f"Error in heatmap generation: {e}")

    def run(self):
        logging.basicConfig(level=logging.INFO)

        logging.info("Extract Background")
        extract_background(
            self.config["video"], 
            self.config["scale"], 
            self.config["output"]
        )

        # setup model for cleaning
        model_resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model_resnet.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logging.info("Tracking...")
        time.sleep(0.5)

        self.warm_up_model()

        threads = [
            Thread(target= self.video_to_frame),
            Thread(target= self.tracking),
            Thread(target= self.histogram_density),
            Thread(target= self.heatmap_generator)
        ]

        start = time.time()
        
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nStopping threads...")
            self.stop_threads.set()

        end = time.time()
        print(f"Last processed frame index: {int(self.last_processed_frame_info['frame_index'])}")
        print(f"Total Execution Time: {end - start:.2f} seconds")

        # save density histogram
        save_csv(f"{self.output}/density.csv", self.densities)

        # save history tracking
        file_path = f"{self.output}/tracking.json"
        with open(file_path, "w") as json_file:
            json.dump({"history": self.history_tracking}, json_file, indent= 4)

        logging.info("Cleaning...")
        time.sleep(0.5)

        # crossing map
        temp = {}
        for tracking_info in self.tracking_zone:
            lane_name = tracking_info["name"]
            lane_info = self.lane_data[lane_name]
            temp[lane_name] = process_images(
                lane_info["cross_map"],
                model_resnet, 
                transform
            )

        logging.info("Post-process")
        time.sleep(0.5)

        anomaly_detect(
            video_path= self.config["video"], 
            output_folder= self.output,
            dict_map= temp
        )

        # saving crossing map
        file_path = f"{self.output}/crossing.json"
        with open(file_path, "w") as json_file:
            json.dump(temp, json_file, indent= 2)

        save_result(self.output, self.config["video"], temp)