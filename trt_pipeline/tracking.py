import torch
from trt_model import TRTModel
from frames_data import VideoFrameDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import json
from tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, lane_data_to_json
)
from shapely.geometry import Point
from shapely import contains
from boxmot.trackers import ByteTrack

class TrafficTracker:
    def __init__(
        self,
        config_path,
        engine_path,
        tracker,
        dict_class,
        batch_size=1
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("TrafficTracker")
        self.config = initial_config(config_path)
        self.model = TRTModel(
            engine_path=engine_path,
            input_shape=(1, 3, 640, 640),
            device=self.device
        )
        self.dataset = VideoFrameDataset(
            video_path=self.config['video'],
            skip=self.config['skip'],
            device=self.device,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        self.tracker = tracker
        self.dict_class = dict_class
        self.tracking_zone = parse_zones(self.config["tracking"])
        self.lane_data = initial_lane_data(self.tracking_zone, self.dict_class)

        self.preprocess_times = []
        self.infer_times = []
        self.tracking_times = []

    @staticmethod
    def collate_fn(batch):
        imgs, tensors, ratios, dwdhs = zip(*batch)
        tensors = torch.stack(tensors, dim=0)
        return imgs, tensors, ratios, dwdhs

    def run(self):
        for i, (process_times, imgs, input_tensor, ratios, dwdhs) in enumerate(tqdm(self.dataloader)):
            ratio = ratios[0]
            dwdh = dwdhs[0]
            dw, dh = dwdh[0], dwdh[1]
            input_tensor = input_tensor.to(self.device)
            self.preprocess_times.append(process_times[0])

            # Inference yolov7 tensorRT
            infer_time, outputs = self.model.infer(input_tensor)

            num = int(outputs["num_dets"][0].item())
            boxes = outputs["det_boxes"][0][:num].cpu().numpy()
            scores = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            # Tracking algorithm
            frame = imgs[0]
            start = time.perf_counter()
            results = self.tracker.update(dets, frame)
            end = time.perf_counter()
            tracking_time = end - start

            self.infer_times.append(infer_time)
            self.tracking_times.append(tracking_time)

            # Crossing logic
            for res in results:
                x1, y1, x2, y2, track_id, conf, class_id, _ = res
                if int(class_id) not in self.dict_class:
                    continue  # Not a tracked class

                # Convert detection coordinates to original image
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cx_orig, cy_orig = to_original_coords(cx, cy, dw, dh, ratio)
                x1o, y1o = to_original_coords(x1, y1, dw, dh, ratio)
                x2o, y2o = to_original_coords(x2, y2, dw, dh, ratio)

                for _, lane_info in self.lane_data.items():
                    if track_id in lane_info["cross_ids"]:
                        continue  # Already counted for this lane

                    lane_polygon = lane_info['polygon']
                    line = lane_info['line']
                    curr_side = side_of_line((cx_orig, cy_orig), line[0], line[1])

                    if contains(lane_polygon, Point(cx_orig, cy_orig)) and curr_side < 0:
                        lane_info["cross_ids"].add(int(track_id))
                        lane_info["count_cls"][int(class_id)] += 1
                        lane_info["cross_obj"].append({
                            "id": int(track_id),
                            "frame": i,
                            "class_id": int(class_id),
                            "bbox": (x1o, y1o, x2o, y2o)
                        })

        cleanup()
        self.logger.info("Resources cleaned up.")

        lane_data_to_json(self.lane_data, os.path.join(self.config['output'], "lane_data.json"))
        self.logger.info("Lane data saved to lane_data.json.")

        with open(os.path.join(self.config['output'], "performance.json"), "w") as f:
            json.dump({
                "preprocess_time": self.preprocess_times,
                "inference_time": self.infer_times,
                "tracking_time": self.tracking_times
            }, f, indent=2)
        self.logger.info("Performance data saved to performance.json.")

if __name__ == "__main__":
    config_path = "config/config_south_1.json"
    engine_path = "models/yolov7-tiny.engine"
    dict_class = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    tracker = ByteTrack(
        det_thresh=0.01,
        max_age=30,
        max_obs=100,
        min_hits=3,
        iou_threshold=0.3,
        per_class=False,
        nr_classes=80,
        asso_func="iou",
        is_obb=False,
        min_conf=0.1,
        track_thresh=0.6,
        match_thresh=0.8,
        frame_rate=25
    )

    traffic_tracker = TrafficTracker(
        config_path=config_path,
        engine_path=engine_path,
        tracker=tracker,
        dict_class=dict_class,
        batch_size=1
    )
    traffic_tracker.run()