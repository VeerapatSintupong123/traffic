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
    parse_zones, side_of_line, save_lane_data, save_performance_data
)
from shapely.geometry import Point
from shapely import contains
from boxmot.trackers import ByteTrack
import cv2 as cv

class TrafficTracker:
    def __init__(
        self,
        config_path,
        engine_path,
        tracker,
        dict_class,
        save_crop: bool,
        batch_size: int = 1,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("TrafficTracker")

        self.config = initial_config(config_path)
        self.save_crop = save_crop

        self.skip = int(self.config.get("skip", 0))
        self.frame_stride = max(1, self.skip)

        self.model = TRTModel(
            engine_path=engine_path,
            input_shape=(1, 3, 640, 640),
            device=self.device,
        )

        self.dataset = VideoFrameDataset(
            video_path=self.config["video"],
            skip=self.skip,
            device=self.device,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn,
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
        process_time, imgs, tensors, ratios, dwdhs = zip(*batch)
        return process_time, imgs, torch.stack(tensors, 0), ratios, dwdhs

    def run(self):
        for idx, (proc_times, imgs, input_tensor, ratios, dwdhs) in enumerate(tqdm(self.dataloader)):
            ratio = ratios[0]
            dw, dh = dwdhs[0]
            frame_idx = idx * self.frame_stride

            self.preprocess_times.append(proc_times[0])

            input_tensor = input_tensor.to(self.device)

            # Inference
            infer_time, outputs = self.model.infer(input_tensor)
            self.infer_times.append(infer_time)

            num = int(outputs["num_dets"][0].item())
            boxes = outputs["det_boxes"][0][:num].cpu().numpy()
            scores = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()

            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            # Tracking
            frame = imgs[0].copy()
            start = time.perf_counter()
            results = self.tracker.update(dets, frame)
            self.tracking_times.append(time.perf_counter() - start)

            h, w = frame.shape[:2]

            for x1, y1, x2, y2, track_id, _, class_id, _ in results:
                class_id = int(class_id)
                if class_id not in self.dict_class:
                    continue

                # Convert coords
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cxo, cyo = to_original_coords(cx, cy, dw, dh, ratio)
                x1o, y1o = to_original_coords(x1, y1, dw, dh, ratio)
                x2o, y2o = to_original_coords(x2, y2, dw, dh, ratio)

                # Clamp + cast
                x1c = int(max(0, min(w, x1o)))
                y1c = int(max(0, min(h, y1o)))
                x2c = int(max(0, min(w, x2o)))
                y2c = int(max(0, min(h, y2o)))

                if x2c <= x1c or y2c <= y1c:
                    continue

                for lane_name, lane in self.lane_data.items():
                    if track_id in lane["cross_ids"]:
                        continue

                    if (
                        contains(lane["polygon"], Point(cxo, cyo))
                        and side_of_line((cxo, cyo), lane["line"][0], lane["line"][1]) < 0
                    ):
                        lane["cross_ids"].add(int(track_id))
                        lane["count_cls"][class_id] += 1
                        lane["cross_obj"].append(
                            {
                                "id": int(track_id),
                                "frame": frame_idx,
                                "class_id": class_id,
                                "bbox": (x1c, y1c, x2c, y2c),
                            }
                        )

                        if self.save_crop:
                            save_dir = os.path.join(
                                self.config["output"],
                                lane_name,
                                self.dict_class[class_id],
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            cv.imwrite(
                                os.path.join(
                                    save_dir,
                                    f"frame_{frame_idx}_id_{track_id}.jpg",
                                ),
                                frame[y1c:y2c, x1c:x2c],
                            )

        cleanup()
        self.logger.info("Resources cleaned up.")

        save_lane_data(self.lane_data, os.path.join(self.config["output"], "lane_data.json"))
        save_performance_data(
            self.config,
            self.preprocess_times,
            self.infer_times,
            self.tracking_times,
        )

        self.logger.info("Outputs saved successfully.")

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
        save_crop=True,
        batch_size=1
    )
    traffic_tracker.run()