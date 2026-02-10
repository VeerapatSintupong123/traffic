import os
import sys
import time
import logging
import numpy as np
import torch
import cv2 as cv
from shapely.geometry import Point

# Add parent directory to path to find trt_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sort import Sort
from trt_pipeline.trt_model import TRTModel
from trt_pipeline.video_stream import VideoStream, AsyncImageSaver, letterbox
from trt_pipeline.tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, save_lane_data
)


class Pipeline:
    def __init__(self, config_path: str, engine_path: str, save_crop: bool = False):
        self.logger = get_logger("JetsonPipeline")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

        self.config = initial_config(config_path)
        self.save_crop = save_crop

        self.skip = int(self.config.get("skip", 0))
        self.frame_stride = max(1, self.skip)

        self.model = TRTModel(
            engine_path=engine_path,
            input_shape=(1, 3, 640, 640),
            device=self.device,
        )

        self.tracker = self.initial_tracker(self.config)
        self.dict_class = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.target_classes = set(self.dict_class.keys())

        self.tracking_zone = parse_zones(self.config["tracking"])
        self.lane_data = initial_lane_data(self.tracking_zone, self.dict_class)

        self.image_saver = AsyncImageSaver()
        self.save_dir = {}
        for lane_name, _ in self.lane_data.items():
            for cls in list(self.dict_class.values()):
                save_dir = os.path.join(self.config["output"], lane_name, cls)
                os.makedirs(save_dir, exist_ok=True)
                self.save_dir[(lane_name, cls)] = save_dir

    def initial_tracker(self, config):
        if config is None:
            return Sort()
        else:
            if hasattr(config, "tracker"):
                if config["tracker"].get("type", "sort").lower() == "ocsort":
                    return Sort(max_age=config["tracker"].get("max_age", 30),
                                min_hits=config["tracker"].get("min_hits", 3),
                                iou_threshold=config["tracker"].get("iou_threshold", 0.3),
                                use_occlusion=True)
                elif config["tracker"].get("type", "sort").lower() == "bytetrack":
                    return Sort(max_age=config["tracker"].get("max_age", 30),
                                min_hits=config["tracker"].get("min_hits", 3),
                                iou_threshold=config["tracker"].get("iou_threshold", 0.3),
                                use_byte=True)

    def _closest_class_id(self, centroid, dets):
        if dets.size == 0:
            return None
        centers = np.stack([
            (dets[:, 0] + dets[:, 2]) / 2,
            (dets[:, 1] + dets[:, 3]) / 2
        ], axis=1)
        dists = np.linalg.norm(centers - np.array(centroid), axis=1)
        idx = int(np.argmin(dists))
        return int(dets[idx, 5])

    def run(self):
        cv.setNumThreads(0)
        cv.ocl.setUseOpenCL(False)
        max_frames = int(25 * 3600)

        stream = VideoStream(
            video_path=self.config["video"],
            skip=self.frame_stride,
            queue_size=2,
        )

        total_start = time.perf_counter()
        processed_frames = 0

        while True:
            item = stream.read()
            if item is None:
                break

            frame_idx, frame_bgr = item
            processed_frames = frame_idx
            if frame_idx % 10 == 0:
                print(f"\rframe: {frame_idx}", end="", flush=True)

            if frame_idx > max_frames:
                break

            img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=(640, 640), auto=False)

            img_chw = img_lb.transpose(2, 0, 1)
            img_chw = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)

            _, outputs = self.model.infer(input_tensor)
            num = int(outputs["num_dets"][0])
            boxes = outputs["det_boxes"][0][:num].cpu().numpy()
            scores = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            if dets.size:
                dets = dets[np.isin(dets[:, 5].astype(int), list(self.target_classes))]

            boxes_only = dets[:, :4] if dets.size else np.empty((0, 4))
            tracker_objects = self.tracker.update(boxes_only)

            h, w = frame_bgr.shape[:2]

            for x1, y1, x2, y2, track_id in tracker_objects:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_id = self._closest_class_id((cx, cy), dets)
                if class_id is None or class_id not in self.target_classes:
                    continue

                cxo, cyo = to_original_coords(cx, cy, dw, dh, ratio)
                x1o, y1o = to_original_coords(x1, y1, dw, dh, ratio)
                x2o, y2o = to_original_coords(x2, y2, dw, dh, ratio)

                x1c = int(max(0, min(w, x1o)))
                y1c = int(max(0, min(h, y1o)))
                x2c = int(max(0, min(w, x2o)))
                y2c = int(max(0, min(h, y2o)))

                if x2c <= x1c or y2c <= y1c:
                    continue

                for lane_name, lane in self.lane_data.items():
                    if int(track_id) in lane["cross_ids"]:
                        continue

                    if (
                        lane["polygon"] is not None
                        and lane["polygon"].contains(Point(cxo, cyo))
                        and side_of_line((cxo, cyo), lane["line"][0], lane["line"][1]) < 0
                    ):
                        lane["cross_ids"].add(int(track_id))
                        lane["count_cls"][class_id] += 1
                        lane["cross_obj"].append(
                            {
                                "id": int(track_id),
                                "frame": frame_idx,
                                "class_id": int(class_id),
                                "bbox": (x1c, y1c, x2c, y2c),
                            }
                        )

                        if self.save_crop:
                            crop = frame_bgr[y1c:y2c, x1c:x2c]
                            save_path = os.path.join(
                                self.save_dir[(lane_name, self.dict_class[int(class_id)])],
                                f"frame_{frame_idx}_id_{int(track_id)}.jpg"
                            )
                            self.image_saver.save(save_path, crop)

        total_time = time.perf_counter() - total_start
        fps = (processed_frames / total_time) if total_time > 0 else 0.0
        self.logger.info(f"Total time: {total_time:.2f}s | FPS: {fps:.2f}")

        inference_path = os.path.join(self.config["output"], "inference.txt")
        with open(inference_path, "w") as f:
            f.write(f"Total frames: {processed_frames}\n")
            f.write(f"Total time (s): {total_time:.2f}\n")
            f.write(f"Average FPS: {fps:.2f}\n")

        cleanup()
        self.image_saver.stop()
        self.logger.info("Resources cleaned up.")

        save_lane_data(self.lane_data, os.path.join(self.config["output"], "lane_data.json"))
        self.logger.info("Outputs saved successfully.")