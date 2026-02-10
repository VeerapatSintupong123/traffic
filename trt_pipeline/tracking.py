import torch
from trt_model import TRTModel
from video_stream import VideoStream, AsyncImageSaver, letterbox
import numpy as np
import os
import time
from tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, save_lane_data,
)
from shapely.geometry import Point
from shapely import contains
from boxmot.trackers import ByteTrack
import cv2 as cv
from payload import Payload

class TrafficTracker:
    def __init__(
        self,
        config_path,
        engine_path,
        tracker,
        dict_class,
        save_crop
    ):
        self.logger = get_logger("TrafficTracker")
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

        self.tracker = tracker
        self.dict_class = dict_class

        self.tracking_zone = parse_zones(self.config["tracking"])
        self.lane_data = initial_lane_data(self.tracking_zone, self.dict_class)

        # API
        self.payload = Payload(intersection_id="INT-001", camera_id="CAM-01")

        self.image_saver = AsyncImageSaver()
        self.save_dir = {}
        for lane_name, _ in self.lane_data.items():
            for cls in list(dict_class.values()):
                save_dir = os.path.join(
                    self.config["output"],
                    lane_name,
                    cls
                )
                os.makedirs(save_dir, exist_ok=True)
                self.save_dir[(lane_name, cls)] = save_dir

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

        while True:
            item = stream.read()
            if item is None:
                break

            frame_idx, frame_bgr = item
            if frame_idx % 10 == 0:
                print(f"\rframe: {frame_idx}", end="", flush=True)

            if frame_idx > max_frames:
                break

            # ------------------ preprocess ------------------
            img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            img_lb, ratio, (dw, dh) = letterbox(
                img_rgb, new_shape=(640, 640), auto=False
            )

            img_chw = img_lb.transpose(2, 0, 1)
            img_chw = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)

            # ------------------ inference -------------------
            _, outputs = self.model.infer(input_tensor)
            num = int(outputs["num_dets"][0])
            boxes = outputs["det_boxes"][0][:num].cpu().numpy()
            scores = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            # ------------------ tracking --------------------
            results = self.tracker.update(dets, frame_bgr)

            # ------------------ post logic ------------------
            h, w = frame_bgr.shape[:2]

            for x1, y1, x2, y2, track_id, _, class_id, _ in results:
                class_id = int(class_id)
                if class_id not in self.dict_class:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
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
                            crop = frame_bgr[y1c:y2c, x1c:x2c]
                            save_path = os.path.join(
                                self.save_dir[(lane_name, self.dict_class[class_id])],
                                f"frame_{frame_idx}_id_{track_id}.jpg"
                            )
                            self.image_saver.save(save_path, crop)

        total_time = time.perf_counter() - total_start
        self.logger.info(f"Total time: {total_time:.2f}s")

        cleanup()
        self.image_saver.stop()
        self.logger.info("Resources cleaned up.")

        save_lane_data(self.lane_data, os.path.join(self.config["output"], "lane_data.json"))
        self.logger.info("Outputs saved successfully.")

if __name__ == "__main__":
    config_path = "config/config_south_1.json"
    engine_path = "models/yolov7-tiny.engine"
    dict_class = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    tracker = ByteTrack(
        min_conf = 0.3,
        track_thresh = 0.5,
        match_thresh = 0.7,
        track_buffer = 25,
        frame_rate = 25,
    )

    traffic_tracker = TrafficTracker(
        config_path=config_path,
        engine_path=engine_path,
        tracker=tracker,
        dict_class=dict_class,
        save_crop=True,
    )
    traffic_tracker.run()