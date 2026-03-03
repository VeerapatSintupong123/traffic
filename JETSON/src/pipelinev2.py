import os
import sys
import time
import numpy as np
import torch
import cv2 as cv
from shapely.geometry import Point
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm.sort import Sort
from algorithm.ocsort import OcSort
from trt_pipeline.trt_model import TRTModel
from trt_pipeline.video_stream import AsyncImageSaver
from trt_pipeline.tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, save_lane_data
)
from JETSON.src.jtop_logging import JTopMonitor

class PipelineV2:
    def __init__(self, config_path: str, engine_path: str, save_crop: bool = False, root_dir: str = None):
        self.logger = get_logger("JetsonPipelineV2")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

        # -- Configuration --
        self.save_crop = save_crop
        self.root_dir = root_dir
        self.engine_path = engine_path

        self.dict_class = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.target_classes = set(self.dict_class.keys())

        self.config = initial_config(config_path, root_dir=root_dir)
        self.video_name = self.config.get("video")
        self.output_name = self.config.get("output", "output")
        self.skip = max(1, int(self.config.get("skip", 1)))
        self.scale = self.config.get("scale", 1.0)
        self.tracker = self._initial_tracker(self.config)
        self.tracking_zone = parse_zones(self.config["tracking"])
        self.lane_data = initial_lane_data(self.config.get("lanes", {}), self.dict_class)

        # Model loading
        self.model = TRTModel(
            engine_path=engine_path,
            input_shape=(1, 3, 640, 640),
            device=self.device,
        )

        # -- Paths --
        self.VIDEO_DIR = os.path.join(self.root_dir, "video")
        self.OUTPUT_DIR = os.path.join(self.root_dir, "output", self.output_name)
        os.makedirs(self.VIDEO_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.video_path = os.path.join(self.VIDEO_DIR, self.video_name)
        if os.path.exists(self.video_path):
            self.logger.info(f"Video found: {self.video_path}")
        else:
            self.logger.error(f"Video not found: {self.video_path}")
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        if self.save_crop:
            self.image_saver = AsyncImageSaver()
            self.save_dir = {}
            for lane_name, _ in self.lane_data.items():
                for cls in list(self.dict_class.values()):
                    save_dir = os.path.join(self.OUTPUT_DIR, lane_name, cls)
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_dir[(lane_name, cls)] = save_dir
        else:
            self.image_saver = None
            self.save_dir = {}

        # -- Video Properties --
        self.original_width, self.original_height, self.fps = self._get_video_properties(self.video_path)
        self.ratio, self.dw, self.dh = self._calculate_transform_params(
            self.original_width, self.original_height
        )

        # -- Resource Monitoring --
        self.jtop_monitor = JTopMonitor()
        self.timing_stats = {"frames": []}
        self.resource_stats = []
    
    def _get_video_properties(self, video_path):
        """Get original video width, height, and fps."""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv.CAP_PROP_FPS))
        cap.release()

        if fps <= 0:
            fps = 25.0

        return width, height, fps

    def _calculate_transform_params(self, orig_width, orig_height, target_size=640):
        """Calculate letterbox transform parameters.
        
        Returns:
            ratio: scale factor from original to letterbox
            dw, dh: padding offsets (used to map back to original coords)
        """
        # Scale to fit in 640x640 while preserving aspect ratio
        r = min(target_size / orig_width, target_size / orig_height)
        
        # New unpadded dimensions
        new_w = int(orig_width * r)
        new_h = int(orig_height * r)
        
        # Padding needed
        dw = (target_size - new_w) / 2
        dh = (target_size - new_h) / 2
        
        return r, dw, dh

    def _initial_tracker(self, config):
        """Initialize tracker with configuration parameters.
        
        Supports tracker types: 'sort' (default), 'ocsort', 'bytetrack'
        """
        # Extract tracker config if it exists
        tracker_config = config.get("tracker", {}) if isinstance(config, dict) else {}
        
        # Get tracker type (default to 'sort')
        tracker_type = tracker_config.get("type", "sort").lower()
        
        # Extract parameters
        max_age = tracker_config.get("max_age", 30)
        min_hits = tracker_config.get("min_hits", 3)
        iou_threshold = tracker_config.get("iou_threshold", 0.3)
        
        self.logger.info(f"Initializing tracker: {tracker_type}")
        
        if tracker_type == "ocsort":
            det_thresh = tracker_config.get("det_thresh", 0.6)
            delta_t = tracker_config.get("delta_t", 3)
            inertia = tracker_config.get("inertia", 0.2)
            use_byte = tracker_config.get("use_byte", False)
            min_conf = tracker_config.get("min_conf", 0.1)
            self._ocsort_min_conf = float(min_conf)
            return OcSort(
                det_thresh=float(det_thresh),
                max_age=int(max_age),
                min_hits=int(min_hits),
                iou_threshold=float(iou_threshold),
                delta_t=int(delta_t),
                inertia=float(inertia),
                use_byte=bool(use_byte)
            )

        self._ocsort_min_conf = None
        return Sort(
            max_age=int(max_age),
            min_hits=int(min_hits),
            iou_threshold=float(iou_threshold)
        )
    
    def _find_closest_class(self, centroid, dets):
        """Find closest detection to centroid (fallback class assignment)."""
        if dets.size == 0:
            return None
        
        centers = np.stack([
            (dets[:, 0] + dets[:, 2]) / 2,
            (dets[:, 1] + dets[:, 3]) / 2
        ], axis=1)
        
        dists = np.linalg.norm(centers - np.array(centroid), axis=1)
        idx = int(np.argmin(dists))
        return int(dets[idx, 5])

    def _get_gstreamer_pipeline(self):
        """Build optimized GStreamer pipeline for hardware-accelerated decoding.
        
        Pipeline:
            filesrc -> avidemux -> h264parse -> nvv4l2decoder (HW decode)
            -> nvvidconv (HW resize/colorspace) -> appsink
        """
        # Use hardware decoder and scaler
        pipeline = (
            f"filesrc location={self.video_path} ! "
            "qtdemux ! h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! "
            "video/x-raw, width=640, height=640, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink"
        )
        
        self.logger.info("Using hardware-accelerated GStreamer pipeline")
        return pipeline

    def save_performance_log(self):
        self.logger.info("Saving performance logs...")

        # Resource stats
        log_file = os.path.join(self.OUTPUT_DIR, "performance_log.csv")
        with open(log_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.resource_stats[0].keys())
            writer.writeheader()
            for entry in self.resource_stats:
                writer.writerow(entry)
        
        # Processing time stats
        timing_file = os.path.join(self.OUTPUT_DIR, "timing_stats.csv")
        with open(timing_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.timing_stats[0].keys())
            writer.writeheader()
            for entry in self.timing_stats:
                writer.writerow(entry)

    def cleanup_save(self):
        self.logger.info("Cleaning up and saving outputs.")
        self.image_saver.cleanup()
        save_lane_data(self.lane_data, os.path.join(self.config["output"], "lane_data.json"))

    def _preprocess_frame(self, frame_bgr):
        """Preprocess frame: resize to 640x640 with letterbox, convert to CHW format and normalize."""
        t0 = time.perf_counter()
        
        # Letterbox resize using pre-calculated transform parameters
        target_size = 640
        h, w = frame_bgr.shape[:2]
        
        # Resize frame using pre-calculated scale factor
        new_w = int(w * self.ratio)
        new_h = int(h * self.ratio)
        resized = cv.resize(frame_bgr, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        
        # Create letterbox canvas
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # Place resized image on canvas using pre-calculated offsets
        dw_int = int(self.dw)
        dh_int = int(self.dh)
        canvas[dh_int:dh_int + new_h, dw_int:dw_int + new_w] = resized
        
        # HWC BGR -> CHW BGR
        img_chw = canvas.transpose(2, 0, 1)
        img_chw = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
        
        # To tensor
        input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        torch.cuda.synchronize(self.device)  # Ensure transfer complete
        
        return input_tensor, time.perf_counter() - t0

    def _postprocess_detections(self, outputs):
        """Extract and filter detections from model output."""
        t0 = time.perf_counter()
        
        num = int(outputs["num_dets"][0])
        boxes = outputs["det_boxes"][0][:num].cpu().numpy()
        scores = outputs["det_scores"][0][:num].cpu().numpy()
        classes = outputs["det_classes"][0][:num].cpu().numpy()
        
        dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)
        
        if dets.size:
            dets = dets[np.isin(dets[:, 5].astype(int), list(self.target_classes))]
        
        return dets, time.perf_counter() - t0

    def _run_tracker(self, dets):
        """Update tracker with detections."""
        t0 = time.perf_counter()
        
        boxes_only = dets[:, :4] if dets.size else np.empty((0, 4))
        
        if isinstance(self.tracker, OcSort):
            tracker_input = dets[:, :5] if dets.size else np.empty((0, 5))
            tracker_objects = self.tracker.update(tracker_input, min_conf=self._ocsort_min_conf)
        else:
            tracker_objects = self.tracker.update(boxes_only)
        
        return tracker_objects, time.perf_counter() - t0

    def _process_lane_crossings(self, tracker_objects, frame_bgr, dets, frame_idx):
        """Detect lane crossings and save cropped images."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        for x1, y1, x2, y2, track_id in tracker_objects:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_id = self._find_closest_class((cx, cy), dets)
            
            if class_id is None or class_id not in self.target_classes:
                continue

            # Map back to original coordinates
            cxo, cyo = to_original_coords(cx, cy, self.dw, self.dh, self.ratio)
            x1o, y1o = to_original_coords(x1, y1, self.dw, self.dh, self.ratio)
            x2o, y2o = to_original_coords(x2, y2, self.dw, self.dh, self.ratio)

            # Clamp to frame bounds
            x1c = int(max(0, min(w, x1o)))
            y1c = int(max(0, min(h, y1o)))
            x2c = int(max(0, min(w, x2o)))
            y2c = int(max(0, min(h, y2o)))

            if x2c <= x1c or y2c <= y1c:
                continue

            # Check each lane
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
                    lane["cross_obj"].append({
                        "id": int(track_id),
                        "frame": frame_idx,
                        "class_id": int(class_id),
                        "bbox": (x1c, y1c, x2c, y2c),
                    })

                    if self.save_crop and self.image_saver:
                        crop = frame_bgr[y1c:y2c, x1c:x2c]
                        save_path = os.path.join(
                            self.save_dir[(lane_name, self.dict_class[class_id])],
                            f"frame_{frame_idx}_id_{int(track_id)}.jpg"
                        )
                        self.image_saver.save(save_path, crop)

        return time.perf_counter() - t0

    def _log_timing_stats(self, frame_idx, timings):
        """Store frame timing statistics."""
        stat = {"frame_idx": frame_idx}
        stat.update({k: v * 1000 for k, v in timings.items()})  # Convert to ms
        self.timing_stats["frames"].append(stat)
        return stat  

    def _save_results(self, total_time, processed_frames):
        """Save performance and resource logs."""
        # Timing statistics
        timing_file = os.path.join(self.OUTPUT_DIR, "timing_stats.csv")
        if self.timing_stats["frames"]:
            keys = self.timing_stats["frames"][0].keys()
            with open(timing_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.timing_stats["frames"])
            self.logger.info(f"Saved timing stats to {timing_file}")

        # Resource statistics
        resource_file = os.path.join(self.OUTPUT_DIR, "resource_stats.csv")
        if self.resource_stats:
            keys = self.resource_stats[0].keys()
            with open(resource_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.resource_stats)
            self.logger.info(f"Saved resource stats to {resource_file}")

        # Summary
        fps = processed_frames / total_time if total_time > 0 else 0.0
        summary_file = os.path.join(self.OUTPUT_DIR, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PIPELINE EXECUTION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Total Frames: {processed_frames}\n")
            f.write(f"Total Time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {fps:.2f}\n")
            f.write(f"GStreamer Preprocessing: Enabled\n")
            f.write("="*60 + "\n")
        
        self.logger.info(f"Saved summary to {summary_file}")
        self.logger.info("Pipeline completed successfully.")

    def run(self):
        self.logger.info("Starting optimized pipeline with GStreamer preprocessing...")
        cv.setNumThreads(0)
        cv.ocl.setUseOpenCL(False)

        if self.jtop_monitor:
            self.jtop_monitor.start()
        total_start = time.perf_counter()
        processed_frames = 0

        gst_pipeline = self._get_gstreamer_pipeline()
        cap = cv.VideoCapture(gst_pipeline, cv.CAP_GSTREAMER)

        if not cap.isOpened():
            self.logger.error("Failed to open GStreamer pipeline. Falling back to standard OpenCV.")
            cap = cv.VideoCapture(self.video_path)

        total_start = time.perf_counter()
        processed_frames = 0
        max_frames = int(self.fps * 3600) if self.fps > 0 else int(25 * 3600)

        try:
            while cap.isOpened():
                frame_start = time.perf_counter()
                timings = {}

                ret, frame_bgr = cap.read()
                if not ret:
                    self.logger.info("End of video stream")
                    break

                # Frame skipping
                if processed_frames % self.skip != 0:
                    processed_frames += 1
                    continue

                # -- Preprocessing --
                input_tensor, preprocess_time = self._preprocess_frame(frame_bgr)
                timings['preprocess'] = preprocess_time

                # -- Inference --
                infer_time, outputs = self.model.infer(input_tensor)
                timings['inference'] = infer_time

                # -- Postprocessing --
                dets, postprocess_time = self._postprocess_detections(outputs)
                timings['postprocess'] = postprocess_time

                # -- Tracking --
                tracker_objects, tracking_time = self._run_tracker(dets)
                timings['tracking'] = tracking_time

                # -- Lane Crossing Detection --
                lane_cross_time = self._process_lane_crossings(
                    tracker_objects, frame_bgr, dets, frame_idx=processed_frames
                )
                timings['lane_crossing'] = lane_cross_time

                timings['total_frame'] = time.perf_counter() - frame_start

                # -- Logging --
                self._log_timing_stats(processed_frames, timings)

                if processed_frames % 50 == 0:
                    avg_fps = processed_frames / (time.perf_counter() - total_start)
                    self.logger.info(
                        f"Frame: {processed_frames:5d} | "
                        f"FPS: {avg_fps:6.2f} | "
                        f"Frame Time: {timings['total_frame']*1000:6.2f}ms"
                    )

                processed_frames += 1
                if processed_frames > max_frames:
                    break
        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise
        finally:
            cap.release()
            total_time = time.perf_counter() - total_start
            
            if self.image_saver:
                self.image_saver.stop()
            
            if self.jtop_monitor:
                self.jtop_monitor.stop()
                self.resource_stats = self.jtop_monitor.get_stats()
            
            cleanup()
            save_lane_data(self.lane_data, os.path.join(self.OUTPUT_DIR, "lane_data.json"))
            
            self._save_results(total_time, processed_frames)
            if total_time > 0:
                self.logger.info(f"Pipeline completed: {processed_frames} frames in {total_time:.2f}s ({processed_frames / total_time:.2f} FPS)")
            