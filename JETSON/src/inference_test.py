import os
import sys
import time
import torch
import numpy as np
import cv2 as cv
from jtop import jtop
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from trt_pipeline.trt_model import TRTModel
from trt_pipeline.video_stream import VideoStream, letterbox
from trt_pipeline.tools import get_logger

ENGINE_PATH = '/home/schauto/traffic/models/yolov7-tiny.engine'
VIDEO_PATH = '/home/schauto/traffic/video/south_video.avi'

def get_system_resources():
    """Get system resource usage (CPU, RAM)."""
    try:
        with jtop() as jetson:
            stats = jetson.stats
            cpu_info = stats.get('CPU', {})
            if isinstance(cpu_info, dict):
                cpu_percent = cpu_info.get('total', {}).get('val', 0)
            else:
                cpu_percent = float(cpu_info)

            ram_stats = stats.get('RAM', {})
            ram_mb = ram_stats.get('use', 0) / 1024
            ram_tot = ram_stats.get('tot', 1)
            ram_percent = (ram_stats.get('use', 0) / ram_tot * 100) if ram_tot > 0 else 0
            return cpu_percent, ram_mb, ram_percent
    except ImportError:
        print("jtop library not found. Cannot get system resources.")
    except Exception as e:
        print(f"Could not get system resources: {e}")
    return 0, 0, 0

def main():
    logger = get_logger("InferenceTest")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Model Loading ---
    t0 = time.perf_counter()
    try:
        model = TRTModel(
            engine_path=ENGINE_PATH,
            input_shape=(1, 3, 640, 640),
            device=device,
        )
        logger.info("TensorRT engine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load TensorRT engine: {e}")
        return
    load_time = time.perf_counter() - t0
    logger.info(f"Model loading time: {load_time * 1000:.2f} ms")

    # --- Video Stream Initialization ---
    stream = VideoStream(video_path=VIDEO_PATH, skip=1, queue_size=2)

    timing_stats = defaultdict(list)
    total_start = time.perf_counter()
    processed_frames = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    while True:
        frame_start = time.perf_counter()
        stage_timings = {}

        # --- Video Read ---
        t0 = time.perf_counter()
        item = stream.read()
        if item is None:
            break
        frame_idx, frame_bgr = item
        stage_timings['video_read'] = time.perf_counter() - t0

        processed_frames += 1
        if frame_idx % 10 == 0:
            print(f"\rProcessing frame: {frame_idx}", end="", flush=True)

        # --- Preprocessing ---
        start_event.record()

        input_img, _, _ = letterbox(frame_bgr, (640, 640))
        # HWC to CHW, BGR to RGB
        input_img = input_img.transpose((2, 0, 1))[::-1]
        input_img = np.ascontiguousarray(input_img)

        input_tensor = torch.from_numpy(input_img).to(device).float()
        input_tensor = input_tensor.div(255.0).unsqueeze(0)
        end_event.record()
        torch.cuda.synchronize() # Wait for preprocessing to finish to get its time
        stage_timings['preprocessing_gpu'] = start_event.elapsed_time(end_event)

        # --- Inference ---
        try:
            inference_time, outputs = model.infer(input_tensor)
            num_dets = int(outputs["num_dets"][0])
        except Exception as e:
            logger.error(f"Inference failed on frame {frame_idx}: {e}")
            continue
        stage_timings['inference'] = inference_time

        frame_time = time.perf_counter() - frame_start
        stage_timings['total_frame'] = frame_time

        for stage, duration in stage_timings.items():
            timing_stats[stage].append(duration / 1000.0 if 'gpu' in stage else duration)

        if frame_idx > 0 and frame_idx % 100 == 0:
            cpu_percent, ram_mb, ram_percent = get_system_resources()
            gpu_alloc = torch.cuda.memory_allocated(device) / 1024**2
            gpu_reserved = torch.cuda.memory_reserved(device) / 1024**2
            logger.info(f"\n--- Frame {frame_idx} Stats ---")
            logger.info(f"  Detections: {num_dets}")
            logger.info(f"  Frame Time: {frame_time * 1000:.2f} ms")
            logger.info(f"  CPU: {cpu_percent:.1f}% | RAM: {ram_mb:.1f}MB ({ram_percent:.1f}%) | GPU: {gpu_alloc:.1f}MB / {gpu_reserved:.1f}MB")

    total_time = time.perf_counter() - total_start
    fps = (processed_frames / total_time) if total_time > 0 else 0.0
    logger.info(f"\n\n--- Pipeline Finished ---")
    logger.info(f"Total frames processed: {processed_frames}")
    logger.info(f"Total time: {total_time:.2f}s | Average FPS: {fps:.2f}")
    
    logger.info("\n--- Performance Summary (ms) ---")
    for stage, timings in timing_stats.items():
        avg_ms = np.mean(timings) * 1000
        std_ms = np.std(timings) * 1000
        logger.info(f"  {stage:20s}: avg={avg_ms:6.2f}ms, std={std_ms:5.2f}ms")
    logger.info("--------------------------------\n")

    stream.stop()

if __name__ == "__main__":
    main()