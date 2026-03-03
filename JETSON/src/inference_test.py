import os
import sys
import time
import torch
import numpy as np
from jtop import jtop

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from trt_pipeline.trt_model import TRTModel
from trt_pipeline.tools import get_logger

ENGINE_PATH = '/home/schauto/traffic/models/yolov7-tiny.engine'

def get_system_resources():
    """Get system resource usage (CPU, RAM)."""
    try:
        with jtop() as jetson:
            stats = jetson.stats
            cpu_percent = stats.get('CPU', {}).get('total', {}).get('val', 0)
            ram_stats = stats.get('RAM', {})
            ram_mb = ram_stats.get('use', 0) / 1024
            ram_percent = ram_stats.get('use', 0) / ram_stats.get('tot', 1) * 100
            return cpu_percent, ram_mb, ram_percent
    except Exception as e:
        print(f"Could not get system resources: {e}")
    return 0, 0, 0

def main():
    logger = get_logger("InferenceTest")
    device = torch.device(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
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

    # --- Dummy Input Creation ---
    t0 = time.perf_counter()
    dummy_input_np = np.random.rand(1, 3, 640, 640).astype(np.float32)
    input_tensor = torch.from_numpy(dummy_input_np).to(device)
    input_creation_time = time.perf_counter() - t0
    logger.info(f"Dummy input creation time: {input_creation_time * 1000:.2f} ms")

    # --- Inference ---
    t0 = time.perf_counter()
    try:
        _, outputs = model.infer(input_tensor)
        logger.info("Inference successful.")
        # Optionally, print some output details
        num_dets = int(outputs["num_dets"][0])
        logger.info(f"Detections found: {num_dets}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return
    inference_time = time.perf_counter() - t0
    logger.info(f"Inference time: {inference_time * 1000:.2f} ms")

    # --- Resource Logging ---
    cpu_percent, ram_mb, ram_percent = get_system_resources()
    gpu_alloc = torch.cuda.memory_allocated(device) / 1024**2
    gpu_reserved = torch.cuda.memory_reserved(device) / 1024**2

    logger.info("\n--- Resource Usage ---")
    logger.info(f"  CPU Usage: {cpu_percent:.1f}%")
    logger.info(f"  RAM Usage: {ram_mb:.1f} MB ({ram_percent:.1f}%)")
    logger.info(f"  GPU Memory: {gpu_alloc:.1f} MB allocated, {gpu_reserved:.1f} MB reserved")
    logger.info("----------------------\n")

if __name__ == "__main__":
    main()