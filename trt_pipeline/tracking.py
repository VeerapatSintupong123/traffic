import torch
from trt_model import TRTModel
from frames_data import VideoFrameDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from boxmot.trackers import ByteTrack
import numpy as np
import time
from utils import get_logger, cleanup
import matplotlib.pyplot as plt

logger = get_logger("Tracking")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    imgs, tensors, ratios, dwdhs = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return imgs, tensors, ratios, dwdhs

model = TRTModel(
    engine_path="models\yolov7-tiny.engine", 
    input_shape=(1, 3, 640, 640), 
    device=device
)

dataset = VideoFrameDataset(
    video_path="video/south_video.avi",
    skip=2,
    device=device,
)

dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True,
    collate_fn=collate_fn
)

tracker = ByteTrack(
    det_thresh=0.01,          # Keep low; ByteTrack handles confidence internally
    max_age=30,               # Frames before track removal
    max_obs=100,              # >= max_age + 5 (history for smoothing)
    min_hits=3,               # Confirm track after 3 detections
    iou_threshold=0.3,        # Used by association function
    per_class=False,          # Traffic usually mixed; enable only if needed
    nr_classes=80,            # Only relevant if per_class=True
    asso_func="iou",          # Fast and sufficient
    is_obb=False,

    min_conf=0.1,             # Drop very weak detections
    track_thresh=0.6,         # High-confidence detections (1st association)
    match_thresh=0.8,         # IoU distance threshold for matching
    frame_rate=25             # Set to your video FPS
)

if __name__ == "__main__":
    infer_times = []
    tracking_times = []

    for i, (imgs, input_tensor, _, _) in enumerate(tqdm(dataloader)):
        input_tensor = input_tensor.to(device)
        infer_time, outputs = model.infer(input_tensor)

        num = int(outputs["num_dets"][0].item())
        boxes = outputs["det_boxes"][0][:num].cpu().numpy()
        scores = outputs["det_scores"][0][:num].cpu().numpy()
        classes = outputs["det_classes"][0][:num].cpu().numpy()
        dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

        frame = imgs[0]
        start = time.perf_counter()
        results = tracker.update(dets, frame)
        end = time.perf_counter()
        tracking_time = end - start

        infer_times.append(infer_time)
        tracking_times.append(tracking_time)

    cleanup()
    logger.info("Resources cleaned up.")

    avg_infer_time = sum(infer_times) / len(infer_times)
    avg_tracking_time = sum(tracking_times) / len(tracking_times)
    logger.info(f"Average inference time per frame: {avg_infer_time:.4f} s")
    logger.info(f"Average tracking time per frame: {avg_tracking_time:.4f} s")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(infer_times, label="Inference Time (s)")
    ax.plot(tracking_times, label="Tracking Time (s)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (s)")
    ax.set_title("Inference and Tracking Time per Frame")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("trt_pipeline/tracking_times.png")