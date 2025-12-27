import torch
from trt_model import TRTModel
from frames_data import VideoFrameDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from boxmot.trackers import ByteTrack
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from utils import get_logger, cleanup

logger = get_logger("Tracking")

def collate_fn(batch):
    imgs, tensors, ratios, dwdhs = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return imgs, tensors, ratios, dwdhs

if __name__ == "__main__":
    writer = SummaryWriter("runs/tracking6")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TRTModel(
        engine_path="models\yolov7-tiny.engine", 
        input_shape=(1, 3, 640, 640), 
        device=device
    )

    dataset = VideoFrameDataset(
        video_path="video/south_video.avi",
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

    for idx, (imgs, input_tensor, _, _) in enumerate(tqdm(dataloader, desc="Processing frames")):
        input_tensor = input_tensor.to(device)
        infer_time, outputs = model.infer(input_tensor)
        writer.add_scalar("Time/Inference", infer_time, idx)

        num = int(outputs["num_dets"][0].item())
        if num == 0:
            continue

        conf_mask = outputs["det_scores"][0] > 0.3
        dets_gpu = torch.cat(
            [
                outputs["det_boxes"][0][conf_mask],
                outputs["det_scores"][0][conf_mask, None],
                outputs["det_classes"][0][conf_mask, None],
            ],
            dim=1,
        )

        dets = dets_gpu.cpu().numpy().astype(np.float32)

        frame = imgs[0]
        start = time.perf_counter()
        results = tracker.update(dets, frame)
        tracking_time = time.perf_counter() - start

        writer.add_scalar("Time/Tracking", tracking_time, idx)

    writer.close()
    logger.info("Tracking complete.")

    cleanup()
    logger.info("Resources cleaned up.")

# python pipeline/tracking.py
# tensorboard --logdir runs/tracking6 (trackings6 is the folder name given to SummaryWriter)