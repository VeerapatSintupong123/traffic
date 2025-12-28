from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm
from tools import get_logger
import time

logger = get_logger("VideoFrameDataset")

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

class VideoFrameDataset(Dataset):
    def __init__(self, video_path, skip, device):
        self.frames = []
        self.device = device
        self.video_path = video_path
        self.skip = skip

        logger.info(f"Opening video: {video_path}")
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames reported: {total}")

        for i, _ in enumerate(tqdm(range(total), desc="Reading video frames")):
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed before expected end")
                break

            if self.skip > 1 and i % self.skip != 0:
                continue
            self.frames.append(frame)

        cap.release()
        logger.info(f"Loaded {len(self.frames)} frames into memory")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        start_time = time.perf_counter()
        img_bgr = self.frames[idx]
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        img_lb, ratio, dwdh = letterbox(
            img_rgb,
            new_shape=(640, 640),
            auto=False,
        )

        img_chw = img_lb.transpose(2, 0, 1)
        img_chw = np.expand_dims(img_chw, axis=0)
        img_chw = np.ascontiguousarray(img_chw, dtype=np.float32)

        tensor = torch.from_numpy(img_chw) / 255.0
        tensor = tensor.squeeze(0)
        end_time = time.perf_counter()
        process_time = end_time - start_time

        return process_time, img_bgr, tensor, ratio, dwdh
    
# frames_dataset = VideoFrameDataset("video\south_1-out.avi", device)
# dataloader = DataLoader(frames_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)