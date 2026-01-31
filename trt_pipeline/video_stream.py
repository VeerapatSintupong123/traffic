import cv2 as cv
import threading
import queue
import cv2 as cv
import numpy as np

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

class AsyncImageSaver:
    def __init__(self, queue_size=256):
        self.q = queue.Queue(maxsize=queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running or not self.q.empty():
            try:
                path, image = self.q.get(timeout=0.1)
                cv.imwrite(path, image, [cv.IMWRITE_JPEG_QUALITY, 30])
            except queue.Empty:
                continue

    def save(self, path, image):
        if self.running:
            self.q.put((path, image))

    def stop(self):
        self.running = False
        self.thread.join()

class VideoStream:
    def __init__(self, video_path, skip=1, queue_size=2):
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")

        self.skip = max(1, skip)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        idx = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break

            if idx % self.skip == 0:
                self.queue.put((idx, frame))
            idx += 1

        self.cap.release()

    def read(self):
        if self.stopped and self.queue.empty():
            return None
        return self.queue.get()

    def stop(self):
        self.stopped = True