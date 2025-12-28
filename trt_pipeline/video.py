import cv2
import json
import numpy as np

SKIP = 0

with open("trt_pipeline/lane_data.json", "r") as f:
    lane_data = json.load(f)

frame_to_objs = {}
for lane in lane_data.values():
    for obj in lane["cross_obj"]:
        frame_idx = obj["frame"]
        if frame_idx not in frame_to_objs:
            frame_to_objs[frame_idx] = []
        frame_to_objs[frame_idx].append(obj)

video_path = "video/south_video.avi"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("trt_pipeline/tracked_output.mp4", fourcc, fps, (w, h))

frame_idx = 0
dataset_frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw lanes and lines
    for lane_name, lane_info in lane_data.items():
        # Draw lane polygon
        lane_pts = np.array(lane_info["polygon"], dtype=np.int32)
        cv2.polylines(frame, [lane_pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw lane line
        line_pts = np.array(lane_info["line"], dtype=np.int32)
        cv2.line(frame, tuple(line_pts[0]), tuple(line_pts[1]), color=(0, 0, 255), thickness=2)

    # Only process frames that match the dataset (i.e., not skipped)
    if SKIP == 0 or frame_idx % SKIP != 0:
        for obj in frame_to_objs.get(dataset_frame_idx, []):
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["id"]
            class_id = obj["class_id"]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f'ID:{track_id} C:{class_id}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        dataset_frame_idx += 1

    out.write(frame)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end='\r')

cap.release()
out.release()
print("Tracked video saved as tracked_output.mp4")