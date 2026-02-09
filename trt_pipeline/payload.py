import json
from datetime import datetime
from zoneinfo import ZoneInfo

# lanes structure
# {
#     "laneId": "N1",
#     "direction": "N",
#     "vehicles": {
#     "density": {
#         // pressure at current frame
#         scores: 0.77,    // [0, 1]
#         level: 4         // [1, 2, 3, 4, 5]
#     },
#     "confident": {
#         // every max frames (50 frames -> 2 seconds)
#         "cars": 0.8,     // [0, 1] mean
#         "motorbike": 0.7 // [0, 1] mean
#     }
#     "moving": {
#         // every max frames (50 frames -> 2 seconds)
#         "cars": 0,       // count
#         "motorbike": 0   // count
#     }
#     }
# },

class LanePayload:
    def __init__(self, lane_id, direction, max_frames=50):
        self.lane_id = lane_id
        self.direction = direction
        self.max_frames = max_frames

        self.reset_window()

        self.density = {"score": 0.0, "level": 0}

    def update_density(self, score, level):
        self.density["score"] = float(score)
        self.density["level"] = int(level)

    def append_confident(self, cars, motorbike):
        self._conf_cars.append(cars)
        self._conf_motorbike.append(motorbike)
        self._frame_counter += 1

    def add_moving(self, cars, motorbike):
        self._moving_cars += cars
        self._moving_motorbike += motorbike

    def ready(self):
        return self._frame_counter >= self.max_frames

    def finalize_window(self):
        def mean(x):
            return sum(x) / len(x) if x else 0.0

        confident = {
            "cars": mean(self._conf_cars),
            "motorbike": mean(self._conf_motorbike),
        }

        moving = {
            "cars": self._moving_cars,
            "motorbike": self._moving_motorbike,
        }

        self.reset_window()

        return {
            "laneId": self.lane_id,
            "direction": self.direction,
            "vehicles": {
                "density": self.density,
                "confident": confident,
                "moving": moving,
            },
        }

    def reset_window(self):
        self._conf_cars = []
        self._conf_motorbike = []
        self._moving_cars = 0
        self._moving_motorbike = 0
        self._frame_counter = 0

class Payload:
    def __init__(self, intersection_id, camera_id):
        self.intersection_id = intersection_id
        self.camera_id = camera_id
        self.meta = {}
        self.lanes = {}

    def get_lane(self, lane_id, direction):
        if lane_id not in self.lanes:
            self.lanes[lane_id] = LanePayload(lane_id, direction)
        return self.lanes[lane_id]

    def set_meta(self, frame_id):
        self.meta = {"frameId": frame_id}

    def collect_ready_lanes(self):
        ready = []
        for lane in self.lanes.values():
            if lane.ready():
                ready.append(lane.finalize_window())
        return ready

    def build(self):
        lanes = self.collect_ready_lanes()

        if not lanes:
            return None

        return {
            "intersectionId": self.intersection_id,
            "cameraId": self.camera_id,
            "timestamp": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="milliseconds"),
            "meta": self.meta,
            "lanes": lanes,
        }

    def build_json(self):
        data = self.build()
        return json.dumps(data, indent=2) if data else None