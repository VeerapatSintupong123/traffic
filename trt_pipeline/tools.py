import gc
import json
import os
import ast
import logging
import torch
from shapely.geometry import Polygon

def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent adding multiple handlers in interactive or repeated runs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging
    return logger

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()

def initial_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in config file: {config_path}")

    required_keys = ["video", "tracking", "density", "output", "scale", "skip"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    os.makedirs(config["output"], exist_ok=True)

    return config

def initial_lane_data(tracking_zone: list, dict_class: dict):
    return {
        lane["name"]: {
            "polygon": lane.get("polygon"),  # The Polygon object
            "lane": lane.get("lane"),        # The lane coordinates
            "line": lane.get("line"),        # The line coordinates
            "cross_ids": set(),
            "cross_obj": [],
            "count_cls": {key: 0 for key in dict_class.keys()},
        }
        for lane in tracking_zone
    }

def parse_zones(zones):
    parsed_zones = []
    for zone in zones:
        parsed_zone = zone.copy()
        for key in ['lane', 'line']:
            if isinstance(parsed_zone.get(key), str):
                try:
                    parsed_zone[key] = ast.literal_eval(parsed_zone[key])
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse {key} coordinates for {parsed_zone.get('name', 'Unknown zone')}")
                    continue

        if 'lane' in parsed_zone and isinstance(parsed_zone['lane'], list):
            try:
                parsed_zone['polygon'] = Polygon(parsed_zone['lane'])
            except Exception as e:
                print(f"Warning: Could not create Polygon for {parsed_zone.get('name', 'Unknown zone')}: {e}")
                parsed_zone['polygon'] = None
        parsed_zones.append(parsed_zone)
    return parsed_zones

def to_original_coords(x, y, dw, dh, ratio):
    return int((x - dw) / ratio), int((y - dh) / ratio)

def side_of_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def save_lane_data(lane_data, path):
    out = {}
    for lane_name, lane_info in lane_data.items():
        out[lane_name] = {
            "polygon": [list(map(int, pt)) for pt in lane_info["lane"]],
            "line": [list(map(int, pt)) for pt in lane_info["line"]],
            "cross_ids": list(lane_info["cross_ids"]),
            "cross_obj": lane_info["cross_obj"],
            "count_cls": {str(k): int(v) for k, v in lane_info["count_cls"].items()},
        }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

def save_performance_data(config, total, preprocess_times, infer_times, tracking_times):
    with open(os.path.join(config['output'], "performance.json"), "w") as f:
        json.dump({
            "total_time": total,
            "preprocess_time": preprocess_times,
            "inference_time": infer_times,
            "tracking_time": tracking_times
        }, f, indent=2)