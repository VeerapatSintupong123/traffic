import argparse
from pipeline import Pipeline
from pathlib import Path
import os

def main():
    root_dir = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Config JSON file name (without .json extension)")
    parser.add_argument("--engine", default=os.path.join(root_dir, "models", "yolov7-tiny.engine"), help="Path to TensorRT engine")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    args = parser.parse_args()

    pipeline = Pipeline(
        config_path=os.path.join(root_dir, "config", f"{args.config}.json"),
        engine_path=args.engine,
        save_crop=args.save_crop,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
