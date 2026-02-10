import argparse
from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--engine", default="../../models/yolov7-tiny.engine", help="Path to TensorRT engine")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    args = parser.parse_args()

    pipeline = Pipeline(
        config_path=args.config,
        engine_path=args.engine,
        save_crop=args.save_crop,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
