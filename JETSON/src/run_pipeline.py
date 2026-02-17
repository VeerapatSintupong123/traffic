import argparse
from pipeline import Pipeline
from pathlib import Path
import os

def resolve_config_path(config_arg, root_dir):
    """
    Resolve config file path with multiple fallbacks.
    Tries: exact path -> add .json -> root_dir/config/ -> ./config/
    """
    config_arg = str(config_arg)
    
    # Try 1: Use as-is (full path or relative)
    if os.path.exists(config_arg):
        return os.path.abspath(config_arg)
    
    # Try 2: Add .json extension
    if os.path.exists(f"{config_arg}.json"):
        return os.path.abspath(f"{config_arg}.json")
    
    # Try 3: Look in root_dir/config/
    config_in_root = os.path.join(root_dir, "config", f"{config_arg}.json")
    if os.path.exists(config_in_root):
        return config_in_root
    
    # Try 4: Look in ./config/ (current working directory)
    config_in_cwd = os.path.join("config", f"{config_arg}.json")
    if os.path.exists(config_in_cwd):
        return os.path.abspath(config_in_cwd)
    
    return config_in_root

def main():
    root_dir = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Config filename (e.g., 'config_south_jetson2')")
    parser.add_argument("--engine", default=os.path.join(root_dir, "models", "yolov7-tiny.engine"), help="Path to TensorRT engine")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, str(root_dir))

    pipeline = Pipeline(
        config_path=config_path,
        engine_path=args.engine,
        save_crop=args.save_crop,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
