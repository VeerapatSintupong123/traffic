import argparse
import os
import sys

# Setup path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from trt_pipeline.tools import get_logger

logger = get_logger("RunPipeline")

def resolve_config_path(config_arg, root_dir, logger):
    """
    Resolve config file path with multiple fallbacks.
    Tries: exact path -> add .json -> root_dir/config/ -> ./config/
    """
    config_arg = str(config_arg)
    logger.info(f"Searching for config: {config_arg} with root_dir: {root_dir}")
    
    # Try 1: Use as-is (full path or relative)
    if os.path.exists(config_arg):
        logger.info(f"Config file found at provided path: {config_arg}")
        return os.path.abspath(config_arg)

    # Try 2: Add .json extension
    if os.path.exists(f"{config_arg}.json"):
        logger.info(f"Config file found with .json extension: {config_arg}.json")
        return os.path.abspath(f"{config_arg}.json")
    
    # Try 3: Look in root_dir/config/
    config_in_root = os.path.join(root_dir, "config", f"{config_arg}.json")
    logger.info(f"Trying: {config_in_root}")
    if os.path.exists(config_in_root):
        logger.info(f"✓ Config file found in root_dir/config/: {config_in_root}")
        return config_in_root
    
    # Try 4: Look in ./config/ (current working directory)
    config_in_cwd = os.path.join("config", f"{config_arg}.json")
    logger.info(f"Trying: {config_in_cwd}")
    if os.path.exists(config_in_cwd):
        logger.info(f"✓ Config file found in current working directory config/: {config_in_cwd}")
        return os.path.abspath(config_in_cwd)
    
    logger.error(f"Config file NOT FOUND! Checked: {config_in_root}, {config_in_cwd}")
    return config_in_root

def resolve_engine_path(engine_arg, root_dir, logger):
    """
    Resolve engine file path with multiple fallbacks.
    Tries: exact path -> root_dir/models/ -> ./models/
    """
    engine_arg = str(engine_arg)
    logger.info(f"Searching for engine: {engine_arg}")
    
    # Try 1: Use as-is (full path or relative to cwd)
    if os.path.exists(engine_arg):
        logger.info(f"Engine file found at provided path: {engine_arg}")
        return os.path.abspath(engine_arg)
    
    # Try 2: Look in root_dir/models/
    engine_in_root = os.path.join(root_dir, "models", engine_arg)
    logger.info(f"Trying: {engine_in_root}")
    if os.path.exists(engine_in_root):
        logger.info(f"✓ Engine file found in root_dir/models/: {engine_in_root}")
        return engine_in_root
    
    # Try 3: Look in ./models/ (current working directory)
    engine_in_cwd = os.path.join("models", engine_arg)
    logger.info(f"Trying: {engine_in_cwd}")
    if os.path.exists(engine_in_cwd):
        logger.info(f"✓ Engine file found in current working directory models/: {engine_in_cwd}")
        return os.path.abspath(engine_in_cwd)
    
    logger.error(f"Engine file NOT FOUND! Checked: {engine_in_root}, {engine_in_cwd}")
    raise FileNotFoundError(f"Engine file not found: {engine_arg}")

def main():
    # Get script location and work backwards to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))  # JETSON/src
    jetson_dir = os.path.dirname(script_dir)  # JETSON
    root_dir = os.path.dirname(jetson_dir)  # traffic (project root)
    
    logger.info(f"Root directory resolved to: {root_dir}")

    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Config filename (e.g., 'config_south_jetson2')")
    parser.add_argument("--engine", default="yolov7-tiny.engine", help="Path to TensorRT engine")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    parser.add_argument("--pipeline-version", type=int, default=2, help="Pipeline version to run (default: 2)")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, root_dir, logger)
    engine_path = resolve_engine_path(args.engine, root_dir, logger)

    logger.info(f"Using config file: {config_path}")
    logger.info(f"Using engine file: {engine_path}")

    if args.pipeline_version == 1:
        from pipeline import Pipeline
        pipeline = Pipeline(
            config_path=config_path,
            engine_path=engine_path,
            save_crop=args.save_crop,
            root_dir=root_dir,
        )
        pipeline.run()
    elif args.pipeline_version == 2:
        from pipelinev2 import PipelineV2
        pipeline = PipelineV2(
            config_path=config_path,
            engine_path=engine_path,
            save_crop=args.save_crop,
            root_dir=root_dir,
        )
        pipeline.run()
    else:
        logger.error(f"Invalid pipeline version: {args.pipeline_version}. Must be 1 or 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
