import argparse
import os
import sys

# Setup path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from trt_pipeline.tools import get_logger
from pipelinev2 import PipelineV2

logger = get_logger("RunPipeline")

def check_file_types(config_name):
    if not config_name.endswith(".json") or len(config_name.split(".json")[0]) == 0:
        logger.error("Config file must be a .json file")
        config_name = f"{config_name}.json"

    return config_name

def main():
    # Get script location and work backwards to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))  # JETSON/src
    jetson_dir = os.path.dirname(script_dir)  # JETSON
    root_dir = os.path.dirname(jetson_dir)  # traffic (project root)
    
    logger.info(f"Root directory resolved to: {root_dir}")

    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Config filename (e.g., 'config_south_jetson2')")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    args = parser.parse_args()

    args.config = check_file_types(args.config)
    
    pipeline = PipelineV2(
        config_name=args.config,
        save_crop=args.save_crop,
        root_dir=root_dir,
    )
    pipeline.run()

if __name__ == "__main__":
    main()
