import argparse
from pipeline import Pipeline
from segmentor import RoadSegmenter

def main():
    parser = argparse.ArgumentParser(description="Vehicle Tracking System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment_parser = subparsers.add_parser("segment", help="Run region/line selection")
    segment_parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    segment_parser.add_argument("--n", type=int, required=True, help="Number of iterations")

    tracking_parser = subparsers.add_parser("tracking", help="Run object tracking")
    tracking_parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    if args.command == "segment":
        for _ in range(args.n):
            segmentor = RoadSegmenter(args.video, 0)
            segmentor.segment()
            print(f"Coordinate: {segmentor.coordinates}\n")

    elif args.command == "tracking":
        pipline = Pipeline(args.config)
        pipline.run()

if __name__ == "__main__":
    main()