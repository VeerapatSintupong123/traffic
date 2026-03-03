from jtop import jtop
import cv2 as cv
import os
import sys
import time
import json

def read_stats(jetson):
    stats = jetson.stats
    # print(stats)
    log_entry = {
        'time': str(stats.get('time')),
        'gpu': stats.get('GPU'),
        'ram': stats.get('RAM'),
        'swap': stats.get('SWAP'),
        'iram': stats.get('IRAM'),
        'cpus': [
            stats.get('CPU1'),
            stats.get('CPU2'),
            stats.get('CPU3'),
            stats.get('CPU4'),
        ],
        'temp': {
            'AO': stats.get('Temp AO'),
            'CPU': stats.get('Temp CPU'),
            'GPU': stats.get('Temp GPU'),
            'PLL': stats.get('Temp PLL'),
            'thermal': stats.get('Temp thermal'),
        },
        'power': {
            'CPU': stats.get('Power POM_5V_CPU'),
            'GPU': stats.get('Power POM_5V_GPU'),
            'total': stats.get('Power TOT'),
        }
    }
    return log_entry

def main():
    log = {}
    log_file_path = 'jetson_stats.json'
    video_path = '/home/schauto/traffic/video/south_video.avi' 
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: CAP COULDN'T OPEN '{video_path}'")
        sys.exit(1)

    stream1 = cv.cuda_Stream()
    start_event = cv.cuda.Event()
    end_event = cv.cuda.Event()

    try:
        with jtop() as jetson:
            while cap.isOpened() and jetson.ok():
                # Start of the loop, read stats
                log_entry = read_stats(jetson)
                log[len(log)] = log_entry

                ret, frame = cap.read()
                if not ret:
                    print("End of video stream.")
                    break

                # Start timer
                start_event.record(stream1)

                # Upload and process frame
                d_frame1 = cv.cuda_GpuMat()
                d_frame1.upload(frame, stream1)
                d_gray1 = cv.cuda.cvtColor(d_frame1, cv.COLOR_BGR2RGB, stream=stream1)

                # End timer and synchronize
                end_event.record(stream1)
                stream1.waitForCompletion()
                
                # Calculate and print elapsed time
                elapsed_time_ms = cv.cuda.Event_elapsedTime(start_event, end_event)
                print(f"GPU processing time: {elapsed_time_ms:.2f} ms")


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        
        with open(log_file_path, 'w') as f:
            json.dump(log, f, indent=4)

        print("Resources released.")
        print(f"Collected {len(log)} log entries and saved to '{log_file_path}'")

if __name__ == "__main__":
    main()