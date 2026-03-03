from jtop import jtop
import cv2 as cv
import os
import sys

def read_stats(jetson):
    stats = jetson.stats
    log_entry = {
        'time': stats.get('time'),
        'gpu': stats.get('GPU'),
        'ram': stats.get('RAM'),
        'swap': stats.get('SWAP'),
        'iram': stats.get('IRAM'),
        'cpus': stats.get('cpus'),
        'temp': stats.get('Temp'),
        'power': stats.get('power'),
    }
    return log_entry

def main():
    log = {}
    video_path = '/home/schauto/traffic/video/south_video.avi' 
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: CAP COULDN'T OPEN '{video_path}'")
        sys.exit(1)

    stream1 = cv.cuda_Stream()
    start_event = cv.cuda.Event(cv.cuda.EVENT_DISABLE_TIMING)
    end_event = cv.cuda.Event(cv.cuda.EVENT_DISABLE_TIMING)

    try:
        with jtop() as jetson:
            while cap.isOpened():
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
        print("Resources released.")
        print(f"Collected {len(log)} log entries.")

if __name__ == "__main__":
    main()