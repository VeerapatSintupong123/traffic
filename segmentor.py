from moviepy.editor import VideoFileClip
import cv2 as cv
import numpy as np

class RoadSegmenter:
    def __init__(self, video_path: str, time_in_seconds: int):
        self.coordinates = []
        self.video = VideoFileClip(video_path)
        self.time_in_seconds = time_in_seconds
        self.frame = self.get_frame_at_time()
        if self.frame is None:
            raise ValueError("Unable to get the frame from the video.")
        self.gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

    def get_frame_at_time(self) -> np.ndarray:
        frame = self.video.get_frame(self.time_in_seconds)
        return frame

    def get_coordinates(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))

    def segment(self) -> None:
        cv.namedWindow("Image")
        cv.setMouseCallback("Image", self.get_coordinates)

        while True:
            frame_with_dots = self.gray_frame.copy()

            # Draw the points on the frame
            for (x, y) in self.coordinates:
                cv.circle(frame_with_dots, (x, y), 5, (255, 0, 0), -1)

            # Display a message for user guidance
            cv.putText(frame_with_dots, "Click to select coordinates", (10, 20), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame_with_dots, "Press 'r' to reset, 'q' to quit", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            cv.imshow("Image", frame_with_dots)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.coordinates = []

        cv.destroyAllWindows()