"""
Real-Time Webcam Detection - YOLOv8
"""
import os
import time
from collections import deque

import cv2
import numpy as np

import config
import utils


# =====================================
# INIT
# =====================================
config.create_dirs()


class WebcamDetector:

    def __init__(self, camera_id=0, model_path=None):
        self.camera_id = camera_id

        # load model
        self.model = utils.load_model(model_path)

        # warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

        # camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps_hist = deque(maxlen=30)
        self.last_time = time.time()

        self.total_inf = 0
        self.total_det = 0

    def _fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        fps = 1/dt if dt > 0 else 0
        self.fps_hist.append(fps)

        return np.mean(self.fps_hist)

    def run(self, record=False):
        writer = None
        recording = False

        if record:
            path = os.path.join(
                config.INFERENCE_OUTPUT_DIR,
                f"webcam_{utils.get_timestamp_string()}.avi"
            )
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"XVID"),
                30,
                (self.frame_width, self.frame_height)
            )

            if not writer.isOpened():
                writer = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timer = utils.PerformanceTimer()
            timer.start("inf")

            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

            timer.end("inf")

            dets = utils.parse_yolo_results(results)
            t = timer.get_elapsed("inf")

            self.total_inf += t
            self.total_det += len(dets)

            fps = self._fps()

            out = utils.draw_detections_on_image(frame, dets)

            cv2.putText(out, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if writer and recording:
                writer.write(out)

            cv2.imshow("Detection", out)

            k = cv2.waitKey(1) & 0xFF

            if k in (27, ord('q')):
                break
            elif k == ord('r') and writer:
                recording = not recording
            elif k == ord('s'):
                p = os.path.join(
                    config.INFERENCE_OUTPUT_DIR,
                    f"shot_{utils.get_timestamp_string()}.jpg"
                )
                cv2.imwrite(p, out)

        self.cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    try:
        WebcamDetector().run(record=True)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()