import cv2
import numpy as np

import threading

class StreamCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video3")
        self.running = False
        self.display_thread = None
        
    def start_stream(self, width, height, fps, pixel_format):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.running = True
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.start()

    def display_loop(self):
        while self.running:
            ret, img = self.cap.read()
            if not ret: 
                continue 
            cv2.imshow('ImageWindow', img) 
            cv2.waitKey(1)

    def wait(self):
        #self.anx.wait() # blocks till interrupt is received
        self.running = False
        self.display_thread.join()
        cv2.destroyAllWindows()
 
def main(): 
    stream_camera = StreamCamera()
    stream_camera.start_stream(fps=30, width=640, height=480, pixel_format=0)
    stream_camera.wait()

if __name__ == "__main__": 
    main()


