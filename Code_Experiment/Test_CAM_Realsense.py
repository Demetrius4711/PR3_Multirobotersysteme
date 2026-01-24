import os
import sys
import time
from pathlib import Path
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import numpy as np
import cv2
import statistics




pipeline = rs.pipeline() # Create a pipeline
pipeline.start() # Start streaming

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        rgb_image = frames.get_color_frame()

        cv2_Image = np.asanyarray(rgb_image.get_data())

        image_stack = np.hstack((cv2_Image))

        cv2.namedWindow('Realsense Window', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realsense Window', image_stack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        
finally:
    pipeline.stop() # Stop streaming

