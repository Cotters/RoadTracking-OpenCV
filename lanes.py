#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from processor import Processor

processor = Processor()

# Format image
img = cv2.imread('test_image.jpg')
lane_img = np.copy(img)
canny_img = processor.canny(lane_img)
cropped_img = processor.region_of_interest(canny_img)

# Create a single line for each side of the road, averaged from the HoughLines algorithm
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=1)
avg_lines = processor.average_slope_intercept(lane_img, lines)

# Display the lines on top of our coloured image
line_img = processor.display_lines(img, avg_lines)
combo_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

# Present image in a window - top-right of your monitor.
cv2.imshow('result',combo_img)
cv2.moveWindow('result', 0, 0)
cv2.waitKey()
