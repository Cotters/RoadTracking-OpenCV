#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from processor import Processor

processor = Processor()

cap = cv2.VideoCapture("test.mp4")
while(cap.isOpened()):
	_, frame = cap.read()
	canny_img = processor.canny(frame)
	cropped_img = processor.region_of_interest(canny_img)
	lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=1)
	avg_lines = processor.average_slope_intercept(frame, lines)

	# Display the lines on top of our coloured image
	line_img = processor.display_lines(frame, avg_lines)
	combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

	# Present image in a window - top-right of your monitor.
	cv2.imshow('result',combo_img)
	cv2.moveWindow('result', 0, 0)
	cv2.waitKey(1)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
