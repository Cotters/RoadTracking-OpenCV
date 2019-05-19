#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_coords(img, line_params):
	slope, intercept = line_params
	y1 = img.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)//slope)
	x2 = int((y2 - intercept)//slope)
	return np.array([x1,y1,x2,y2])

def average_slope_intercept(img, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		slope, intercept = np.polyfit((x1,x2), (y1,y2), 1)
		left_fit.append((slope, intercept)) if slope < 0 else right_fit.append((slope, intercept))
	left_fit_avg = np.average(left_fit, axis=0)
	right_fit_avg = np.average(right_fit, axis=0)
	left_line = make_coords(img, left_fit_avg)
	right_line = make_coords(img, right_fit_avg)
	return np.array([left_line, right_line])	

def canny(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	return cv2.Canny(blur, 50, 150)

def region_of_interest(img):
	height = img.shape[0]
	polygons = np.array([[(220, height), (1100, height), (550, 250)]])
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, polygons, 255)
	return cv2.bitwise_and(img, mask)

def display_lines(img, lines):
	line_img = np.zeros_like(img)
	if lines is not None:
		for x1, y1, x2, y2 in lines:
			cv2.line(line_img, (x1,y1), (x2,y2), (255, 0, 0), 10)
	return line_img

# Format image
img = cv2.imread('test_image.jpg')
lane_img = np.copy(img)
canny_img = canny(lane_img)
cropped_img = region_of_interest(canny_img)

# Create a single line for each side of the road, averaged from the HoughLines algorithm
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=1)
avg_lines = average_slope_intercept(lane_img, lines)

# Display the lines on top of our coloured image
line_img = display_lines(img, avg_lines)
combo_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

# Present image in a window - top-right of your monitor.
cv2.imshow('result',combo_img)
cv2.moveWindow('result', 0, 0)
cv2.waitKey()
