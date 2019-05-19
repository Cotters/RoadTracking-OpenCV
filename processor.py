import cv2
import numpy as np

class Processor(object):
	#def __init__(img):
		#self.img = img

	def make_coords(self, img, line_params):
		slope, intercept = line_params
		y1 = img.shape[0]
		y2 = int(y1*(3/5))
		x1 = int((y1 - intercept)//slope)
		x2 = int((y2 - intercept)//slope)
		return np.array([x1,y1,x2,y2])

	def average_slope_intercept(self, img, lines):
		left_fit = []
		right_fit = []
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			slope, intercept = np.polyfit((x1,x2), (y1,y2), 1)
			left_fit.append((slope, intercept)) if slope < 0 else right_fit.append((slope, intercept))
		left_fit_avg = np.average(left_fit, axis=0)
		right_fit_avg = np.average(right_fit, axis=0)
		left_line = self.make_coords(img, left_fit_avg)
		right_line = self.make_coords(img, right_fit_avg)
		return np.array([left_line, right_line])	

	def canny(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		blur = cv2.GaussianBlur(gray, (5,5), 0)
		return cv2.Canny(blur, 50, 150)

	def region_of_interest(self, img):
		height = img.shape[0]
		polygons = np.array([[(220, height), (1100, height), (550, 250)]])
		mask = np.zeros_like(img)
		cv2.fillPoly(mask, polygons, 255)
		return cv2.bitwise_and(img, mask)

	def display_lines(self, img, lines):
		line_img = np.zeros_like(img)
		if lines is not None:
			for x1, y1, x2, y2 in lines:
					cv2.line(line_img, (x1,y1), (x2,y2), (255, 0, 0), 10)
		return line_img

