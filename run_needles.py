#!/usr/local/bin/python3.11

import cv2
import os, sys
import numpy as np
from scipy import stats
import math

#th = {0: 150, 1: 120, 2: 120, 3:150}
th = {0: 130, 1: 70, 2: 100}

def process_image(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY, 0)
	cv2.imwrite('results/'+str(ip)+'_gray.jpg', gray)

	blurred = cv2.GaussianBlur(gray, (15,15), 6)
	cv2.imwrite('results/'+str(ip)+'_blurred.jpg', blurred)

	ret, thresh = cv2.threshold(blurred, th[ip], 255, cv2.THRESH_BINARY)
	cv2.imwrite('results/'+str(ip)+'_thresh.jpg', thresh)

	canny = cv2.Canny(thresh, 50, 100, 3)
	cv2.imwrite('results/'+str(ip)+'_canny.jpg', canny)

	return canny

class Point:
	def __init__(self, x, xerr, y, yerr):
		self.x = x
		self.xerr = xerr
		self.y = y
		self.yerr = yerr
	
	# Method used to display X and Y coordinates
	# of a point
	def displayPoint(self, p):
		print(f"({p.x}, {p.y})")

def linesIntersection(A, B, C, D):
	# Line AB represented as a1x + b1y = c1
	a1 = B.y - A.y
	b1 = A.x - B.x
	c1 = a1*(A.x) + b1*(A.y)
	a1_err = math.sqrt(B.yerr**2 + A.yerr**2)
	b1_err = math.sqrt(A.xerr**2 + B.xerr**2)
	c1_err = math.sqrt((a1*A.x*math.sqrt((a1_err/a1)**2 + (A.xerr/A.x)**2))**2 + (b1*A.y*math.sqrt((b1_err/b1)**2 + (A.yerr/A.y)**2))**2)
	
	# Line CD represented as a2x + b2y = c2
	a2 = D.y - C.y
	b2 = C.x - D.x
	c2 = a2*(C.x) + b2*(C.y)
	a2_err = math.sqrt(B.yerr**2 + A.yerr**2)
	b2_err = math.sqrt(A.xerr**2 + B.xerr**2)
	c2_err = math.sqrt((a2*C.x*math.sqrt((a2_err/a2)**2 + (C.xerr/C.x)**2))**2 + (b2*C.y*math.sqrt((b2_err/b2)**2 + (C.yerr/C.y)**2))**2)
	
	det = a1*b2 - a2*b1
	det_err = math.sqrt((a1*b2*math.sqrt((a1_err/a1)**2 + (b2_err/b2)**2))**2 + (a2*b1*math.sqrt((a2_err/a2)**2 + (b1_err/b1)**2))**2)
	print(det,det_err)
	
	if (det == 0):
	    # The lines are parallel. This is simplified
	    # by returning a pair of FLT_MAX
	    return Point(10**9, 0, 10**9, 0)
	else:
		x = (b2*c1 - b1*c2)/det
		x_err = x*math.sqrt((math.sqrt((b2*c1*((b2_err/b2)**2 + (c1_err/c1)**2))**2 + (b1*c2*((b1_err/b1)**2 + (c2_err/c2)**2))**2)/(b2*c1 - b1*c2))**2 + (det_err/det)**2)
		y = (a1*c2 - a2*c1)/det
		y_err = y*math.sqrt((math.sqrt((a1*c2*((a1_err/a1)**2 + (c2_err/c2)**2))**2 + (a2*c1*((a2_err/a2)**2 + (c1_err/c1)**2))**2)/(a1*c2 - a2*c1))**2 + (det_err/det)**2)
		return Point(x, x_err, y, y_err)

if __name__ == '__main__':
	#img = cv2.imread("pics/raw_9.030025910684348.jpg")
	img = cv2.imread("pics/raw_5.333333333333333.jpg")
	#img = cv2.imread("pics/raw_1.6089857575911533.jpg")
	ip = 1

	# Initial processing
	processed_img = process_image(img)

	# Find line segments along the edges + sorting in ascending order in x-coordinate of the left edge
	lines = cv2.HoughLinesP(processed_img, 1, np.pi / 360, 50, 0, 0)
	lines = np.squeeze(lines,1)
	lines = lines[lines[:,0].argsort()]

	# Determine what the index of the last left segment is (the "separator" index)
	x1_nolastx1 = lines[:-1,0]
	x1_nofirstx1 = lines[1:,0]
	x1_diff = x1_nofirstx1-x1_nolastx1
	separator = np.argmax(x1_diff)

	# Build x and y arrays to prepare for the regression
	outimg = np.copy(img)
	ireg = 0
	x_reg1 = np.zeros(2*(separator+1))
	y_reg1 = np.zeros(2*(separator+1))
	x_reg2 = np.zeros(2*(len(lines)-separator-1))
	y_reg2 = np.zeros(2*(len(lines)-separator-1))
	for line in lines:
		x1 = 0
		x2 = 0
		y1 = 0
		y2 = 0
		i = 0
		for xi in line:
			if i==0: x1 = xi
			if i==1: y1 = xi
			if i==2: x2 = xi
			if i==3: y2 = xi
			if ireg < (2*separator+1):
				x_reg1[ireg] = x1
				x_reg1[ireg+1] = x2
				y_reg1[ireg] = y1
				y_reg1[ireg+1] = y2
			else:
				x_reg2[ireg-2-(2*separator)] = x1
				x_reg2[ireg-1-(2*separator)] = x2
				y_reg2[ireg-2-(2*separator)] = y1
				y_reg2[ireg-1-(2*separator)] = y2
			i += 1
		ireg += 2
		cv2.line(outimg,(x1, y1),(x2, y2),(255,0,0),5)

	# Apply a regression on the arrays
	res1 = stats.linregress(x_reg1, y_reg1)
	res2 = stats.linregress(x_reg2, y_reg2)

	# Draw the two lines on the original image
	#line1 = cv2.line(outimg,(0, int(res1.intercept)),(img.shape[0], int(res1.intercept+res1.slope*img.shape[0])),(0,0,255),3)
	#line2 = cv2.line(outimg,(0, int(res2.intercept)),(img.shape[0], int(res2.intercept+res2.slope*img.shape[0])),(0,0,255),3)
	line1 = cv2.line(outimg,(1, int(res1.intercept+res1.slope)),(img.shape[0], int(res1.intercept+res1.slope*img.shape[0])),(0,0,255),3)
	line2 = cv2.line(outimg,(1, int(res2.intercept+res1.slope)),(img.shape[0], int(res2.intercept+res2.slope*img.shape[0])),(0,0,255),3)

	# Find intersection from line 1 (points A,B) and line 2 (points C,D)
	A = Point(1,0,res1.intercept+res1.slope,res1.intercept_stderr)
	B = Point(img.shape[0],0,res1.intercept+res1.slope*img.shape[0],math.sqrt((res1.intercept_stderr**2)+((res1.stderr**2)*img.shape[0])))
	C = Point(1,0,res2.intercept+res2.slope,res2.intercept_stderr)
	D = Point(img.shape[0],0,res2.intercept+res2.slope*img.shape[0],math.sqrt((res2.intercept_stderr**2)+((res2.stderr**2)*img.shape[0])))
	xl = int(linesIntersection(A, B, C, D).x)
	xl_err = linesIntersection(A, B, C, D).xerr
	yl = int(linesIntersection(A, B, C, D).y)
	yl_err = linesIntersection(A, B, C, D).yerr
	print(xl,xl_err,yl,yl_err)
	cv2.circle(outimg, (xl, yl), 2, (0, 0, 255), 20)
	cv2.putText(outimg, '('+str(int(xl))+','+str(int(yl))+')', (xl+100, yl+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

	cv2.imwrite('results/canvasOutput.jpg', outimg)
