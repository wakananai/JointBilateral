import cv2
import numpy as np
#has guassian function in
from scipy import signal

img = cv2.imread('./test2.png', cv2.IMREAD_COLOR);
noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR);
flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR);


blurredImg = cv2.bilateralFilter(img,5,100,100)

#makes the dimensions of our mask in range for corners
def MakeInRange(img,low_x,low_y,high_x,high_y):
	inRange = False
	while not inRange:
		low_x_inRange = True
		try:
			p = img[low_x]
		except:
			low_x +=1
			low_x_inRange = False
		low_y_inRange = True
		try:
			p = img[low_x,low_y]
		except:
			low_y +=1
			low_y_inRange = False
		high_x_inRange = True
		try:
			p = img[high_x]
		except:
			high_x +=1
			high_x_inRange = False
		high_y_inRange = True
		try:
			p = img[low_x,high_y]
		except:
			high_y +=1
			high_y_inRange = False
		if low_y_inRange and high_y_inRange and low_x_inRange and high_x_inRange:
			inRange = True		

#guassian function based on color difference
#for one pixel
def colorGaussian(img,d,i,j,stdev):	
	#don't use indexes that are out of range
	#find ranges of mask
	if d % 2 == 0:
		low_x = i - (d//2-1)
		low_y = j - (d//2-1)

	else:
		low_x = i - (d//2)
		low_y = j - (d//2)
	high_x = i + (d//2)
	high_y = j + (d//2)

	inRange = False
	MakeInRange(img,low_x,low_y,high_x,high_y)
	total = 0
	newPixel = []
	for a in range(low_x,high_x + 1):
		for b in range(low_y,high_y +1):
			for c in range(3)
				signal.guassian(img)

#guassian function based on distance
def distanceBlur(img,d,i,j,stdev)

def jointBilateral(flash,noFlash,d):
	#loop through flash image
	for i in range(flash.shape[0]):
		for j in range(flash.shape[1])

	print(flash)


	img = cv2.bilateralFilter(flash,5,100,100)
	#loop through indexes
	return img

def main()
	jointImg = jointBilateral(flashImg,noFlashImg)
	if not blurredImg is None:

	    cv2.namedWindow('joint bilateral filter');

	    # set a loop control flag

	    keep_processing = True;

	    while (keep_processing):
	        cv2.imshow('joint bilateral filter', jointImg);
	        key = cv2.waitKey(40) & 0xFF; #

	        if (key == ord('x')):
	            keep_processing = False;

	else:
	    print("No image file successfully loaded.");

	# ... and finally close all windows

	cv2.destroyAllWindows();

