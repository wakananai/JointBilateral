import cv2
import numpy as np
import math


#makes the dimensions of our mask in range for corners
def MakeInRange(img,low_x,low_y,high_x,high_y):
	if low_x < 0:
		low_x = 0
	if low_y < 0 :
		low_y = 0
	if high_x > (img.shape[0] -1):
		high_x = img.shape[0] -1

	if high_y > img.shape[1] - 1:
		high_y = img.shape[1] -1
	return low_x,low_y,high_x,high_y
#the actual gaussian function
def gaussian(number,stdev):
	power = -0.5 * ((number/stdev)**2)
	divisor = stdev * math.sqrt(2 * math.pi )
	result = math.exp(power)/divisor
	return result
#for one pixel
def bilateralPix(img,d,x,y,stdevC,stdevD):	
	#don't use indexes that are out of range
	#find ranges of mask
	if d % 2 == 0:
		low_x = x - (d//2-1)
		low_y = y - (d//2-1)

	else:
		low_x = x - (d//2)
		low_y = y - (d//2)
	high_x = x + (d//2)
	high_y = y + (d//2)
	low_x,low_y,high_x,high_y  = MakeInRange(img,low_x,low_y,high_x,high_y)

	#find the sum of the gaussian functions
	total= np.zeros(3)
	totalDivisor= np.zeros(3)

	newPixel = []
	for a in range(low_x,high_x + 1):
		for b in range(low_y,high_y +1):
			for c in range(3): # do this for each color
				colorGauss = gaussian(abs(img[x,y,c] - img[a,b,c]),stdevC)
				distanceGauss = gaussian(math.sqrt((x-a)**2 + (y-b)**2),stdevD)    #
				total[c] += colorGauss * distanceGauss * img[x,y,c]
				totalDivisor[c] += colorGauss * distanceGauss
	img[x,y] = total/totalDivisor

#full bilateral filter
def bilateral(img,d,stdevC,stdevD):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			bilateralPix(img,d,i,j,stdevC,stdevD)
		print('col done ' + str(i))
	return img

def jointBilateral(flash,noFlash,d):
	#loop through flash image
	for i in range(flash.shape[0]):
		for j in range(flash.shape[1]):
			pass
	print(flash)


	img = cv2.bilateralFilter(flash,5,100,100)
	#loop through indexes
	return img

def main():
	img = cv2.imread('./test2.png', cv2.IMREAD_COLOR);
	noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR);
	flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR);
	#jointImg = jointBilateral(flashImg,noFlashImg)
	filteredImg = bilateral(img,5,100,100)
	if not blurredImg is None:
	    cv2.namedWindow('bilateral filter');
	    # set a loop control flag
	    keep_processing = True;
	    while (keep_processing):
	        cv2.imshow('bilateral filter', filteredImg);
	        key = cv2.waitKey(40) & 0xFF; #

	        if (key == ord('x')):
	            keep_processing = False;

	else:
	    print("No image file successfully loaded.");

	# ... and finally close all windows

	cv2.destroyAllWindows();

main()