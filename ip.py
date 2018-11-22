import cv2
import numpy as np
import math
import os
sqrt2pi = math.sqrt(2 * math.pi )

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
	divisor = stdev * sqrt2pi
	result = math.exp(power)/divisor

	return result

#the joint bilateral filter for a pixel
def jointBilateralPix(flash,noFlash,d,x,y,gaussianColorDict,gaussianDistDict):
	#find ranges of mask
	if d % 2 == 0:
		low_x = x - (d//2)
		low_y = y - (d//2)
	else:
		low_x = x - (d//2 +1)
		low_y = y - (d//2 +1)
	high_x = x + (d//2)
	high_y = y + (d//2)
	low_x,low_y,high_x,high_y  = MakeInRange(flash,low_x,low_y,high_x,high_y)
	#find the sum of the gaussian functions
	total= np.zeros(3) 
	totalDivisor= np.zeros(3)
	colour = flash.item(x,y,c)
	for a in range(low_x,high_x + 1):
		for b in range(low_y,high_y +1):
			distanceGauss = gaussianDistDict[math.sqrt((x-a)**2 + (y-b)**2)] 
			for c in range(3): # do this for each color
				colorGauss = gaussianColorDict[abs(colour - flash.item(a,b,c))]
				total[c] += (colorGauss * distanceGauss * noFlash.item(a,b,c))
				totalDivisor[c] += colorGauss * distanceGauss 
	return total/totalDivisor

def jointBilateral(flash,noFlash,d,stdevC,stdevD):
	#precompute guassian values in dict
	gaussianColorDict = {}
	for i in range(256):
		gaussianColorDict[i] = gaussian(i,stdevC)
	gaussianDistDict = {}
	for i in range(d//2 + 2):
		for j in range(d//2 + 2):
			pythag = math.sqrt(i**2 + j**2)
			gaussianDistDict[pythag] = gaussian(pythag,stdevD)
	newImg = np.zeros((flash.shape[0],flash.shape[1],3),dtype = int)
	for i in range(flash.shape[0]):
		for j in range(flash.shape[1]):
			newImg[i,j] = jointBilateralPix(flash,noFlash,d,i,j,gaussianColorDict,gaussianDistDict)
	result = np.asarray(newImg, dtype=np.uint8)	

	return result

def main():
	imgname = 'meme.jpg' 
	img = cv2.imread('./' + imgname, cv2.IMREAD_COLOR);
	#noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR);
	#flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR);
	#jointImg = jointBilateral(flashImg,noFlashImg)
	stdev1 =100
	stdev2 = 100
	diam = 9
	#filteredImg = jointBilateral(flashImg,noFlashImg,diam,stdev1,stdev2)
	filteredImg = cv2.bilateralFilter(img,diam,stdev1,stdev2)
	name = imgname + str(diam) + '_' + str(stdev1) + '_' + str(stdev2) +'.png'
	path = 'C:/Users/rowan/Documents/uni_year_2/image processing/coursework'
	cv2.imwrite(os.path.join(path,name),filteredImg)
	if not filteredImg is None:
		cv2.namedWindow('bilateral filter')
    # set a loop control flag
		keep_processing = True
		while (keep_processing):
			cv2.imshow('bilateral filter', filteredImg)
			key = cv2.waitKey(40) & 0xFF #
			if (key == ord('x')):
				keep_processing = False
	else:
		print("No image file successfully loaded.")

# ... and finally close all windows

cv2.destroyAllWindows();


main()