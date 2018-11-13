import cv2
import numpy as np
import math
import os


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
		low_x = x - (d//2)
		low_y = y - (d//2)

	else:
		low_x = x - (d//2 +1)
		low_y = y - (d//2 +1)
	high_x = x + (d//2)
	high_y = y + (d//2)
	low_x,low_y,high_x,high_y  = MakeInRange(img,low_x,low_y,high_x,high_y)

	#find the sum of the gaussian functions
	total= np.zeros(3)
	totalDivisor= np.zeros(3)
	#print('range: ' + str(high_x - low_x))
	#print('range: ' + str(high_y - low_y))

	for a in range(low_x,high_x + 1):
		for b in range(low_y,high_y +1):
			for c in range(3): # do this for each color
				colorGauss = gaussian(abs(img.item(x,y,c) - img.item(a,b,c)),stdevC)
				distanceGauss = gaussian(math.sqrt((x-a)**2 + (y-b)**2),stdevD) 
				total[c] += (colorGauss * distanceGauss * img[a,b,c])
				totalDivisor[c] += colorGauss * distanceGauss 
	return total/totalDivisor

#the joint bilateral filter for a pixel
def jointBilateralPix(flash,noFlash,d,x,y,gaussianColorDict,gaussianDistDict):
	#don't use indexes that are out of range
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
	#print('range: ' + str(high_x - low_x))
	#print('range: ' + str(high_y - low_y))
	for a in range(low_x,high_x + 1):
		for b in range(low_y,high_y +1):
			for c in range(3): # do this for each color
				colorGauss = gaussianColorDict[abs(flash.item(x,y,c) - flash.item(a,b,c))]
				distanceGauss = gaussianDistDict[math.sqrt((x-a)**2 + (y-b)**2)] 
				total[c] += (colorGauss * distanceGauss * noFlash[a,b,c])
				totalDivisor[c] += colorGauss * distanceGauss 
	return total/totalDivisor	

#full bilateral filter
def bilateral(img,d,stdevC,stdevD):
	newImg = np.zeros((img.shape[0],img.shape[1],3),dtype = int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			newImg[i,j] = bilateralPix(img,d,i,j,stdevC,stdevD)
	result = np.asarray(newImg, dtype=np.uint8)	

	return result

def bilateralInbuilt(img,d,stdevC,stdevD):
	result = cv2.bilateralFilter(img,d,stdevC,stdevD)
	print(img)
	print('res')
	print(result)	
	return result

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
	img = cv2.imread('./meme.jpg', cv2.IMREAD_COLOR);
	noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR);
	flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR);
	#jointImg = jointBilateral(flashImg,noFlashImg)
	stdev1 = 10
	stdev2 = 10
	filteredImg = jointBilateral(flashImg,noFlashImg,5,stdev1,stdev2)
	name = 'filteredimg' + str(stdev1) + ' ' + str(stdev2) +'.png'
	path = 'C:/Users/rowan/Documents/uni_year_2/image processing/coursework'
	cv2.imwrite(os.path.join(path,name),filteredImg)
	if not filteredImg is None:
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