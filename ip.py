import cv2
import numpy as np
import math
import os
sqrt2pi = math.sqrt(2 * math.pi )

#makes the dimensions of our mask in range for corners
def MakeInRange(img,low_x,low_y,high_x,high_y):
	add_x=add_y=cut_x=cut_y = 0
	if low_x < 0:
		add_x = abs(low_x)
		low_x = 0
	if low_y < 0:
		add_y = abs(low_y)
		low_y = 0

	if high_x > (img.shape[0] -1):
		cut_x = high_x - (img.shape[0] -1) 
		high_x = img.shape[0] -1
	if high_y > img.shape[1] - 1:
		cut_y = high_y - (img.shape[1] -1) 
		high_y = img.shape[1] -1

	return add_x,add_y,cut_x,cut_y,low_x,low_y,high_x,high_y
	
#the actual gaussian function
def gaussian(mask,stdev):
	power = -0.5 * (np.square(mask/stdev))
	divisor = stdev * sqrt2pi
	result = np.exp(power)/divisor
	return result

#2d array of gaussian weights for distances
def get2dKernel2(d,stdev):
	k=  cv2.getGaussianKernel(d,stdev)
	x = np.stack((k,)*3, axis=-1)
	y = x.reshape((1,d,3))
	b = np.ones((d,d,3))
	return x * y * b

#2d array of gaussian weights for distances
def get2dKernel(d,stdev):
	array = np.ones((d,d,3))
	for i in range(d):
		for j in range(d):
			for c in range(3):
				array[i,j,c] = math.sqrt((d//2 -i)**2 + (d//2 -j)**2)
	return array

	
#the joint bilateral filter for a pixel
def jointBilateralPix(flash,noFlash,d,x,y,stdevC,stdevD,distanceKernel):
	#find ranges of mask
	if d % 2 == 0:
		low_x = x - (d//2)
		low_y = y - (d//2)
	else:
		low_x = x - (d//2 +1)
		low_y = y - (d//2 +1)
	high_x = x + (d//2)
	high_y = y + (d//2)
	add_x,add_y,cut_x,cut_y,low_x,low_y,high_x,high_y  = MakeInRange(flash,low_x,low_y,high_x,high_y)
	#find the sum of the gaussian functions
	total= np.zeros(3) 
	totalDivisor= np.zeros(3)
	colorGauss = np.zeros(3)
	colour = flash[x,y]
 	#added 
	# get gaussian masks
	colourMask = gaussian(flash[low_x:high_x,low_y:high_y] - colour,stdevC)
	distanceMask = distanceKernel[add_x:d-cut_x,add_y:d-cut_y]
	#distanceMask = get2dKernel(d,stdevD)[add_x:d-cut_x,add_y:d-cut_y]
	mask =  distanceMask * colourMask
	total = np.sum(mask * noFlash[low_x:high_x,low_y:high_y], axis = (0,1))
	totalDiv = np.sum(mask, axis = (0,1))
	return total/totalDiv

def jointBilateral(flash,noFlash,d,stdevC,stdevD):
	newImg = np.zeros((flash.shape[0],flash.shape[1],3),dtype = int)
	distanceKernel = get2dKernel(d,stdevD)
	for i in range(flash.shape[0]):
		for j in range(flash.shape[1]):
			newImg[i,j] = jointBilateralPix(flash,noFlash,d,i,j,stdevC,stdevD,distanceKernel)
	result = np.asarray(newImg, dtype=np.uint8)	
	return result

def main():
	imgname = 'joint' 
	#img = cv2.imread('./' + imgname, cv2.IMREAD_COLOR)
	noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR)
	flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR)

	stdev1 =3
	stdev2 = 35
	diam = 25
	filteredImg = jointBilateral(flashImg,noFlashImg,diam,stdev1,stdev2)
	#filteredImg = cv2.bilateralFilter(img,diam,stdev1,stdev2)
	name = imgname + str(diam) + '_' + str(stdev1) + '_' + str(stdev2) +'.png'
	path = 'C:/Users/rowan/Documents/uni_year_2/image processing/coursework'
	cv2.imwrite(os.path.join(path,name),filteredImg)
	if not filteredImg is None:
		cv2.namedWindow('bilateral filter')
		keep_processing = True
		while (keep_processing):
			cv2.imshow('bilateral filter', filteredImg)
			key = cv2.waitKey(40) & 0xFF #
			if (key == ord('x')):
				keep_processing = False
	else:
		print("No image file successfully loaded.")

cv2.destroyAllWindows();

def test():
	imgname = 'joint' 
	#img = cv2.imread('./' + 'test2.png', cv2.IMREAD_COLOR)
	diam = 25
	noFlashImg = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR)
	flashImg = cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR)
	stdev1 = 10
	stdev2 = 35
	#for stdev1 in [5,10,15]:
		#for stdev2 in [15,25,35]:
	Img = jointBilateral(flashImg,noFlashImg,diam,stdev1,stdev2) 
	filteredImg = Img
	name = 'new' + imgname + str(diam) + '_' + str(stdev1) + '_' + str(stdev2) +'.png'
	path = 'C:/Users/rowan/Documents/uni_year_2/image processing/coursework'
	cv2.imwrite(os.path.join(path,name),filteredImg)

test()

