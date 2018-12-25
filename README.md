'ip.py' implements the joint bilateral filter. This filter takes a pair of images, 
a noisy, non flash image taken in a low light environment, and a less noisy flash image(which has washed out colours
due to the camera flash).

The joint bilateral filter attempts to cosntruct an image with less noise than the non flash image, but with
better colour saturation than the flash image.

The main() function prompts for diameter, sigma color and sigma space.
It then runs the joint bilateral filter on 'test3a.jpg' and 'test3b.jpg'

This is saved to a file called 'jointbilat[diameter]_[sigmacolour]_[sigmaspace].png'

Where [var] is the numeric value of the input variable var
