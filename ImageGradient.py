'''
Image Gradient--
It is a directional change in the color or intensity in an image.
It is most important part to find information from image
Like finding edges within the images.
There are various methods to find image gradient.
These are - Laplacian Derivatives,SobelX and SobelY.
All these functions have different mathematical approach to get result.
All load image in the gray scale
'''

import cv2
import numpy as np
img=cv2.imread("UB.jpg")
img=cv2.resize(img,(400,400))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

'''
Laplacian Derivative---It calculate laplace derivate
parameter(img,data_type for -ve val,ksize)
kernal size must be odd
'''
lap=cv2.Laplacian(gray,cv2.CV_64F,ksize=3) #also pass kernel size
lap=np.uint8(np.absolute(lap)) # laplace func can give negative values, so we make it from 0 to inf

'''
Sobel operation is a joint Gausssian smoothing plus differentiation operation, so it is more resistant to noise.
This is use for x and y both parameter (img,type for -ve val,x = 1,y = 0,ksize).
Sobel X focus on vertical lines
Sobel y focus on horizontal lines
Sobel X and Sobel Y are first-order derivatives.
'''

sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobelx = np.uint8(np.absolute(sobelx))
sobely= np.uint8(np.absolute(sobely))
sobelcombine=cv2.bitwise_or(sobelx,sobely)
cv2.imshow("original==",img)
cv2.imshow("gray====",gray)
cv2.imshow("Laplacian==",lap)
cv2.imshow("SobelX===",sobelx)
cv2.imshow("SobelY==",sobely)
cv2.imshow("COmbined image==",sobelcombine)
cv2.waitKey(0)
cv2.destroyAllWindows()