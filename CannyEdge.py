'''
Canny Edge Detection using OpenCV
Canny Edge Detection is a popular edge detection approach.
It is use multi-stage algorithm to detect a edges.
It was developed by John F. Canny in 1986.
This approach combine with 5 steps.
1)Noise reduction(gauss)
2)Gradient calculation
3)Non-maximum suppresson
4)Double Threshold
5)Edge Tracking by Hysteresis
'''

import cv2
import numpy as np

img=cv2.imread("sources/UB.jpg")
img=cv2.resize(img,(400,400))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny=cv2.Canny(gray,50,70)

cv2.imshow("gray",gray)
cv2.imshow("image",img)
cv2.imshow("canny",canny)

cv2.waitKey()
cv2.destroyAllWindows()