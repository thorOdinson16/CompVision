import cv2
import numpy as np

'''
Contours:
Contours can be explained simply as a curve joining all the continuous points (along the boundary),
having the same color or intensity.

Contours are useful for:
- Shape analysis
- Object detection and recognition

Best practices:
- Use binary images (black & white) for better accuracy
- Apply edge detection before finding contours
- The findContours() function modifies the original image, so copy it before use
- findContours() assumes white objects on a black background
'''

# Step 1: Load image
img = cv2.imread("logo2.png")

# Step 2: Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply thresholding to get binary image
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

# Step 4: Find contours
'''
cv2.findContours returns:
- contours: a list of contour points
- hierarchy: information about the image topology

Arguments:
- thresh: input binary image
- cv2.RETR_TREE: retrieves all contours and reconstructs a full hierarchy of nested contours
- cv2.CHAIN_APPROX_SIMPLE: removes all redundant points and compresses the contour
'''
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(cnts))

# Step 5: Draw a specific contour (e.g., 3rd contour)
c = cnts[2]
cv2.drawContours(img, [c], 0, (0, 255, 0), 3)

# Step 6: Display images
cv2.imshow("Original with Contour", img)
cv2.imshow("Gray Image", gray)
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
To calculate the center of a contour, use image moments.
cv2.moments(contour) returns a dictionary of spatial moments.
Common moments:
- m00: Area
- m10/m00, m01/m00: Centroid coordinates
'''

for c in cnts:
    # Draw each contour
    cv2.drawContours(img, [c], -1, (252, 3, 161), 2)

    # Calculate moments and centroid
    m = cv2.moments(c)
    if m["m00"] != 0:
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        # Draw the centroid
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(img, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)