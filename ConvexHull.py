import cv2
import numpy as np

'''
Approximate Convex Hulls:

- Approximate convex hulls are very useful because they give a good estimate of the boundary of a set of points, 
  often using much less space than the true convex hull.

- Example:
    The star shape can be represented with a bounding rectangle instead of outlining the entire star.
    This rectangle is the smallest possible rectangle that can fit the star.

Key Concepts:
- cv2.arcLength(): Returns the perimeter of a contour.
- cv2.approxPolyDP(): Approximates the contour shape to a simpler polygon.
- cv2.boundingRect(): Finds the smallest rectangle that encloses the shape.
- cv2.convexHull(): Finds the convex hull — a tight-fitting convex boundary.

Difference:
- Approximation = closely traces the actual shape.
- Convex Hull = connects the outermost points to form a convex polygon (e.g., fingertips of a hand).
'''

# Step 1: Load image
img = cv2.imread("shapes.jpg")

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply binary threshold
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Step 4: Find contours
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = []

# Step 5: Loop over contours
for c in cnts:
    # Calculate centroid using image moments
    m = cv2.moments(c)
    if m["m00"] == 0:
        continue  # Avoid division by zero
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])

    # Calculate area of the contour
    area = cv2.contourArea(c)
    areas.append(area)

    # Approximate contour using arcLength and approxPolyDP
    epsilon = 0.01 * cv2.arcLength(c, True)  # Lower epsilon = better approximation
    '''
    cv2.approxPolyDP Parameters:
    - curve: Input contour
    - epsilon: Approximation accuracy (max distance from original curve)
    - closed: Whether the shape is closed
    '''
    data = cv2.approxPolyDP(c, epsilon, True)

    # Compute convex hull from approximated points
    '''
    Convex hull is the convex shape formed by joining the outermost points of a figure.
    - Example: For a hand, the convex hull might be a polygon connecting fingertips and wrist.
    '''
    hull = cv2.convexHull(data)

    # Option 1: Draw bounding rectangle as a forced "convex hull"
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(img, (x, y), (x + w, y + h), (125, 10, 20), 5)

    # Option 2 (commented): Draw natural convex hull
    # cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)

    # Draw the approximated shape
    cv2.drawContours(img, [data], -1, (50, 100, 50), 2)

    # Draw centroid
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 255, 255), 2)

# Step 6: Show images
cv2.imshow("Original with Contours", img)
cv2.imshow("Gray", gray)
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()