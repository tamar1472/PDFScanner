
import sys
import cv2
import numpy as np
import math


# Getting argument from terminal
img = sys.argv[1]
output = sys.argv[2]
image = cv2.imread(img, cv2.IMREAD_COLOR)


# resize window to see image
img_resized = cv2.resize(image, (700, 800))
imgGray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# blurring the image
blurred = cv2.GaussianBlur(imgGray, (11, 11), cv2.BORDER_DEFAULT)

# creating Binarization
ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cannyImage = cv2.Canny(thresh, threshold1=120, threshold2=255, edges=1)

# find contours
contours, hierarchy = cv2.findContours(cannyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Store the corners:
cornerList = []

# Look for the outer bounding boxes:
for i, c in enumerate(contours):
    # Approximate the contour to a polygon:
    contoursPoly = cv2.approxPolyDP(c, 3, True)

    # Convert the polygon to a bounding rectangle:
    boundRect = cv2.boundingRect(contoursPoly)

    # Get the bounding rectangle's data:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight
    # Set a min area threshold
    minArea = 100000

    # Filter blobs by area:
    if rectArea > minArea:
        # Get the convex hull for the target contour: - convexhull is a tight-fitting convex boundary around the point
        hull = cv2.convexHull(c)
        # Draw the hull:
        color = (0, 255, 0)
        cv2.polylines(img_resized, [hull], True, color, 2)

        # Create image for good features to track:
        (height, width) = cannyImage.shape[:2]

        # Black image same size as original input:
        hullImg = np.zeros((height, width), dtype=np.uint8)

        # Draw the points:
        cv2.drawContours(hullImg, [hull], 0, 255, 2)

        # Get the corners: - goodFeaturesToTrack finds the n strongest corners in the image
        corners = cv2.goodFeaturesToTrack(hullImg, 4, 0.01, int(max(height, width) / 4))
        corners = np.int0(corners)

        # Loop through the corner array and store/draw the corners:
        for c in corners:
            # Flat the array of corner points:
            (x, y) = c.ravel()
            # Store the corner point in the list:
            cornerList.append((x, y))
            print(cornerList)
            # Draw the corner points:
            cv2.circle(img_resized, (x, y), 5, (0, 0, 255), 5)

            # print all 4 corners
            cv2.imshow("Corners", img_resized)
        cv2.waitKey(0)

# sort corner list from top left to bottom left (clockwise), atan- calculates degree
cornerList = sorted(cornerList, key=lambda point: math.atan2(point[1] - (height / 2), point[0] - (width / 2)))
# corners
topLeft = cornerList[0]
topRight = cornerList[1]
bottomRight = cornerList[2]
bottomLeft = cornerList[3]
# get height and width
w1 = abs(bottomRight[0] - bottomLeft[0])
w2 = abs(topRight[0] - topLeft[0])
h1 = abs(topRight[1] - bottomRight[1])
h2 = abs(topLeft[1] - bottomLeft[1])

width = max([w1, w2])
height = max([h1, h2])
# sort them as list
input_pts = np.float32([list(topLeft), list(bottomLeft), list(bottomRight), list(topRight)])

output_pts = np.float32([[0, 0],
                         [0, height - 1],
                         [width - 1, height - 1],
                         [width - 1, 0]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)
# apply the perspective transformation to the image
out = cv2.warpPerspective(img_resized, M, (width, height), flags=cv2.INTER_LINEAR)

cv2.imshow("transformed", out)
cv2.imwrite(output, out)
cv2.waitKey(0)
