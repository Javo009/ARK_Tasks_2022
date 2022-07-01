import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys






print("Loading image...")
image = cv2.imread("aruco_ark.png")
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)




# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
print(ids, corners, rejected)
detected_markers = aruco_display(corners, ids, rejected, image)

if len(corners)>0:
		ids=ids.flatten()

		for (corner,id) in zip(corners,ids):
			corners=corner.reshape(4,2)
			(topLeft,topRight,bottomRight,bottomLeft)=corners

			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image,topLeft,topRight,(0,255,0),2)
			cv2.line(image,topRight,bottomRight,(0,255,0),2)
			cv2.line(image,bottomRight,bottomLeft,(0,255,0),2)
			cv2.line(image,bottomLeft,topLeft,(0,255,0),2)

			cv2.putText(image, "ID = " + str(id),(topLeft[0], topLeft[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))

cv2.imshow("Image", detected_markers)

# # Uncomment to save
# cv2.imwrite("output_sample.png",detected_markers)

cv2.waitKey(0)