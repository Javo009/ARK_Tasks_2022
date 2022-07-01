import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys



arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
video = cv2.VideoCapture(-1)
while True:
	ret, frame = video.read()

	if ret is False:
		break


	h, w, _ = frame.shape

	width=1000
	height = int(width*(h/w))
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

	detected_markers = aruco_display(corners, ids, rejected, frame)

	if len(corners)>0:
		ids=ids.flatten()

		for (corner,id) in zip(corners,ids):
			corners=corner.reshape(4,2)
			(topLeft,topRight,bottomRight,bottomLeft)=corners

			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(frame,topLeft,topRight,(0,255,0),2)
			cv2.line(frame,topRight,bottomRight,(0,255,0),2)
			cv2.line(frame,bottomRight,bottomLeft,(0,255,0),2)
			cv2.line(frame,bottomLeft,topLeft,(0,255,0),2)

			cv2.putText(frame, "ID = " + str(id),(topLeft[0], topLeft[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))


	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()