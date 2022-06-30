import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


#left_image = cv.imread('left.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('right-1.png', cv.IMREAD_GRAYSCALE)

left_image = cv.imread('left.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('right-1.png', cv.IMREAD_GRAYSCALE)
bike_image = cv.imread('bike.png', cv.IMREAD_GRAYSCALE)

#left_image = cv.imread('items2_l.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('items2_r.png', cv.IMREAD_GRAYSCALE)         



stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
depth = stereo.compute(left_image, right_image)

threshold = 0.91
res = cv.matchTemplate(left_image,bike_image,cv.TM_CCOEFF_NORMED)
cood_match = np.where(res >= threshold)

if(len(cood_match[0]) != 0 and len(cood_match[1]) != 0):
    for pt in zip(*cood_match[::-1]):
        cv.rectangle(left_image, pt, (pt[0] + bike_image.shape[0], pt[1] + bike_image.shape[1]), (0,255,255), 3)
        cv.imshow("temp",left_image)
        cv.waitKey(0)


print(depth)

cv.imshow("Left", left_image)
cv.imshow("right", right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()