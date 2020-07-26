import cv2
import numpy as np

PATH = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/ColorDetection/car.jpg'
lower = np.array([1,16,136])
higher = np.array([20,255,255])

img = cv2.imread(PATH)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(imgHSV,lower,higher)
imgResult = cv2.bitwise_and(img,img,mask = mask)

cv2.imshow("Orginal",img)
cv2.imshow("HSV",imgHSV)
cv2.imshow("Mask",mask)
cv2.imshow("Result",imgResult)
cv2.imwrite("/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/ColorDetection/carres.jpg",imgResult)
cv2.waitKey(0)