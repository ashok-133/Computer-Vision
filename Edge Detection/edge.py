import cv2


PATH = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/Edge Detection/edge.jpg'

img = cv2.imread(PATH)
img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(3,3),0)


canny = cv2.Canny(img_blur,100,200)
laplacian = cv2.Laplacian(img_blur,cv2.CV_64F)
sobelx = cv2.Sobel(img_blur,cv2.CV_64F,1,0,ksize = 5)
sobely = cv2.Sobel(img_blur,cv2.CV_64F,0,1,ksize = 5)
#Lets Display every edge detection technique

cv2.imshow("Orginal",img)
cv2.imshow("Canny",canny)
cv2.imshow("Laplacian",laplacian)
cv2.imshow("Sobelx",sobelx)
cv2.imshow("Sobelly",sobely)
cv2.imwrite("/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/Edge Detection/edge1.jpg",canny)
cv2.waitKey(0)
