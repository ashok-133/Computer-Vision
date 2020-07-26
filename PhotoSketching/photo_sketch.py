#lets import required libraries
import cv2
import scipy.ndimage

#Read the image 
Path = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/PhotoSketching/obj2.jpg'
img = cv2.imread(Path)
img = cv2.resize(img,(1400,800))
#convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#invert the image
inverted_img = 255 - img_gray
#apply blur
blur = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=5)
final = cv2.addWeighted(blur,1,img_gray,1,0)


#lets display result image
cv2.imshow("result",final)

cv2.waitKey(0)
