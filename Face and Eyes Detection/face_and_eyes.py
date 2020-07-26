import cv2

#Lets import required two files
#path to haarcascade face file
path1 = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/Face and Eyes Detection/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path1)
#path2 is path to haarcascade eyes file
path2 = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/Face and Eyes Detection/haarcascade_eye.xml'
eyes_cascade = cv2.CascadeClassifier(path2)
#import video
path_video = '/Users/kandagadlaashokkumar/Desktop/OPENCV-YOUTUBE/Face and Eyes Detection/avengers1.mp4'

cap = cv2.VideoCapture(path_video)

while True:
    ret,img = cap.read()
    #every frame of video is stored in img variable
    #convert img(RGB) to GRAY
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,2)
    #draw bounding boxes 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray,1.3,4)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("output",img)
    cv2.waitKey(1000//25)
cap.release()
    
