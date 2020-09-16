import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
time.sleep(2)
tracker = cv2.TrackerMOSSE_create()
#tracker = cv2.TrackerCSRT_create()
success,img = cap.read()
bbox = cv2.selectROI("Tracking",img,False,True)
print(bbox)
print(img.shape)
tracker.init(img,bbox)

def drawbox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img,"TRACKING",(75,75),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)



while True:
    timer = cv2.getTickCount()
    success,img = cap.read()

    success,bbox = tracker.update(img)
    if success == True:
        drawbox(img,bbox)
    else:
            cv2.putText(img,"LOST",(75,75),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)


    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,"FPS:"+str(int(fps)),(75,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("tracking",img)
    
    
    if cv2.waitKey(1) & 0xFF =='q':
        break