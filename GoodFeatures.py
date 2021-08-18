import cv2 as cv
import numpy as np

cap = cv.VideoCapture('video.mp4')
orb = cv.ORB_create()

crop_value = 2

if cap.isOpened():
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//crop_value)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//crop_value)
    fps = cap.get(cv.CAP_PROP_FPS)


if (cap.isOpened()== False): 
  print("Error opening vid eo stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # 1. Detection
    img = cv.resize(frame, (width, height))
    frame = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #MASK? 
    # x,y,w,h = cv.boundingRect(frame)
    # frame = frame[int(h*cropY//2):int(h*(1-cropY//2)), int(w*cropX//2):int(w*(1-cropX//2))] 
        
    corners = cv.goodFeaturesToTrack(image=frame, maxCorners=250, qualityLevel=0.01, minDistance=3)   #corners=..., mask=..., blockSize=..., useHarrisDetector=..., k=...
    kps = [cv.KeyPoint(x=f[0][0], y=f[0,1], _size=20) for f in corners]
    kps, des = orb.compute(img, kps)

    frame = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv.destroyAllWindows()