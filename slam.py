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
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    frame = cv.resize(frame, (width, height))
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    frame = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv.destroyAllWindows()