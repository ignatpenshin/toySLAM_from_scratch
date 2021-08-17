import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
orb = cv2.ORB_create()

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv2.destroyAllWindows()