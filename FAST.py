import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('video.mp4')

if cap.isOpened():
    width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)   
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
      


# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()


if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # find and draw the keypoints
    kp = fast.detect(frame,None)
    img2 = cv.drawKeypoints(frame, kp, None, color=(0,255,0))

    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    # Display the resulting frame
    cv.imshow('Frame', img2)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv.destroyAllWindows()


# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)
# print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# cv.imwrite('fast_false.png',img3)