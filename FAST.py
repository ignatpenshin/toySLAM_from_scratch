import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('video.mp4')

crop_value = 2

if cap.isOpened():
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//crop_value)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//crop_value)
    fps = cap.get(cv.CAP_PROP_FPS)


# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=1)

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()


if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # find and draw the keypoints
    frame = cv.resize(frame, (width, height))
    kp = fast.detect(frame,None)
    kp, des = brief.compute(frame, kp)
    img2 = cv.drawKeypoints(frame, kp, None, color=(0,255,0))
    

    # Print all default params
    print("BRIEF Desc SIZE: {}".format(brief.descriptorSize()))
    print("Total Keypoints: {}".format(len(kp)))

    # Display the resulting frame
    cv.imshow('Frame', img2, )

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv.destroyAllWindows()
