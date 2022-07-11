import os
import cv2 as cv
import numpy as np
from R_to_Euler import rotationMatrixToEulerAngles
from slam import Extractor

def process_frame(video_list):

  s = Extractor()

  for video in video_list:
    cap = cv.VideoCapture(video)
    crop_value = 4

    if cap.isOpened():
      if (s.W and s.H and s.K) is None:
        s.W  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//crop_value)
        s.H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//crop_value)
        s.K = np.array([[s.F,0,s.W//2], [0,s.F,s.H//2], [0, 0, 1]])
        s.Kinv = np.linalg.inv(s.K)
        fps = cap.get(cv.CAP_PROP_FPS)

    if (cap.isOpened()== False): 
      print("Error opening vid eo stream or file")

    while(cap.isOpened()):

      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        img = cv.resize(frame, (s.W, s.H))
        frame = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        kps, des, matches, R, t = s.extract(frame)
        
        if R is None: continue

        print("Translation: ", t.T)
        alpha, betta, kappa = rotationMatrixToEulerAngles(R)
        print("Rotation: ", alpha, betta, kappa)
        
        #main_plot(t_l)

        for pt1, pt2 in matches:
          u1, v1 = s.denormailze(pt1)
          u2, v2 = s.denormailze(pt2)
          cv.circle(img, (u1,v1), color=(0,0,255), radius=10)
          cv.line(img, (u1,v1), (u2,v2), color=(255,0,0), thickness = 2)

        frame = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

        # Display the resulting frame
        cv.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
          break
      else:
        cap.release() 

video_list = []
os.chdir('VIDEO')
for i in os.listdir():
  if i.endswith('.mp4'):
    video_list.append(i)

process_frame(video_list)