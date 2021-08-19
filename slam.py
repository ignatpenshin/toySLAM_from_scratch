import cv2 as cv
import numpy as np


class Extractor(object):
  def __init__(self):
    self.orb = cv.ORB_create()
    self.bf = cv.BFMatcher(cv.NORM_HAMMING)
    self.last = None
  

  def extract(self, frame):
    #detection
    corners = cv.goodFeaturesToTrack(image=frame, maxCorners=5000, qualityLevel=0.01, minDistance=3) 

    #extraction
    kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in corners]
    kps, des = self.orb.compute(frame, kps)
     
    #matching
    ret = []
    if self.last is not None:
      matches = self.bf.knnMatch(des, self.last['des'], k=2)
      for m,n in matches:
        if m.distance < 0.65*n.distance:
          ret.append((kps[m.queryIdx], self.last['kps'][m.trainIdx]))
    self.last = {'kps': kps, 'des' : des}
    return kps, des, ret

  
def process_frame(video):

  cap = cv.VideoCapture(video)
  crop_value = 2
  
  if cap.isOpened():
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//crop_value)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//crop_value)
    fps = cap.get(cv.CAP_PROP_FPS)

  s = Extractor()

  if (cap.isOpened()== False): 
    print("Error opening vid eo stream or file")

  while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      # print(cap.get(cv.CAP_PROP_POS_FRAMES))
      img = cv.resize(frame, (width, height))
      frame = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

      kps, des, matches = s.extract(frame)

      for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1.pt)
        u2, v2 = map(lambda x: int(round(x)), pt2.pt)
        cv.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv.line(img, (u1,v1), (u2,v2), color=(255,0,0))


      frame = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
    
      # Display the resulting frame
      cv.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv.waitKey(25) & 0xFF == ord('q'):
        break

process_frame('video.mp4')

