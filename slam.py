import cv2 as cv
import numpy as np
import os

from matplotlib import pyplot as plt

class Extractor(object):
  def __init__(self):
    self.orb = cv.ORB_create()
    self.bf = cv.BFMatcher(cv.NORM_HAMMING2)
    self.last = None
    self.F = 75
    self.K = None 
    self.Kinv = None
    self.W, self.H = None, None

  def add_ones(self, x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

  def normalize(self, pts):
    return np.dot(self.Kinv, self.add_ones(pts).T).T[:, 0:2]

  def denormailze(self, pt):
    ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
    return int(ret[0]), int(ret[1])

  def pose_from_E(self, E):
    #Hartley & Zisserman (Chapter 6, Essential -> R,t)
    W=np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    u, w, vt = np.linalg.svd(E)
    if np.linalg.det(vt.T) < 0:  # vt or v?
      vt *= -1.0
    if np.linalg.det(u) < 0:
      u *= -1.0
    R = np.dot(np.dot(u, W), vt)
    if np.sum(R.diagonal()) < 0:
      R = np.dot(np.dot(u, W.T), vt)   
    t = u[:, 2]
    return R, t

  def extract(self, frame):
    #detection
    corners = cv.goodFeaturesToTrack(image=frame, maxCorners=1000, qualityLevel=0.01, minDistance=3) 

    #extraction
    kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in corners]
    kps, des = self.orb.compute(frame, kps)
     
    #matching
    ret = []

    if self.last is not None:
      matches = self.bf.knnMatch(des, self.last['des'], k=2)
      for m,n in matches:
        if m.distance < 0.65*n.distance:
          ret.append((kps[m.queryIdx].pt, self.last['kps'][m.trainIdx].pt))
      ret = np.array(ret)

      # normalize keypoint coords to move to 0:
      ret[:, 0, :] = self.normalize(ret[:, 0, :]) 
      ret[:, 1, :] = self.normalize(ret[:, 1, :])
      
      #filter
      E, mask = cv.findEssentialMat(points1 = np.int32([i[0] for i in ret]),
                                    points2 = np.int32([i[1] for i in ret]),
                                    cameraMatrix = self.K,
                                    method = cv.FM_RANSAC, 
                                    prob = 0.99, threshold = 0.01)
      ret = ret[mask.ravel()==1]
      
      R, t = self.pose_from_E(E)
      print(R, t)      
      
    self.last = {'kps': kps, 'des' : des}
    return kps, des, ret

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

        kps, des, matches = s.extract(frame)

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

