import cv2 as cv
import numpy as np
from plot import Anim_pos

class Extractor(object):
  def __init__(self):
    self.orb = cv.ORB_create()
    self.bf = cv.BFMatcher(cv.NORM_HAMMING2)
    self.last = None
    self.F = 1
    self.K = None 
    self.Kinv = None
    self.W, self.H = None, None
    #
    self.animate = Anim_pos()

  def world_poses(self, R, t):
    Rt = np.eye(4)
    Rt[:3, :4] = np.concatenate([R, t.reshape(3, 1)], axis = 1)
    print(Rt) 
    if self.animate.poses_wTi == []:
      self.animate.poses_wTi += [np.eye(4)]
    wTi1 = self.animate.poses_wTi[-1]
    Rt_2inv = np.linalg.inv(Rt)
    wTi2 = wTi1 @ Rt_2inv
    self.animate.poses_wTi += [wTi2]

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
    Z=np.mat([[0,1,0],[-1,0,0],[0,0,0]],dtype=float)
    u, lm, vt = np.linalg.svd(E)
    lm = np.diag(lm)

    if np.linalg.det(vt.T) < 0:  # vt or v?
      vt *= -1.0
    if np.linalg.det(u) < 0:
      u *= -1.0
    R = np.dot(np.dot(u, W), vt)
    if np.sum(R.diagonal()) < 0:
      R = np.dot(np.dot(u, W.T), vt)   
    t = u[:, 2]
    Rt = np.concatenate([R, t.reshape(3, 1)], axis = 1)  
    return Rt

  def extract(self, frame):
    #detection
    corners = cv.goodFeaturesToTrack(image=frame, maxCorners=1000, qualityLevel=0.01, minDistance=3) 

    #extraction
    kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in corners]
    kps, des = self.orb.compute(frame, kps)
    R = None 
    t = None
     
    #matching
    ret = []

    if self.last is not None:
      matches = self.bf.knnMatch(des, self.last['des'], k=2)
      for m,n in matches:
        if m.distance < 0.5*n.distance:
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

      #R, t = self.pose_from_E(E) #Self-made method
      _, R, t, mask = cv.recoverPose(E = E, points1 = np.int32([i[0] for i in ret]),
                                      points2 = np.int32([i[1] for i in ret]),
                                      cameraMatrix = self.K, mask = mask)
      self.world_poses(R, t)
      self.animate.plot_poses()


      ret = ret[mask.ravel()==1]

            
    self.last = {'kps': kps, 'des' : des}
    return kps, des, ret, R, t



