from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


class Anim_pos:
    def __init__(self, figsize=(7, 8)):
        self.axis_length = 0.5
        self.poses_wTi = []
        self.num_poses = len(self.poses_wTi)
        #self.colors_arr = np.array([[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), self.num_poses)]).squeeze()
        self._, self.ax = plt.subplots(figsize=figsize)

    def anim(self, poses_wTi):
        for i, wTi in enumerate(poses_wTi):
            wti = wTi[:3, 3]
            posx = wTi @ np.array([self.axis_length, 0, 0, 1]).reshape(4, 1)
            posz = wTi @ np.array([0, 0, self.axis_length, 1]).reshape(4, 1)
            self.ax.plot([wti[0], posx[0]], [wti[2], posx[2]], "b", zorder=1)
            self.ax.plot([wti[0], posz[0]], [wti[2], posz[2]], "k", zorder=1)
            self.ax.scatter(wti[0], wti[2], 40, marker=".", zorder=2)
        plt.axis("equal")
        plt.title("Egovehicle trajectory")
        plt.xlabel("x camera coordinate (of camera frame 0)")
        plt.ylabel("z camera coordinate (of camera frame 0)")
    

    def plot_poses(self) -> None:
        line_ani = animation.FuncAnimation(fig = self._, func = self.anim, fargs=(self.poses_wTi))
        plt.show()


