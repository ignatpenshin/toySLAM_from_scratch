import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def easy_plot(r):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    x.append(num[0])
    y.append(num[1])
    z.append(num[2])
    ax.scatter(r) # plot the point (2,3,4) on the figure
    plt.show()


def main_plot(nums):
    def update_lines(graph, num):
        #dx, dy, dz = num[0], num[1], num[2]  # replace this line with code to get data from serial line
        #text.set_text("{:d}: [{:.0f},{:.0f},{:.0f}]".format(num, dx, dy, dz))  # for debugging
        x.append(num[0])
        y.append(num[1])
        z.append(num[2])
        graph._offsets3d = (x, y, z)
        return graph


    x = [0]
    y = [0]
    z = [0]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    graph = ax.scatter(x, y, z, color='orange')
    text = fig.text(0, 1, "TEXT", va='top')  # for debugging

    ax.set_xlim3d(-255, 255)
    ax.set_ylim3d(-255, 255)
    ax.set_zlim3d(-255, 255)

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, fargs=(graph, nums), blit=False, repeat=True)
    plt.show()