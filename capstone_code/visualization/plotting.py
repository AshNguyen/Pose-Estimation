import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def plot_pose(xs, ys, w, h):
    import matplotlib.pyplot as plt
    pairs = [[0,1], [0,2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2,8], [7, 9], [9, 11], [8, 10], [10, 12], [7, 8]]
    plt.figure(figsize=(12,8))
    ys = h - ys
    plt.scatter(xs, ys, s=150)
    for pair in pairs:
        plt.plot(xs[pair], ys[pair], lw=5)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.show()
    
    
def plot_pose_visible(xs, ys, vis, w, h):
    import matplotlib.pyplot as plt
    pairs = [[0,1], [0,2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2,8], [7, 9], [9, 11], [8, 10], [10, 12], [7, 8]]
    plt.figure(figsize=(12,8))
    ys = h - ys
    vis = list(vis)
    print(vis)
    plt.scatter(xs[vis], ys[vis])
    for pair in pairs:
        vis_or_not = [vis[point] for point in pair]
        if all(vis_or_not):
            plt.plot(xs[pair], ys[pair], lw=5)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.show()
