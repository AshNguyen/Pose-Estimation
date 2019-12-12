import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def make_animation(pose, draw_function, frames, interval, filename=None,
                   verbose=False, figure_params=None):
    from matplotlib import pyplot, animation
    from IPython.display import HTML, display, clear_output
    import random

    if filename is None:
        filename = 'animate_%06i.gif' % random.randint(0, 999999)
    # Create figure
    if figure_params is None:
        figure_params = {}
    figure = pyplot.figure(**figure_params)
    # Wrap draw_function if we need to print to console
    if verbose:
        old_draw_function = draw_function
        def draw_function(pose, current_frame_number, total_frame_count):
            old_draw_function(pose, current_frame_number, total_frame_count)
            print('Processed frame', current_frame_number + 1, '/', total_frame_count)
            clear_output(wait=True)
            if current_frame_number + 1 == total_frame_count:
                print('Writing animation to file...')
                clear_output(wait=True)
    # Generate animation
    anim = animation.FuncAnimation(
        figure, draw_function, frames=frames, interval=interval,
        init_func=lambda: None, fargs=(frames,))
    anim.save(filename, writer='imagemagick')
    # Close the animation figure so the last frame does not get displayed
    # in the notebook.
    pyplot.close()

def animate_pose(pose, frame, total_frames):
    pairs = [[0,1], [0,2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2,8], [7, 9], [9, 11], [8, 10], [10, 12], [7, 8]]
    xs = pose[frame, :13]
    ys = pose[frame, 13:26]
    ys = 1 - ys
    plt.clf()
    plt.scatter(xs, ys)
    for pair in pairs:
        plt.plot(xs[pair], ys[pair], lw=5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    
def animate_pose_visible(pose, frame, total_frames):
    pairs = [[0,1], [0,2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2,8], [7, 9], [9, 11], [8, 10], [10, 12], [7, 8]]
    xs = pose[frame, :13]
    ys = pose[frame, 13:26]
    ys = 1 - ys
    vis = [1 if v > 0.5 else 0 for v in pose[frame, 26:]]
    plt.clf()
    plt.scatter(xs[vis], ys[vis])
    for pair in pairs:
        vis_or_not = [vis[point] for point in pair]
        if all(vis_or_not):
            plt.plot(xs[pair], ys[pair], lw=5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    