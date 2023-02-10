import numpy as np
import matplotlib.pyplot as plt

line_type = ['rx-', 'cx-', 'mx-', 'yx-', 'kx-']


class TracePlot():
    def __init__(self, fun, xlim, ylim, x_resolution=20, y_resolution=20, plot_log=False):
        self.fun = fun
        self.xlim = xlim
        self.ylim = ylim
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.plot_log = plot_log

    def plot_contour(self, ax):
        x = np.linspace(self.xlim[0], self.xlim[1], self.x_resolution)
        y = np.linspace(self.xlim[0], self.xlim[1], self.y_resolution)
        x_, y_ = np.meshgrid(x, y)
        z = self.fun.fun_2D(x_, y_)
        if self.plot_log:
            z = np.log(z)
        ax.contourf(x_, y_, z, levels=20)
        # ax.colorbar()
        self.ax = ax

        return

    def prepare_trace(self, trace):
        '''Given the trace input, this step is to prepare the trace into ndarray self.trace,
        and also return the step number of the algorithm'''
        step_number = len(trace)
        self.trace = np.array(trace)
        return step_number

    def animate(self, i):
        if i == 0:
            self.ax.plot(self.trace[0, 0], self.trace[0, 1],'go',markersize=20)
        else:
            self.ax.plot(self.trace[i - 1:i + 1, 0], self.trace[i - 1:i + 1, 1], 'rx-')
