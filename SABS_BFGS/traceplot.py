import numpy as np
import matplotlib.pyplot as plt

class TracePlot():
    def __init__(self,trace,fun,xlim,ylim,x_resolution=20,y_resolution=20,plot_log=False):
        self.x = np.array(trace['x_k'])
        self.p = trace['p_k']
        self.alpha = trace['alpha_k']
        self.s = trace['s_k']
        self.fun = fun
        self.xlim = xlim
        self.ylim = ylim
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.plot_log = plot_log


    def plot_contour(self,ax):
        x = np.linspace(self.xlim[0], self.xlim[1], self.x_resolution)
        y = np.linspace(self.xlim[0], self.xlim[1], self.y_resolution)
        x_, y_ = np.meshgrid(x, y)
        z = self.fun.fun_2D(x_, y_)
        if self.plot_log:
            z = np.log(z)
        ax.contourf(x_, y_, z, levels=20)
        #ax.colorbar()
        self.ax = ax

        return


    def animate(self,i):
        self.ax.plot(self.x[i-1:i+1,0],self.x[i-1:i+1,1],'r')


