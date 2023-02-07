import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SABS_BFGS.bfgs import BFGS

def func_1_p(x):
    return [2 * x[0],2 * x[1]]

def func_2(x):
    return np.sin(x[0]) + np.cos(x[1])

def func_2_p(x):
    return [np.cos(x[0]), - np.sin(x[1])]

class Resenbrock():
    def __init__(self, a=1, b=100):
        self.a = a
        self.b = b

    def fun(self, x):
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

    def fun_2D(self, x, y):
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def Jacobi(self, x):
        f_1 = 2 * (x[0] - self.a) + 4 * self.b * x[0] * (x[0] ** 2 - x[1])
        f_2 = 2 * self.b * (x[1] - x[0] ** 2)
        return np.array([f_1, f_2])

    def Jacobi_2D(self, x, y):
        f_1 = 2 * (x - self.a) + 4 * self.b * x * (x ** 2 - y)
        f_2 = 2 * self.b * (y - x ** 2)
        return f_1, f_2

    def Hess(self, x):
        f_11 = 2 + 12 * self.b * x[0] ** 2 - 4 * self.b * x[1]
        f_12 = -4 * self.b * x[0]
        f_22 = 2 * self.b
        return np.array([[f_11, f_12], [f_12, f_22]])

if __name__ == '__main__':
    '''
    fun_class = Resenbrock()
    x0 = np.array([-1.5, -1.5])
    bfgs = BFGS()
    x_argmin, f_min, trace = bfgs.minimize(fun_class.fun, x0, return_trace=True)
    print(x_argmin)
    print(f_min)
    print(trace)
    print(np.array(trace['x_k']).shape)
    '''

    fig, ax = plt.subplots()

    x = np.arange(0, 2 * np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))


    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))  # update the data
        return line,


    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,


    ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                                  interval=25, blit=True)
    plt.savefig('try.jpg')