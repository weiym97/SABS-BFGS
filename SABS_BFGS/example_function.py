import numpy as np

class Resenbrock():
    def __init__(self,a=1,b=100):
        self.a = a
        self.b = b

    def fun(self,x):
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

    def fun_2D(self,x,y):
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def Jacobi(self,x):
        f_1 = 2 * (x[0] - self.a) + 4 * self.b * x[0] * (x[0] ** 2 -x[1])
        f_2 = 2 * self.b * (x[1] -x[0] ** 2)
        return np.array([f_1,f_2])

    def Jacobi_2D(self,x,y):
        f_1 = 2 * (x - self.a) + 4 * self.b * x * (x ** 2 - y)
        f_2 = 2 * self.b * (y - x ** 2)
        return f_1,f_2

    def Hess(self,x):
        f_11 = 2 + 12 * self.b * x[0] ** 2 - 4 * self.b * x[1]
        f_12 = -4 * self.b * x[0]
        f_22 = 2 * self.b
        return np.array([[f_11,f_12],[f_12,f_22]])