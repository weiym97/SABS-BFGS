import numpy as np
import scipy.optimize


class BFGS():
    '''This is the place for class comments'''
    def __init__(self):
        pass

    def minimize(self,fun,x0,jac=None,args=(),tol=1e-8, return_trace = False):
        '''This is the place for minimize comments'''

        # Initial Hessian Inverse matrix
        x = x0
        H = np.eye(len(x))
        jac_old = np.array(jac(x))

        while True:
            p = -H.dot(jac(x))
            alpha,_,_,_,_,_ = scipy.optimize.line_search(fun, jac, x,p)
            s = alpha * p
            jac_new = np.array(jac(x + s))
            y = jac_new - jac_old
            H = self.update_H(H,y,s)
            x = x + s
            # If the norm of jacobi is smaller than the threshold, then break
            if  np.linalg.norm(jac_new,ord=2) < tol:
                break
            jac_old = jac_new

        return x,fun(x)

    def update_H(self,H,y,s):
        temp_1 = (s.dot(y) + y.dot(H).dot(y)) / (s.dot(y)) ** 2 * np.outer(s,s)
        temp_2 = - (np.matmul(H,np.outer(y,s)) + np.matmul(np.outer(s,y),H)) / s.dot(y)
        return H + temp_1 + temp_2

def func_1(x):
    return x[0] ** 2 + x[1] ** 2

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
    x0 = np.array([1.5,1.5])
    bfgs = BFGS()
    x_argmin,f_min = bfgs.minimize(fun_class.fun,x0,jac=fun_class.Jacobi)
    print(x_argmin)
    print(f_min)
    '''

    x0 = np.array([10, 10])
    bfgs = BFGS()
    x_argmin, f_min = bfgs.minimize(func_2, x0, jac=func_2_p)
    print(x_argmin)
    print(f_min)