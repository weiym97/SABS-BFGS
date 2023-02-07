import collections.abc

import autograd.numpy as np
from autograd import grad
import scipy.optimize


class BFGS():
    '''This is the place for class comments'''

    def minimize(self,fun,x0,jac=None,args=(),tol=1e-8, return_trace = False):
        '''This is the place for minimize comments'''
        jac,x,H,jac_old = self.initialize(fun,jac,x0)

        if return_trace:
            trace = self.initialize_trace()

        while True:
            # Itegrate over to get the next state
            p,alpha,s,jac_new,y = self.iteration(fun,jac,x,jac_old,H)
            if return_trace:
                trace = self.update_trace(trace,x,p,alpha,s)

            # Update x, H,jac_old for the next step
            x,H,jac_old = self.update_state(x,H,y,s,jac_new)

            # If the norm of jacobi is smaller than the threshold, then break
            if  np.linalg.norm(jac_old,ord=2) < tol:
                break
        if return_trace:
            return x,fun(x),trace
        else:
            return x,fun(x)

    def initialize(self,fun,jac,x0):
        # If Jacobian is not given, use autograd
        if jac is None:
            jac = grad(fun)
        # Deal with univariate input
        if np.isscalar(x0):
            x0 = np.array([x0])
        else:
            x0 = np.array(x0)
        # Change to float to ensure differentiable
        x0 = x0.astype(np.float32)

        # Initial Hessian Inverse matrix
        x = x0
        H = np.eye(len(x))

        # Initialize Jacobian matrix
        jac_old = np.array(jac(x))

        return jac,x,H,jac_old

    def initialize_trace(self):
        return {'x_k':[],'p_k':[],'alpha_k':[],'s_k':[]}

    def iteration(self,fun,jac,x,jac_old,H):
        p = -H.dot(jac(x))
        alpha, _, _, _, _, _ = scipy.optimize.line_search(fun, jac, x, p)
        s = alpha * p
        jac_new = np.array(jac(x + s))
        y = jac_new - jac_old
        return p, alpha, s, jac_new,y

    def update_state(self,x,H,y,s,jac_new):
        x = x + s
        H = self.update_H(H, y, s)
        jac_old = jac_new
        return x, H, jac_old

    def update_trace(self,trace,x,p,alpha,s):
        trace['x_k'].append(x)
        trace['p_k'].append(p)
        trace['alpha_k'].append(alpha)
        trace['s_k'].append(s)
        return trace

    def update_H(self,H,y,s):
        temp_1 = (s.dot(y) + y.dot(H).dot(y)) / (s.dot(y)) ** 2 * np.outer(s,s)
        temp_2 = - (np.matmul(H,np.outer(y,s)) + np.matmul(np.outer(s,y),H)) / s.dot(y)
        return H + temp_1 + temp_2


def func_1(x):
    return x[0] ** 2 + x[1] ** 2




if __name__ == '__main__':
    '''
    fun_class = Resenbrock()
    x0 = np.array([-1.5,-1.5])
    bfgs = BFGS()
    x_argmin,f_min,trace = bfgs.minimize(fun_class.fun,x0,return_trace=True)
    print(x_argmin)
    print(f_min)
    print(trace)
    '''

    x0 = np.array([10, 10])
    bfgs = BFGS()
    x_argmin, f_min,trace = bfgs.minimize(func_1, x0,return_trace=True)
    print(x_argmin)
    print(f_min)
    print(trace)
