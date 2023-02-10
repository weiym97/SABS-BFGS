import autograd.numpy as np
from autograd import grad
import scipy.optimize


class BFGS():
    '''
    Global otimisation using BFGS methods.
    Practice in Modelling and Scientific Computing Module.
    Written by Yiming Wei 2023-02-09.
    '''

    def minimize(self, fun, x0, jac=None, tol=1e-8, return_trace = False):

        '''
        The method to look for the global minimum
        :param fun: callable f(x).
                    Objective function.
        :param x0: ndarray.
                   Starting point.
        :param jac: callable vector function f'(x).
                    the first order derivative of the objective function.
                    If unspecified, we use autograd
        :param args:
        :param tol: float.
                    stop iterating when ||f'(x)||<tol
        :param return_trace:bool.
                            whether to return the trace of optimization.
        :return:
        -------
        x: ndarray.
           The global minimal point.
        y: float.
           The minimum of f(x), evalutated at x.
        trace: dictionary.
           Tracing back the process of optimization, include x_k,p_k,alpha_k,s_k.
           See introduction.ipynb for details and visualisation.

        -------
        Example:
        import numpy as np
        from SABS_BFGS.bfgs import BFGS
        bfgs = BFGS()
        x0 = np.array([1.0])
        x_argmin, f_min = bfgs.minimize(lambda x: x**2, x0)
        '''
        jac,x,H,jac_old = self.initialize(fun, jac, x0)

        if return_trace:
            trace = self.initialize_trace()

        while True:
            # Itegrate over to get the next state
            p,alpha,s,jac_new,y = self.iteration(fun, jac, x, jac_old, H)
            if return_trace:
                trace = self.update_trace(trace, x, p, alpha, s)

            # Update x, H,jac_old for the next step
            x,H,jac_old = self.update_state(x, H, y, s, jac_new)

            # If the norm of jacobi is smaller than the threshold, then break
            if  np.linalg.norm(jac_old, ord=2) < tol:
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
