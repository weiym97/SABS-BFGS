import autograd.numpy as np
from autograd import grad

class GradientDescent():
    def minimize(self, fun, x0, alpha=1e-3,jac=None, tol=1e-8,iter_upper=100000,return_trace = False):
        '''
        This is a class to implement gradient descent for optimisation.
        :param fun:
        :param x0:
        :param jac:
        :param tol:
        :param return_trace:
        :return:
        '''
        jac, x,  = self.initialize(fun,jac,x0)
        trace = [x]
        iter_times = 0
        while True:
            jac_new = jac(x)
            if (np.linalg.norm(jac_new,ord=2) < tol) or (iter_times>iter_upper):
                break
            x = x - alpha * jac_new
            trace.append(x)
        if return_trace:
            return x,fun(x),{'x_k':trace,'p_k':[],'alpha_k':[],'s_k':[]}
        else:
            return x,fun(x)

    def initialize(self, fun, jac, x0):
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
        x = x0

        return jac, x