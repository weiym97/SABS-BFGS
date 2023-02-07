"""Tests for the BFGS class"""
import math
import numpy as np
import numpy.testing as npt

def test_univariate_function():
    '''Test for univariate function, f(x) = x^2'''
    from SABS_BFGS.bfgs import BFGS
    bfgs = BFGS()
    x0 = 1.0
    x_argmin, f_min = bfgs.minimize(lambda x: x**2, x0)
    npt.assert_almost_equal(x_argmin,0.0,decimal=8)
    npt.assert_almost_equal(f_min,0.0,decimal=8)

def test_rosenbrock_function():
    from SABS_BFGS.bfgs import BFGS
    bfgs = BFGS()
    x0=[2.0,3.0,4.0]
    def func(x):
        return (x[0] - 1) ** 2 + x[1] ** 2 + (x[2] ** 2 - 2 * x[0]) ** 2
    def Jacobi(x):
        return [2 * (x[0] - 1) - 4 * (x[2] ** 2 - 2 * x[0]),2 * x[1], 4 * x[2] * (x[2] ** 2 - 2 * x[0])]
    x_argmin,fmin = bfgs.minimize(func,x0)
    npt.assert_almost_equal(x_argmin[0],1.0, decimal=8)
    npt.assert_almost_equal(np.abs(x_argmin),[1.0,0.0,math.sqrt(2)],decimal=8)
    npt.assert_almost_equal(fmin,0.0,decimal=8)

    x_argmin, fmin = bfgs.minimize(func, x0,jac=Jacobi)
    npt.assert_almost_equal(x_argmin[0], 1.0, decimal=8)
    npt.assert_almost_equal(np.abs(x_argmin), [1.0, 0.0, math.sqrt(2)], decimal=8)
    npt.assert_almost_equal(fmin, 0.0, decimal=8)

