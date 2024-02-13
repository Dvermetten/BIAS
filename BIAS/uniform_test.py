import numpy as np
from scipy.stats import uniform
import ctypes
from numpy.ctypeslib import ndpointer

def ddst_phi(x, j, base=None):
    if base is None:
        base = np.polynomial.legendre.legval
    return np.mean(base(x, j))


def ddst_uniform_Nk(x, base=None, Dmax=10):
    n = len(x)
    maxN = max(min(Dmax, n-2, 20), 1)
    coord = np.zeros(maxN)
    for j in range(1, maxN+1):
        coord[j-1] = ddst_phi(x, j, base)
    coord = np.cumsum(coord**2 * n)
    return coord

def ddst_IIC(coord, n, c=2.4):
    IIC = float(np.max([coord[0], np.max(np.diff(coord))]) < c * np.log(n))
    return np.argmax(coord - (IIC * np.log(n) + (1 - IIC) * 2) * np.arange(1, len(coord) + 1))


def ddst_uniform_test(x, base=None, c=2.4, B=1000, compute_p=False, Dmax=10, *args):
    # base is np.polynomial.legendre.legval
    method_name = "ddst.base.legendre"
    n = len(x)
    if n < 5:
        raise ValueError("length(x) should be at least 5")
    coord = ddst_uniform_Nk(x, base, Dmax=Dmax)  # Assuming ddst_uniform_Nk is defined elsewhere
    l = ddst_IIC(coord, n, c)  # Assuming ddst_IIC is defined elsewhere
    t = coord[l]
    result = {'statistic': t, 'parameter': l, 'method': "Data Driven Smooth Test for Uniformity"}
    result['data_name'] = f"{str(x)},   base: {method_name}   c: {c}"
    if compute_p:
        tmp = np.zeros(B)
        for i in range(B):
            y = uniform.rvs(size=n)
            tmpC = ddst_uniform_Nk(y, base, Dmax=Dmax)
            l = ddst_IIC(tmpC, n, c)
            tmp[i] = tmpC[l]
        result['p_value'] = np.mean(tmp > t)
    return result




