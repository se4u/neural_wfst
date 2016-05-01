# Filename: test_autograd.py
# Description:
# Author: Pushpendre Rastogi
# Created: Mon Apr 13 18:36:26 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 51

# Commentary:
# Test autograd's features.
# from RectangleDependencyParser import logsum
import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check

def say_my_name(fun):
    def ff(*args, **kwargs):
        print fun.__name__
        return fun(*args, **kwargs)
    return ff

@say_my_name
def test_logsum():
    x = np.random.rand(3,3)
    f = lambda x: np.sum(logsum(x, axis=0))
    gradf = grad(f)
    print gradf(x), x
    print gradf(x)
    return

def assignment(y, x):
    for i in range(3):
        x['a'] = np.sum(np.array([y[i], x['a']]))
    return x['a']

@say_my_name
def test_assignment():
    def f(y):
        x = dict(a=1)
        return assignment(y, x)
    y = np.array([1.0,2.0,3.0])
    quick_grad_check(f, y, verbose=False)
    return

if __name__ == "__main__":
    # try:
    # test_logsum()
    test_assignment()
    # except:
    #     import sys, traceback, pdb
    #     type, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
