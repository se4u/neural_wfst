# Filename: test_neural_lib_simplification.py
# Description:
# Author: Pushpendre Rastogi
# Created: Sun Apr 12 19:36:14 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 22

# Commentary:
#
from theano import tensor as T
import theano
from neural_lib_simplification import _th_logsumexp
from numpy import log, exp
import numpy as np

def approx_equal(m1, m2):
    dims = (m1.ndim, m2.ndim)
    if dims[0] != dims[1]:
        raise Exception("Dont know what to do")
    if dims == (0,0):
        return abs(m1 - m2) < 1e-10
    elif dims == (1, 1):
        order = 2
    else:
        order = 'fro'
    return np.linalg.norm(m1 - m2, order) < 1e-10

def test_th_logsumexp():
    x = theano.shared(np.array([[1, 2],
                                [3, 4]]))
    # Test 1
    actual_out = theano.function(inputs=[], outputs=_th_logsumexp(x, axis=0))()
    expected_out = np.array([log(exp(1) + exp(3)), log(exp(2) + exp(4))])
    assert approx_equal(actual_out, expected_out)
    # Test 2
    actual_out = theano.function(inputs=[], outputs=_th_logsumexp(x, axis=1))()
    expected_out = np.array([log(exp(1) + exp(2)),
                             log(exp(3) + exp(4))]).T
    assert approx_equal(actual_out, expected_out)
    y = theano.shared(np.array([[[1, 0], [2, 0]],
                                [[3, 0], [4, 0]]]))
    actual_out = theano.function(inputs=[], outputs=_th_logsumexp(y, axis=2))()
    expected_out = np.array([[log(exp(1) + exp(0)), log(exp(2) + exp(0))],
                             [log(exp(3) + exp(0)), log(exp(4) + exp(0))]])
    assert approx_equal(actual_out, expected_out)

if __name__ == "__main__":
    test_th_logsumexp()
