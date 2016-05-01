# Filename: test_pdp_script.py
# Description:
# Author: Pushpendre Rastogi
# Created: Sun May 24 04:12:07 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 32

import numpy
numpy.random.seed(100)
n = 20
arc_scores = -numpy.random.rand(n,n,1)*2
# Slow results
from depparse import RectangleDependencyParser
grad_slow, alpha_slow, beta_slow, logz_slow = RectangleDependencyParser.DependencyParser().insideOutside_expressive(arc_scores)

# Fast results
from depparse.RectangleDependencyParser_cython import  dp_insideOutside_expressive
grad_fast, alpha_fast, beta_fast, logz_fast = dp_insideOutside_expressive(arc_scores)
beta_fast = numpy.asarray(beta_fast)
alpha_fast = numpy.asarray(alpha_fast)
def tolerant_eq(a, b):
    return (True
            if (numpy.isinf(a) and numpy.isinf(b))
            or (numpy.isneginf(a) and numpy.isneginf(b))
            or abs(a-b) < 1e-10
            else False)
assert tolerant_eq(logz_slow, logz_fast), (logz_slow, logz_fast)
assert all([tolerant_eq(a, b) for a, b in zip(beta_slow.flatten(), beta_fast.flatten())])
assert all([tolerant_eq(a, b) for a, b in zip(alpha_slow.flatten(), alpha_fast.flatten())])
print 'test passed'
