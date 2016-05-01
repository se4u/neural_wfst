#!/usr/bin/env python
'''
| Filename    : transducer_minimal.py
| Description : A minimal SGD based trainer for transducer.
| Author      : Pushpendre Rastogi
| Created     : Mon Nov 30 20:44:55 2015 (-0500)
| Last-Updated: Wed Dec 16 01:27:12 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 241
'''
import sys

from src.transducer import Transducer
from contextlib import contextmanager
from scipy.optimize import fmin_l_bfgs_b
import fst

import code
import numpy
import numpy as np
import itertools
import warnings
INSERTION_LIMIT = 3
vocsize = 5
endpoint = Transducer(vocsize, INSERTION_LIMIT)
SUBSTITUTION = 0
DELETION = 1
INSERTION = 2
def fT(l=2):
    a = - numpy.ones((l, vocsize, vocsize, 3)).astype('float64')
    a[:, :, :, INSERTION] -= 1
    return a
x = numpy.array([1], dtype='int32')
# NOTE: The transducer breaks ties arbitrarily.
# #---------------------------#
# # Demonstrate Substitution. #
# #---------------------------#
# T = fT()
# for loc in [0, 1]:
#     for lc in [0, 1]:
#         for uc in [2]:
#             T[loc, lc, uc, SUBSTITUTION] = 1
# # print endpoint.enumerate_func(x, y, T, ATOL=0.0001)
# # print 'cost', endpoint.func(x, y, T)
# print 'Substitution of lc=[0,1] with uc=2', endpoint.decode(x, T, V=0)
# #-----------------------------------------------------------------------#
# # Demonstrate the gradient of non-useful parameters. (For Substitution) #
# #-----------------------------------------------------------------------#
# for y in [numpy.array([2], dtype='int32'), numpy.array([], dtype='int32')]:
#     grad = numpy.array(endpoint.grad(x, y, T))
#     assert numpy.all(grad[:, :, 0, SUBSTITUTION] == 0)
#     assert numpy.all(grad[:, :, 0, SUBSTITUTION] == 0)

# #-----------------------#
# # Demonstrate Deletion. #
# #-----------------------#
# T = fT()
# for loc in [0, 1]:
#     for lc in [0, 1]:
#         for uc in [0,]:
#             T[loc, lc, uc, DELETION] = 1
# warnings.warn('For the deletion action only uc=0 is meaningful.')
# print 'Deletion of lc=[0,1] with uc=0?', endpoint.decode(x, T, V=0)
# #-------------------------------------------------------------------#
# # Demonstrate the gradient of non-useful parameters. (For Deletion) #
# #-------------------------------------------------------------------#
# for y in [numpy.array([2], dtype='int32'), numpy.array([], dtype='int32')]:
#     grad = numpy.array(endpoint.grad(x, y, T))
#     assert numpy.all(grad[:, :, 1, DELETION] == 0)
#     assert numpy.all(grad[:, :, 2, DELETION] == 0)
# #------------------------#
# # Demonstrate Insertion. #
# #------------------------#
# T = fT()
# for loc in [0, 1]:
#     for lc in [0, 1]:
#         for uc in [2,]:
#             T[loc, lc, uc, INSERTION] = 1
# print 'Insertion at lc=[0,1] of uc=2', endpoint.decode(x, T, V=0)
# #--------------------------------------------------------------------#
# # Demonstrate the gradient of non-useful parameters. (For Insertion) #
# #--------------------------------------------------------------------#
# for y in [numpy.array([2], dtype='int32'), numpy.array([], dtype='int32')]:
#     grad = numpy.array(endpoint.grad(x, y, T))
#     assert numpy.all(grad[:, :, 0, INSERTION] == 0)
#     assert numpy.all(grad[:, :, 0, INSERTION] == 0)

# #---------------------------------#
# # Demonstrate how to copy string. #
# #---------------------------------#
x = numpy.array([1, 2, 1], dtype='int32')


T = fT(l=len(x)+1)
for loc in range(1, 4):
     prev_out_char = 0 if loc == 1 else x[loc-2]
     cur_out_char = x[loc - 1]
     T[loc, prev_out_char, cur_out_char, SUBSTITUTION] = 10
     print (loc, prev_out_char, cur_out_char, SUBSTITUTION)

T = np.random.rand(len(x)+1, vocsize, vocsize, 3)
#T = np.ones_like(T) 
print 'Test case demonstrating the impossibility of copying'
#print 'Copy string', x, '->', endpoint.decode(x, T, V=0)


print endpoint.decode(x, T, V=True)
#print endpoint.sample(x, T, 1000000)
print endpoint.enumerate_func(x, x, T, ATOL=0.0001)

# invert the alphabet dictionary
Sigma_inv = {}
for _ in range(vocsize):
    Sigma_inv[_] = str(_)


result = endpoint.to_openfst(x, T)

decoded = []
score = fst.TropicalWeight.ONE
for path in result.shortest_path().paths():
    for arc in path:
        score *= arc.weight
        if arc.ilabel > 0:
            decoded.append(arc.ilabel)
print score
print result.shortest_distance(True)[0]
print decoded
print "CRUNCH 1"
print endpoint.crunch(x, T, n=1, V=True)
print "CRUNCH 10"
print endpoint.crunch(x, T, n=10, V=True)
