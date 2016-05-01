#!/usr/bin/env python
# -*- coding: utf-8 -*-
##cython: boundscheck=False
##cython: wraparound=False
##cython: nonecheck=False
##cython: cdivision=True
##cython: infertypes=True
##cython: c_string_type=unicode, c_string_encoding=ascii
##cython: profile=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

import numpy as np
import numpy.random as npr
cimport numpy as np
#from numpy.math cimport logaddexp
from numpy import zeros, ones, empty
from libc.math cimport log, exp, log1p
from libc.float cimport DBL_MIN, DBL_MAX
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from cython.operator cimport dereference as deref, preincrement as inc
from arsenal.alphabet import Alphabet
from collections import defaultdict as dd
import fst

ctypedef vector[vector[vector[vector[vector[int]]]]] FEATURES
ctypedef vector[vector[vector[vector[vector[vector[int]]]]]] FEATURES_ALL

DEF MIN = -128.0
DEF MAX = 0.0
DEF WID = MAX-MIN
DEF TBL = 65536
DEF INC = TBL*1.0/WID

cdef inline double logaddexp(double x, double y) nogil:
    """
    Needs to be rewritten
    """
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y


cdef class Transducer:
    """
    Conditional Random Field Weighted Finite-State Transducer (CRF-WFST).

    See Dreyer et al. (2008) for more details. 
    """

    cdef int Sigma_size, insertion_limit
    cdef FEATURES_ALL extracted_all
    cdef object features
    cdef inline double[TBL+2] _tbl

 
    def __init__(self, Sigma_size, insertion_limit):
        self.Sigma_size = Sigma_size
        self.insertion_limit = insertion_limit
        #self.features = features
        #self.extracted_all = self.features.extracted

        # need the +2 for linear extrapolation
        cdef int i
        for i in range(TBL+2):
            self._tbl[i] = logaddexp(0, i / INC + MIN)


    cdef inline double logadd(self, double a, double b) nogil:
        cdef:
            int i
            double x
        if b > a:
            a, b = b, a
        if a <= -DBL_MAX:
            return b
        if b <= -DBL_MAX:
            return a
        if b - a < MIN:   # function behaves linearly after this point.
            return a
        x = ((((b - a) - MIN) * INC))
        i = <int>x    # round down
        return a + self._tbl[i] + (x - i)*(self._tbl[i+1] - self._tbl[i])


    cdef inline void _convert(self, int[:] string1, int string1_size, double[:] theta, double[:, :, :, :] W, int strid, bool to_W) nogil:
        " Convert from feature weights to arc weights "

        cdef FEATURES *extracted = &self.extracted_all[strid]

        cdef int i, j, x, y, featid
        cdef vector[int].iterator it
        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue

                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        it = extracted[0][i][x][0][1].begin()
                        while it != extracted[0][i][x][0][1].end():
                            featid = deref(it)
                            if to_W == True:
                                W[i, x, 0, 1] += theta[featid]
                            else:
                                theta[featid] += W[i, x, 0, 1]
                            inc(it)
        
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        it = extracted[0][i][x][y][0].begin()
                        while it != extracted[0][i][x][y][0].end():
                            featid = deref(it)
                            if to_W == True:
                                W[i, x, y, 0] += theta[featid]
                            else:
                                theta[featid] += W[i, x, y, 0]
                            inc(it)

                        # insertion
                        if j > 0:
                            it = extracted[0][i][x][y][2].begin()

                            while it != extracted[0][i][x][y][2].end():
                                featid = deref(it)
                                if to_W == True:
                                    W[i, x, y, 2] += theta[featid]
                                else:
                                    theta[featid] += W[i, x, y, 2]
                                inc(it)
        
        
    cpdef double[:, :, :, :] convert_to_W(self, string1, strid, theta, features):
        " Converts to from theta to W "

        cdef double[:, :, :, :] W = zeros((len(string1)+1, self.Sigma_size, self.Sigma_size, 3))
        self._convert(string1, len(string1), theta, W, strid, True)
        return W


    cpdef double[:] convert_to_theta(self, string1, strid, W, theta, features):
        " Converts to from theta to W "

        self._convert(string1, len(string1), theta, W, strid, False)
        return theta
        

    def func_features(self, string1, string2, strid, theta, features, threshold=10.0):
        " Function with respect to the feature vector "

        cdef double[:, :, :, :] W = self.convert_to_W(string1, strid, theta, features) 
        return self._func(len(string1), string1, len(string2), string2, W, threshold)


    def grad_features(self, string1, string2, strid, theta, theta_grad, features, threshold=10.0):
        " Gradient with respect to the feature vector "

        cdef double[:, :, :, :] W = self.convert_to_W(string1, strid, theta, features)
        cdef double[:, :, :, :] W_grad = self._gradient(len(string1), string1, len(string2), string2, W, threshold)
        return self.convert_to_theta(string1, strid, W_grad, theta_grad, features)

    def func(self, string1, string2, W, threshold=10.0):
        " Python wrapper method for calculating the value of the function "
        return self._func(len(string1), string1, len(string2), string2, W, threshold)


    cdef double _func(self, int string1_size, int[:] string1, int string2_size, int[:] string2,  double[:, :, :, :] W, double threshold):
        " Calculates the value of the function "

        # initializes the matrices for the dynamic program
        cdef double [:, :] B_observed = empty((string1_size+1, string2_size+1))
        cdef double[:, :, :] B_expected = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef int[:, :, :, :] mask = self._pruning_mask(string1_size, W, threshold)
        cdef int i, j, k

        # sets initial values
        for i in xrange(string1_size+1):
            for j in xrange(string2_size+1):
                B_observed[i, j] = -DBL_MAX
        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    if i == string1_size:
                        B_expected[i, j, k] = 0.0
                    else:
                        B_expected[i, j, k] = -DBL_MAX

        B_observed[string1_size, string2_size] = 0.0

        # populates matrices
        self.backward_observed(string1_size, string2_size, string2, W, B_observed)
        self.backward_expected(string1_size, W, B_expected, mask)

        # gets respective partition functions
        cdef double Z_observed = B_observed[0, 0]
        cdef double Z_expected = B_expected[0, 0, 0]

        # computes value as -log of the ration of the two partition functions
        return -Z_observed + Z_expected


    def grad(self, string1, string2, W, threshold=10.0):
        " Python wrapper method for calculating the gradient "
        
        return self._gradient(len(string1), string1, len(string2), string2, W, threshold)

        
    cdef double[:, :, :, :] _gradient(self, int string1_size, int[:] string1, int string2_size, int[:] string2,  double[:, :, :, :] W, double threshold):
        " Calculates the gradient - first the observed counts than the expected counts "

        cdef double[:, :, :, :] W_grad = zeros((string1_size+1, self.Sigma_size, self.Sigma_size, 3))
        
        self.gradient_expected(string1_size, W, W_grad, threshold)
        self.gradient_observed(string1_size, string2_size, string2, W, W_grad)
        
        return W_grad


    cdef void gradient_observed(self, int string1_size, int string2_size, int[:] string2, double[:, :, :, :] W, double[:, :, :, :] W_grad):
        """
        Calculates the observed counts -- the training data's contribution to the gradienet

        Note: counts are *subtracted* from the gradient vector
        """

        cdef int i, j, x, y
        cdef double [:, :] B = empty((string1_size+1, string2_size+1))
        cdef double[:, :] F = empty((string1_size+1, string2_size+1))

        for i in xrange(string1_size+1):
            for j in xrange(string2_size+1):
                B[i, j] = -DBL_MAX
        B[string1_size, string2_size] = 0.0

        for i in xrange(string1_size+1):
            for j in xrange(string2_size+1):
                F[i, j] = -DBL_MAX
        F[0, 0] = 0.0

        self.backward_observed(string1_size, string2_size, string2, W, B)
        cdef double logZ = B[0, 0]

        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string2_size+1, 1):
                if i == 0 and j == 0:
                    continue

                # substitution
                if i > 0 and j > 0:
                    x = 0 if j < 2 else string2[j-2]
                    y = string2[j-1]

                    F[i, j] = self.logadd(F[i, j], W[i, x, y, 0] + F[i-1, j-1])
                    W_grad[i, x, y, 0] -= exp(F[i-1, j-1]+W[i, x, y, 0]+B[i, j]-logZ)
                # deletion
                if i > 0:
                    x = 0 if j < 1 else string2[j-1]

                    F[i, j] = self.logadd(F[i, j], W[i, x, 0, 1] + F[i-1, j])
                    W_grad[i, x, 0, 1] -= exp(F[i-1, j]+W[i, x, 0, 1]+B[i, j]-logZ)
                # insertion
                if j > 0:
                    x = 0 if j < 2 else string2[j-2]
                    y = string2[j-1]

                    F[i, j] = self.logadd(F[i, j], W[i, x, y, 2] + F[i, j-1])
                    W_grad[i, x, y, 2] -= exp(F[i, j-1]+W[i, x, y, 2]+B[i, j]-logZ)


    cdef void gradient_expected(self, int string1_size, double[:, :, :, :] W, double[:, :, :, :] W_grad, double threshold):
        """
        Calculates the expected counts -- the log partition function's contribution to the gradient

        Note: counts are *added* to the gradient vector
        """

        cdef double[:, :, :] B = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef double[:, :, :] F = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
                
        cdef int i, j, k, x, y
        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    if i == string1_size:
                        B[i, j, k] = 0.0
                    else:
                        B[i, j, k] = -DBL_MAX
                    F[i, j, k] = -DBL_MAX
        F[0, 0, 0] = 0.0

        cdef int[:, :, :, :] mask = self._pruning_mask(string1_size, W, threshold)
        self.backward_expected(string1_size, W, B, mask)
        cdef double logZ = B[0, 0, 0]

        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue

                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        if mask[i, x, 0, 1] == 1:
                            F[i, j, x] = self.logadd(F[i, j, x], W[i, x, 0, 1] + F[i-1, j, x])
                            W_grad[i, x, 0, 1] += exp(F[i - 1, j, x] + W[i, x, 0, 1] + B[i, j, x] - logZ)
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            if mask[i, x, y, 0] == 1:
                                F[i, j, y] = self.logadd(F[i, j, y], W[i, x, y, 0] + F[i-1 ,j-1, x])
                                W_grad[i, x, y, 0] += exp(F[i-1, j-1, x] + W[i, x, y, 0] +  B[i, j, y] - logZ)
                        # insertion
                        if j > 0:
                            if mask[i, x, y, 2] == 1:
                                F[i, j, y] = self.logadd(F[i, j, y], W[i, x, y, 2] + F[i, j-1, x])
                                W_grad[i, x, y, 2] += exp(F[i, j-1, x] + W[i, x, y, 2] + B[i, j, y] - logZ)

    cpdef sample(self, string1, double[:, :, :, :] W, int N):
        " Sample some strings "
        from collections import defaultdict as dd
        samples = dd(int)
        
        cdef int string1_size = len(string1)
        cdef double[:, :, :] B = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef double[:, :, :] F = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
                
        cdef int i, j, k, x, y, n, counter
        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    if i == string1_size:
                        B[i, j, k] = 0.0
                    else:
                        B[i, j, k] = -DBL_MAX
                    F[i, j, k] = -DBL_MAX
        F[0, 0, 0] = 0.0

        cdef int[:, :, :, :] mask = self._pruning_mask(string1_size, W, 0.0)

        self.backward_expected(string1_size, W, B, mask)
        cdef double globalLogZ = B[0, 0, 0]

        cdef double logZ, p, q, sampled
        cdef vector[int] sample
        
        cdef double score = 0.0
        # acts as a form of rejection sampling when the probabilities are small
        cdef double renorm = 1.0
        n, i, j = 0, 0, 0
        
        for n in xrange(N):
            i, j, x, = 0, 0, 0
            sample = vector[int]()
            score = 0.0
            counter = 0
            while i < string1_size and j < string1_size + self.insertion_limit:
                counter += 1
                if counter == 100:
                    break
                #print "here"
                
                # compute normalizer
                logZ = -DBL_MAX
                # deletion
                logZ = self.logadd(logZ, W[i+1, x, 0, 1] + B[i+1, j, 0])
                #print i+1, 0, j, B[i+1, j, 0]
                for y in xrange(1, self.Sigma_size):
                    # substitution
                    logZ = self.logadd(logZ, W[i+1, x, y, 0] + B[i+1, j, y])
                    #print i+1, j, y, B[i+1, j, y]
                    # insertion
                    logZ = self.logadd(logZ, W[i, x, y, 2] + B[i, j, y])
                    #print i, j, y, B[i, j, y]

                # sample a random number in [0, 1]
                sampled = npr.rand()

                # deletion
                q = exp(W[i+1, x, 0, 1] + B[i+1, j, x] - logZ) / renorm
                p = q
                if p >= sampled:
                    score += W[i+1, x, 0, 1]
                    i = i+1
                    
                else:
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        q = exp(W[i+1, x, y, 0] + B[i+1, j, y] - logZ) / renorm
                        p += q                        
                        if p >= sampled:
                            #print "SUB(" + Sigma_inv[string1[i]] + "," + Sigma_inv[y] + ")", q
                            score += W[i+1, x, y, 0]
                            i, j, x = i+1, j+1, y
                            sample.push_back(y)
                            break
                        
                        # insertion
                        q = exp(W[i, x, y, 2] + B[i, j, y] - logZ) / renorm
                        p += q 
                        if p >= sampled:
                            score += W[i, x, y, 2]
                            j += 1
                            x = y

                            sample.push_back(y)
                            break

                    if p < sampled:
                        renorm = p

            samples[tuple(sample)] += 1

        max_count = 0
        max_string = None

        for a, b in dict(samples).items():
            if b > max_count:
                max_count = b
                max_string = a
        return max_string

    
    def decode(self, string1, W, V=False):
        " Decode based on an input string and a parameter vector "

        string1_size = len(string1)
        #cdef double[:, :, :] B = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        #cdef int[:, :, :, :] mask = self._pruning_mask(string1_size, W, 0.0)
        #self.backward_expected(string1_size, W, B, mask)
        #logZ = B[0, 0, 0]

        
        score, string = self._decode(len(string1), W)


        if V:
            # get prob, not just score
            #string1_size = len(string1)
            #B = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
            #mask = self._pruning_mask(string1_size, W, 0.0)
            #self.backward_expected(string1_size, W, B, mask)
            #logZ = B[0, 0, 0]
            return score, string
        return string


    def decode_features(self, string1, strid, theta, features, V=False):
        " Decode based on an input string and a parameter vector "

        cdef double[:, :, :, :] W = self.convert_to_W(string1, strid, theta, features)
        score, string = self._decode(len(string1), W)
        if V:
            return score, string
        return string


    def forward_tropical_features(self, string1, strid, theta, features, threshold=1.0):
        " Function with respect to the feature vector "

        cdef double[:, :, :, :] W = self.convert_to_W(string1, strid, theta, features)
        mask = self._pruning_mask(len(string1), W, threshold)
        
        #counter = 0
        #for i in xrange(len(string1)+1):
        #    for j in xrange(self.Sigma_size):
        #        for k in xrange(self.Sigma_size):
        #            for l in xrange(3):
        #                if mask[i, j, k, l]  == 1:
        #                    counter +=1 


    cdef int[:, :, :, :] _pruning_mask(self, int string1_size, double[:, :, :, :] W, double threshold):
        " gets the pruning mask based on the max-marginals "

        cdef double[:, :, :] V_forward = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef double[:, :, :] V_backward = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        # TODO: FIX!
        cdef int[:, :, :, :] mask = ones((string1_size+1, self.Sigma_size, self.Sigma_size, 3), dtype=np.int32)

        cdef int i, j, k, x, y

        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    V_forward[i, j, k] = -DBL_MAX
        V_forward[0, 0, 0] = 0.0

        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    if i == string1_size:
                        V_backward[i, j, k] = 0.0
                    else:
                        V_backward[i, j, k] = -DBL_MAX

        # go forward and backward
        self._forward_tropical(string1_size, W, V_forward)
        self._backward_tropical(string1_size, W, V_backward)

        # TODO: put this somewhere else?
        # cdef double max_val = -DBL_MAX
        # for j in xrange(string1_size+1+self.insertion_limit):
        #     for k in xrange(self.Sigma_size):
        #         if V_forward[string1_size, j, k] >= max_val:
        #             max_val = V_forward[string1_size, j, k]
        # assert abs(max_val - V_backward[0, 0, 0]) < 0.01

        cdef double max_Z = V_backward[0, 0, 0]
        cdef double max_marginal = 0.0

        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue
                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        max_marginal = V_forward[i-1, j, x] + W[i, x, 0, 1] + V_backward[i, j, x]
                        if abs(max_marginal - max_Z) <= threshold:
                            mask[i, x, 0, 1] = 1

                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            max_marginal = V_forward[i-1, j-1, x] + W[i, x, y, 0] + V_backward[i, j, y]
                            if abs(max_marginal - max_Z) <= threshold:
                                mask[i, x, y, 0] = 1
                        # insertion
                        if j > 0:
                            max_marginal = V_forward[i, j-1, x] + W[i, x, y, 2] + V_backward[i, j, y]
                            if abs(max_marginal - max_Z) <= threshold:
                                mask[i, x, y, 2] = 1

        return mask
        

    cdef inline void _forward_tropical(self, int string1_size, double[:, :, :, :] W, double[:, :, :] V):
        " The forward algorithm over the tropical semiring " 

        cdef int i, j, k, x, y
        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue
                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        V[i, j, x] = max(V[i, j, x], W[i, x, 0, 1] + V[i-1, j, x])
                         
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            V[i, j, y] = max(V[i, j, y], W[i, x, y, 0] + V[i-1, j-1, x])
                        # insertion
                        if j > 0:
                            V[i, j, y] = max(V[i, j, y], W[i, x, y, 2] + V[i, j-1, x])


    cdef inline void _backward_tropical(self, int string1_size, double[:, :, :, :] W, double[:, :, :] V):
        " The backward algorithm over the tropical semiring "

        cdef int i, j, k, x, y
        for i in xrange(string1_size, -1, -1):
            for j in xrange(string1_size+self.insertion_limit, -1, -1):
                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i-1 >= 0:
                        V[i-1, j, x] = max(V[i-1, j, x], W[i, x, 0, 1] + V[i, j, x])
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i-1 >= 0 and j-1 >= 0:
                            V[i-1, j-1, x] = max(V[i-1, j-1, x], W[i, x, y, 0] + V[i, j, y])
                        # insertion
                        if j-1 >= 0:
                            V[i, j-1, x] = max(V[i, j-1, x], W[i, x, y, 2] + V[i, j, y])



    cdef pair[double, vector[int]] _decode(self, int string1_size, double[:, :, :, :] W):
        """
        Decodes the model using Viterbi. It finds the highest scoring alignment
        path through the lattice and results its yield (a string). 

        TODO: Rather inefficient (i.e., not Cython)
        """
        cdef double[:, :, :] V = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef int [:, :, :, :] bp = zeros((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size, 4), dtype=np.int32)

        cdef int i, j, k, x, y
        cdef double val

        for i in xrange(string1_size+1):
            for j in xrange(string1_size+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    V[i, j, k] = -DBL_MAX

        V[0, 0, 0] = 0.0
        bp[0, 0, 0, 0] = 0 
        bp[0, 0, 0, 1] = 0 
        bp[0, 0, 0, 2] = 0 
        bp[0, 0, 0, 3] = 0 
        
        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue

                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        val = W[i, x, 0, 1] + V[i-1, j, x]
                        if val >= V[i, j, x]:
                            V[i, j, x] = val
                            bp[i, j, x, 0] = i-1
                            bp[i, j, x, 1] = j
                            bp[i, j, x, 2] = x
                            bp[i, j, x, 3] = 0

                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            val = W[i, x, y, 0] + V[i-1, j-1, x]
                            if val >= V[i, j, y]:
                                V[i, j, y] = val
                                bp[i, j, y, 0] = i-1
                                bp[i, j, y, 1] = j-1
                                bp[i, j, y, 2] = x
                                bp[i, j, y, 3] = y

                        # insertion
                        if j > 0:
                            val = W[i, x, y, 2] + V[i, j-1, x]
                            if val >= V[i, j, y]:
                                V[i, j, y] = val
                                bp[i, j, y, 0] = i
                                bp[i, j, y, 1] = j-1
                                bp[i, j, y, 2] = x
                                bp[i, j, y, 3] = y
                                                     
        # get best derivation
        cdef int best_j, best_k
        best_j, best_k = 0, 0
        cdef double max_val = -DBL_MAX
        for j in xrange(string1_size+1+self.insertion_limit):
            for k in xrange(self.Sigma_size):
                if V[string1_size, j, k] >= max_val:
                    best_j, best_k = j, k
                    max_val = V[string1_size, j, k]

        # follow the backpointers
        i, j, k = string1_size, best_j, best_k
        cdef vector[int] decoded = vector[int](best_j)
        while i > 0 or j > 0:
            i, j, k, y = bp[i, j, k, 0], bp[i, j, k, 1], bp[i, j, k, 2], bp[i, j, k, 3]
            if y != 0:
                decoded[j] = y

        return pair[double, vector[int]](V[string1_size, best_j, best_k], decoded)


    cdef inline void backward_observed(self, int string1_size, int string2_size, int[:] string2, double[:, :, :, :] W, double [:, :] B):
        " Populates the backward matrix for the observed count computation "

        cdef int i, j, x, y

        for i in xrange(string1_size, -1, -1):
            for j in xrange(string2_size, -1, -1):
                # substitution
                if i-1 >= 0 and j-1 >= 0:
                    x = 0 if j-2 < 0 else string2[j-2]
                    y = string2[j-1]
                    B[i-1, j-1] = self.logadd(B[i-1, j-1], W[i, x, y, 0] + B[i, j])
                # deletion
                if i-1 >= 0:
                    x = 0 if j-1 < 0 else string2[j-1]
                    B[i-1, j] = self.logadd(B[i-1, j], W[i, x, 0, 1] + B[i, j])
                # insertion
                if j-1 >= 0:
                    x = 0 if j-2 < 0 else string2[j-2]
                    y = string2[j-1]
                    B[i, j-1] = self.logadd(B[i, j-1], W[i, x, y, 2] + B[i, j])
    

    cdef inline void backward_expected(self, int string1_size, double[:, :, :, :] W, double [:, :, :] B, int[:, :, :, :] mask):
        " Populates the backward matrix for the expected count computation "

        cdef int i, j
        cdef int x, y

        cdef int computed = 0
        cdef int total = 0
        for i in xrange(string1_size, -1, -1):
            for j in xrange(string1_size+self.insertion_limit, -1, -1):
                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i-1 >= 0:
                        total += 1
                        if mask[i, x, 0, 1] == 1:
                            computed += 1
                            B[i-1, j, x] = self.logadd(B[i-1, j, x], W[i, x, 0, 1] + B[i, j, x])
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i-1 >= 0 and j-1 >= 0:
                            total += 1
                            if mask[i, x, y, 0] == 1:
                                computed += 1
                                B[i-1, j-1, x] = self.logadd(B[i-1, j-1, x], W[i, x, y, 0] + B[i, j, y])
                                # insertion
                        if j-1 >= 0:
                            total += 1
                            if mask[i, x, y, 2] == 1:
                                computed += 1
                                B[i, j-1, x] = self.logadd(B[i, j-1, x], W[i, x, y, 2] + B[i, j, y])

        #print computed, total
        #print float(computed) / total

        
    def crunch(self, string1, W, n=1, V=False):
        """ Crunching  """

        #print "STRING1"
        #print string1
        viterbi_score, viterbi_best = self.decode(string1, W, V=True)
        # get partition function
        string1_size = len(string1)
        cdef double[:, :, :] B = empty((string1_size+1, string1_size+1+self.insertion_limit, self.Sigma_size))
        cdef int[:, :, :, :] mask = self._pruning_mask(string1_size, W, 0.0)
        self.backward_expected(string1_size, W, B, mask)
        logZ = B[0, 0, 0]

        machine = self.to_openfst(string1, W)
        table = {}
        for path in machine.shortest_path(n=n).paths():
            decoded = []
            score = fst.TropicalWeight.ONE
            for arc in path:
                score *= arc.weight
                if arc.ilabel > 0:
                    decoded.append(arc.ilabel)
            key = tuple(decoded)
            if key not in table:
                table[key] = -np.inf
            #print "crunched", list(key), -float(score)
            table[key] = logaddexp(table[key], -float(score))
            
        max_v = 0.0
        best = None
        for k, v in table.items():
            if v >= max_v:
                best = k
                max_v = v
        best = list(best)
        #id2label = {1: '^', 2: u'a', 3: u'c', 4: u'b', 5: u'e', 6: u'd', 7: u'g', 8: u'f', 9: u'i', 10: u'h', 11: u'k', 12: u'j', 13: u'm', 14: u'l', 15: u'o', 16: u'n', 17: u'q', 18: u'p', 19: u's', 20: u'r', 21: u'u', 22: u't', 23: u'w', 24: u'v', 25: u'y', 26: u'x', 27: u'z'}
        #print "INPUT", "".join(map(lambda x: id2label[x], string1))
        #print "CRUNCH:", "".join(map(lambda x: id2label[x], best)), max_v
        #print "VITERBI:", "".join(map(lambda x : id2label[x], viterbi_best)), viterbi_score
        #print
        print "CRUNCH", best, exp(max_v - logZ)
        print "VITERBI", viterbi_best, exp(viterbi_score - logZ)
        print
        if V:
            return max_v, best
        return best
            

    def to_openfst(self, string1, W):
        """ 
        The forward algorithm over the tropical semiring 

        TOOD: need a better unit test. 
        Test in transducer_behavior.py shows it works
        """

        string1_size = len(string1)
        state = Alphabet()
        state.add((0, 0, 0))
        
        t = fst.Acceptor()
        t.add_state()
        t.start = 0

        for i in xrange(0, string1_size+1, 1):
            for j in xrange(0, string1_size+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue

                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        val = W[i, x, 0, 1]
                        source = state[(i-1, j, x)]
                        target = state[(i, j, x)]
                        t.add_arc(source, target, fst.EPSILON, -val)

                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            val = W[i, x, y, 0]
                            source = state[(i-1, j-1, x)]
                            target = state[(i, j, y)]
                            t.add_arc(source, target, str(y), -val)

                        # insertion
                        if j > 0:
                            val = W[i, x, y, 2]
                            source = state[(i, j-1, x)]
                            target = state[(i, j, y)]
                            t.add_arc(source, target, str(y), -val)
        # add final states
        for j in xrange(string1_size+1+self.insertion_limit):
            for y in xrange(self.Sigma_size):
                stateid = state[(string1_size, j, y)]
                t[stateid].final = True
                
        return t
                            

    def fd_check(self, string1, string2, W, EPS=0.01, ATOL=0.01):
        " Finite difference check to verify the gradient "

        W_grad = self.grad(string1, string2, W)
        W_approx_grad = zeros((len(string1)+1, self.Sigma_size, self.Sigma_size, 3))

        for i in xrange(len(string1)+1):
            for j in xrange(self.Sigma_size):
                for k in xrange(self.Sigma_size):
                    for l in xrange(3):
                        W[i, j, k, l] += EPS
                        val1 = self.func(string1, string2, W)
                        W[i, j, k, l] -= 2 * EPS
                        val2 = self.func(string1, string2, W)
                        W[i, j, k, l] += EPS
                        W_approx_grad[i,j,k,l] = (val1 - val2) / (2 * EPS)
                        
        assert np.allclose(np.asarray(W_grad), W_approx_grad, atol=ATOL)


    def _join(self, w, new, lst):
        " Multiplication for the derivation semiring "

        new_lst = []
        for z, align in lst:
            new_lst.append((w*z, align+[new]))
        return new_lst


    def enumerate_func(self, string1, string2, W, ATOL=0.01):
        " Checks the log-likelihood by enumeration "

        # initialize DP table
        F = zeros((len(string1)+1, len(string1)+1+self.insertion_limit, self.Sigma_size), dtype='object')
        for i in xrange(len(string1)+1):
            for j in xrange(len(string1)+1+self.insertion_limit):
                for k in xrange(self.Sigma_size):
                    F[i, j, k] = []
        F[0, 0, 0] = [(1.0, [(0, 0)])]

        # perform forward algorithm in the derivation semiring
        for i in xrange(0, len(string1)+1, 1):
            for j in xrange(0, len(string1)+1+self.insertion_limit, 1):
                if i == 0 and j == 0:
                    continue

                for x in xrange(0, self.Sigma_size):
                    # deletion
                    if i > 0:
                        F[i, j, x] += self._join(exp(W[i, x, 0, 1]), (string1[i-1], 0), F[i-1, j, x])
                    for y in xrange(1, self.Sigma_size):
                        # substitution
                        if i > 0 and j > 0:
                            F[i, j, y] += self._join(exp(W[i, x, y, 0]), (string1[i-1], y), F[i-1, j-1, x])
                        # insertion
                        if j > 0:
                            F[i, j, y] += self._join(exp(W[i, x, y, 2]), (0, y), F[i, j-1, x])
                
        # check the paths
        all_paths, correct_paths = [], []
        for column in F[-1, :]:
            for entry in column:
                if entry == []:
                    continue
                all_paths += entry

                for pair in entry:
                    upper, lower = [], []
                    score, alignments = pair
                    for x, y in alignments:
                        if x > 0:
                            upper.append(x)
                        if y > 0:
                            lower.append(y)

                    if  lower == list(string2):
                        correct_paths.append(pair)

        # compute partition functions
        Z_expected = 0.0

        for score, path in all_paths:
            #print 'path in all_paths', path, score
            Z_expected += score

        Z_observed = 0.0
        for score, path in correct_paths:
            #print 'path in correct_paths', path, score
            Z_observed += score

        ll = -log(Z_observed) + log(Z_expected)

        try:
            assert np.allclose(ll, self.func(string1, string2, W), atol=ATOL)
        except:
            print 'Z_observed', Z_observed
            print 'Z_expected', Z_expected
            print 'll', ll
            print 'self.func(string1, string2, W)', self.func(string1, string2, W)
            print 'Failed np.allclose(ll, self.func(string1, string2, W), atol=ATOL)'
            exit(1)
        # find best path
        best_score, best_path = -np.inf, None
        for score, path in all_paths:
            if score >= best_score:
                best_score = score
                best_path = path

        best_yield = []
        for x, y in best_path:
            if y != 0:
                best_yield.append(y)

        # crunch
        table = {}
        N = 1000
        best_paths = sorted(all_paths, key=lambda x: -x[0])[:N]
        for score, path in best_paths:
            lst = []
            for x, y in path:
                if y != 0:
                    lst.append(y)
            if tuple(lst) not in table:
                table[tuple(lst)] = 0.0
            table[tuple(lst)] += score
        # convert to log space

        # crunch over paths
        lowest_score, lowest_yield = 0.0, None
        for k, v in table.items():
            table[k] = log(v)
            if table[k] >= lowest_score:
                lowest_score = table[k]
                lowest_yield = k
            #print "crunching", k, table[k]
        lowest_yield = list(lowest_yield)        
        # TODO: enumeration unit test for gradient 
        score, decoded = self.decode(string1, W, V=True)
        score_crunched, decoded_crunched = self.crunch(string1, W, n=N, V=True)
        
        try:
            assert best_yield == decoded
            print 'best_yield', best_yield, log(best_score)
            print 'decoded', decoded, score
        except:
            print 'Failed  best_yield == decoded'
            exit(1)
        try:
            assert lowest_yield == decoded_crunched
            print 'crunched best_yield', lowest_yield, lowest_score
            print 'crunched decoded', decoded_crunched, score_crunched
        except:
            print 'Failed  lowest_yield == decoded_crunched'
            exit(1)
        try:
            assert score_crunched >= score
            assert np.allclose(log(best_score), score, atol=ATOL)
            assert np.allclose(lowest_score, score_crunched, atol=ATOL)
        except:
            print 'best_score', log(best_score)
            print 'score', score
            print 'Failed'
            exit(0)
    #
    # This isn't *that* big of a deal because the combination
    # of an enumeration tes for the likelihood and a finte-difference
    # check basically means the gradient is right. However, it's simple
    # enough to add that we should have it for completeness

    # TODO: unit test for Viterbi
    # Currently, I just test the yield and score.
    # For completeness, I should check the derivation too
    # but I don't return that in my current decode method.
