# -*- coding: utf-8 -*-
'''
| Filename    : util_theano.py
| Description : Theano utility functions
| Author      : Pushpendre Rastogi
| Created     : Sat Oct 24 12:56:23 2015 (-0400)
| Last-Updated: Thu Nov 12 00:05:49 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 16
'''
import theano
import theano.tensor
import numpy as np


def th_floatX(data):
    return theano.tensor.cast(data, theano.config.floatX)


def th_reverse(tensor):
    ''' A theano operation to reverse a tensor by treating it as a list of
    matrices along its first dimension.
    Params
    ------
    tensor :
    '''
    rev, _ = theano.scan(lambda itm: itm,
                         sequences=tensor,
                         go_backwards=True,
                         strict=True,
                         name='reverse_rand%d' % np.random.randint(1000))
    return rev

def th_squeeze(x, axis):
    '''
    Params
    ------
    x    : Tensor to squeeze.
    axis : The axis to squeeze along.
    Returns
    -------
    A squeezed tensor.
    '''
    return x.dimshuffle([e for e in xrange(x.ndim) if e != axis])

def th_logsumexp(x, axis=None, keepdims=False):
    module = theano.tensor
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = (xmax if keepdims else th_squeeze(xmax, axis))
    return (xmax_ + module.log(module.exp(x - xmax).sum(axis=axis, keepdims=keepdims)))


def th_access_each_row_of_x_by_index(x, idx_arr):
    return x[theano.tensor.arange(idx_arr.shape[0]), idx_arr]

def th_choose(X, K, reshape=True):
    '''
    Params
    ------
    X : X is a n-dimensional tensor. where n is either 2 or 3.
    K : K is a n-1 dimensional tensor of integers whose
      (i_0, i_1, i_(n-2)) - th element is an index k, denoting the
      (i_0, i_1, i_(n-2), k) -th element of X.
    reshape: Should we reshape the parameter so that the returned value
      resembles the shape of K?
    Returns
    -------
    A tensor of the same size as K that contains entries indexed from X.
    '''
    assert K.ndim == X.ndim - 1
    if X.ndim == 2:
        return X[theano.tensor.arange(K.shape[0]), K]
    elif X.ndim == 3:
        i = K.shape[0]
        j = K.shape[1]
        k = X.shape[2]
        ij = (i * j).astype('int32')
        ret = X.reshape((ij, k))[
            theano.tensor.arange(ij, dtype='int32'), K.reshape((ij,))]
        return (ret.reshape((i, j))
                if reshape
                else ret)
    else:
        raise NotImplementedError

def th_batch_choose(X, K, reshape=True):
    '''
    Params
    ------
    X : X is a 4D tensor.
    K : K[i, j] is the index of the entry that I need to choose
      from the last dimension(axis=3) and I am allowed to choose
      all the values from second last dimension. (axis=2).
    Returns
    -------
    It essentially returns the submatrix resulting from choosing
    X[i, j, :, K[i, j]] for all i, j
    '''
    assert X.ndim == 4
    i = X.shape[0]
    j = X.shape[1]
    k = X.shape[2]
    l = X.shape[3]
    ij = i * j
    ret = X.reshape((ij, k, l))[
        theano.tensor.arange(ij), :, K.reshape((ij,))]
    return (ret.reshape(i, j, k)
            if reshape
            else ret)
