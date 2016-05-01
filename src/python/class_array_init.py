'''
| Filename    : class_array_init.py
| Description : Class for initializing arrays in a theano model.
| Author      : Pushpendre Rastogi
| Created     : Wed Oct 21 18:40:04 2015 (-0400)
| Last-Updated: Thu Nov 19 04:19:19 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 6
'''
import gzip
import os
from theano import config
import numpy as np


def read_matrix_from_file(fn, dic, desired_dimension):
    ''' Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    open_f = gzip.open if fn.endswith('.gz') else open
    _data = open_f(fn).read().strip().split('\n')
    _data = [e.strip().split() for e in _data]
    data = {}
    # NOTE: The norm of onesided_uniform rv is sqrt(n)/sqrt(3)
    # Since the expected value of X^2 = 1/3 where X ~ U[0, 1]
    # => sum(X_i^2) = dim/3
    # => norm       = sqrt(dim/3)
    # => norm/dim   = sqrt(1/3dim)
    # multiplier = np.sqrt(1.0/(3*dim))
    for e in _data:
        data[e[0]] = np.array([float(_e) for _e in e[1:]])
    M = ArrayInit(ArrayInit.onesided_uniform,
                  multiplier=(1.0 / desired_dimension)).initialize(len(dic) + 2,
                                                                   desired_dimension)
    dim_r = min(len(e) - 1, desired_dimension)
    for word, idx in dic.iteritems():
        if word in data:
            M[idx, :dim_r] = data[word][:dim_r]
    print 'loaded embeddings from %s' % fn
    return M


class ArrayInit(object):
    normal = 'normal'
    onesided_uniform = 'onesided_uniform'
    twosided_uniform = 'twosided_uniform'
    ortho = 'ortho'
    zero = 'zero'
    unit = 'unit'
    ones = 'ones'
    fromfile = 'fromfile'
    penalty = 'penalty'
    identity = 'identity'
    def __init__(self, option,
                 multiplier=0.01,
                 matrix=None,
                 word2idx=None,
                 desired_dimension=None):
        self.option = option
        self.matrix_filename = None
        m = self._matrix_reader(matrix, word2idx, desired_dimension)
        self.matrix = (m[:, :desired_dimension]
                       if desired_dimension is not None
                       else m)
        self.multiplier = (1
                           if self.matrix is not None
                           else multiplier)
        return

    def _matrix_reader(self, matrix, word2idx, desired_dimension):
        if type(matrix) is str:
            self.matrix_filename = matrix
            assert os.path.exists(matrix), "File %s not found" % matrix
            # DONT PICKLE BECAUSE AS WE SHUFFLE THEN THE ORDER OF WORDS IN VOCAB ALSO CHANGES.
            # pickle_filename = matrix+".pickle"
            # if (os.path.exists(pickle_filename) and newer(pickle_filename, matrix)):
            #     return pickle.load(pickle_filename)
            # else:
            matrix = read_matrix_from_file(matrix, word2idx, desired_dimension)
            #    pickle.dump(matrix, pickle_filename, protocol=-1)
            return matrix
        else:
            return None

    def initialize(self, *xy, **kwargs):
        if self.option == ArrayInit.normal:
            M = np.random.randn(*xy)
        elif self.option == ArrayInit.onesided_uniform:
            M = np.random.rand(*xy)
        elif self.option == ArrayInit.twosided_uniform:
            M = np.random.uniform(-1.0, 1.0, xy)
        elif self.option == ArrayInit.ortho:
            f = lambda dim: np.linalg.svd(np.random.randn(dim, dim))[0]
            if int(xy[1] / xy[0]) < 1 and xy[1] % xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(tuple(f(xy[0]) for _ in range(int(xy[1] / xy[0]))),
                               axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix
        elif self.option == ArrayInit.penalty:
            M = np.zeros(xy)
            for i in range(xy[0]):
                for j in range(xy[1]):
                    M[i, j] = -float(abs(i - j))
        elif self.option == ArrayInit.identity:
            M = np.eye(*xy)
        else:
            raise NotImplementedError
        return (M * self.multiplier).astype(config.floatX)

    def __repr__(self):
        mults = ', multiplier=%s' % ((('%.3f' % self.multiplier)
                                      if type(self.multiplier) is float
                                      else str(self.multiplier)))
        mats = ((', matrix="%s"' % self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)" % (self.option, mults, mats)
