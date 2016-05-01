"""
Tests for the Theano CRF classifier implementation.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import numpy as np
import theano
from lib_crf_theano import theano_logsumexp, forward_theano, score_labels, viterbi_theano, loglinear_cost
from lib_crf_numpy import forward_vectorized, score_x_y, viterbi, viterbi_vectorized
from theano import tensor


def test_theano_logsumexp():
    x_ = theano.tensor.vector()
    f = theano.function([x_], theano_logsumexp(x_))
    x = -50000. - np.arange(1, 4) / 10.
    np.testing.assert_allclose(f(x), -49999.098057151772)
    y = 50000. + np.arange(1, 4) / 10.
    np.testing.assert_allclose(f(y), 50001.301942848229)
    z = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    y_ = theano.tensor.matrix()
    g = theano.function([y_], theano_logsumexp(y_, axis=0))
    np.testing.assert_allclose(g(z), [-49999.098057151772,
                                      50001.301942848229])
    g = theano.function([y_], theano_logsumexp(y_, axis=1))
    np.testing.assert_allclose(g(z.T), [-49999.098057151772,
                                        50001.301942848229])
    # Negative indices make Theano barf at the moment.
    # (Theano ticket gh-1430)
    # g = theano.function([y_], theano_logsumexp(y_, axis=-2))
    # np.testing.assert_allclose(g(z), [-49999.098057151772,
    #                                   50001.301942848229])
    # np.testing.assert_allclose(g(z.T), [-49999.098057151772,
    #                                     50001.301942848229])


def test_forward_theano():
    rng = np.random.RandomState([2013, 6, 1])
    o = theano.tensor.matrix()
    c = theano.tensor.matrix()
    y = theano.tensor.ivector()
    f = theano.function([o, c], forward_theano(o, c))
    #g = theano.function([o, c], forward_theano(o, c, viterbi=True))
    v = theano.function([o, c], viterbi_theano(o, c))
    s = theano.function([o, y, c], score_labels(o, y, c))
    c = theano.function([o, y, c], loglinear_cost(o, y, c))

    for i in range(20):
        num_labels = rng.random_integers(2, 10)
        num_timesteps = rng.random_integers(2, 10)
        obs = rng.uniform(size=(num_timesteps, num_labels))
        chain = rng.uniform(size=(num_labels, num_labels))
        labels = rng.randint(0, num_labels, num_timesteps).astype('int32')
        print labels
        #print obs
        #print chain
        #print score_x_y(obs, chain, labels) 
        #print s(obs, labels, chain)
        #np.testing.assert_allclose(score_x_y(obs, chain, labels), s(obs, labels, chain))
        #print g(obs, chain)
        #print forward_vectorized(obs, chain, viterbi=True)
        #print viterbi(obs, chain) 
        #print viterbi_vectorized(obs, chain) 
        #print v(obs, chain)
        score_y_x = s(obs, labels, chain)
        print score_y_x
        print f(obs, chain)
        cost = c(obs, labels, chain)
        print cost
        #np.testing.assert_allclose(f(obs, chain),
        #                           forward_vectorized(obs, chain))
        #np.testing.assert_allclose(g(obs, chain),
        #                           forward_vectorized(obs, chain,
        #                                              viterbi=True))
test_forward_theano()
