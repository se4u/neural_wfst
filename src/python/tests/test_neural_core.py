# USAGE: THEANO_FLAGS=mode=FAST_COMPILE python test_neural_core.py
# This basically checks that the individual pieces of neural_core are correct.
#!/usr/bin/python
# ---------------------------------------
# File Name : lstm_core.py
# Creation Date : 14-02-2015
# Last Modified : Fri Mar 13 15:27:38 2015
# Created By : wdd
# ---------------------------------------
import os, random
import theano
from collections import OrderedDict
import theano.tensor as T
import numpy as np
from itertools import *
from theano import config
from neural_lib_simplification import *

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def rand_weight_uniform(x_dim, y_dim):
    return np.random.uniform(-1.0, 1.0, (x_dim, y_dim)).astype(config.floatX)


def rand_weight(x_dim, y_dim):
    return np.random.rand(x_dim, y_dim).astype(config.floatX)


def randn_weight(x_dim, y_dim=None):
    if y_dim:
        return np.random.randn(x_dim, y_dim).astype(config.floatX)
    else:
        return np.random.randn(x_dim).astype(config.floatX)


def ortho_weight(ndim):
    W = randn_weight(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def p(pp, name):
    return '%s_%s' % (pp, name)


floatX = os.environ.get('TYPE_PYTHON_FLOAT', 'float32')

class TheanoNeuralLayer:

    def __init__(self, name, x_dim, y_dim):
        self._name = name
        self._xdim = x_dim
        self._ydim = y_dim
        self._params = OrderedDict()

    def reg_params(self, tparams):
        for kk, pp in self._params.iteritems():
            tparams[p(self._name, kk)] = self._params[kk]

    def get_layer(self, *x_in):
        pass


class Lstm(TheanoNeuralLayer):

    def __init__(self, name, x_dim, y_dim, go_backwards=False):
        TheanoNeuralLayer.__init__(self, name, x_dim, y_dim)
        self.go_backwards = go_backwards
        self._params['W'] = theano.shared(rand_weight_uniform(self._xdim, 4*self._ydim),
                                          name=p(self._name, 'W'))
        self._params['U'] = theano.shared(np.concatenate([ortho_weight(self._ydim),
                                                          ortho_weight(
                                                              self._ydim),
                                                          ortho_weight(
                                                              self._ydim),
                                                          ortho_weight(self._ydim)],
                                                         axis=1), name=p(self._name, 'U'))
        self._params['b'] = theano.shared(
            np.zeros(4 * self._ydim).astype(config.floatX), p(self._name, 'b'))

    def get_layer(self, x_in):
        assert x_in.ndim == 2
        n_steps = x_in.shape[0]

        def __slice(x_, n, dim):
            return x_[n * dim: (n + 1) * dim]

        def __step(x_, h_, c_):
            preact = T.dot(h_, self._params['U']) + x_ + self._params['b']
            i = T.nnet.sigmoid(__slice(preact, 0, self._ydim))
            f = T.nnet.sigmoid(__slice(preact, 1, self._ydim))
            o = T.nnet.sigmoid(__slice(preact, 2, self._ydim))
            c = T.tanh(__slice(preact, 3, self._ydim))
            c = f * c_ + i * c
            h = o * T.tanh(c)
            return h, c

        x_in = T.dot(x_in, self._params['W']) + self._params['b']
        rval, updates = theano.scan(__step, sequences=x_in, go_backwards=self.go_backwards,
                                    outputs_info=[T.alloc(np_floatX(0.), self._ydim),
                                                  T.alloc(np_floatX(0.), self._ydim)],
                                    name='lstm_layers', n_steps=n_steps)
        return reverse(rval[0]) if self.go_backwards else rval[0]


class Wemb(TheanoNeuralLayer):

    def __init__(self, name, x_dim, y_dim, pre_train={'emb': 'NAN'}):
        TheanoNeuralLayer.__init__(self, name, x_dim, y_dim)
        init_mtx = 0.01 * rand_weight(self._xdim + 2, self._ydim)
        if pre_train and 'emb' in pre_train and os.path.isfile(pre_train['emb']):
            assert 'dict' in pre_train and not (pre_train['dict'] is None)
            assert 'load' in pre_train and not (pre_train['load'] is None)
            print 'loading word embedding! from ' + pre_train['emb']
            pre_train['load'](pre_train['emb'], pre_train['dict'], init_mtx)
        else:
            print 'random initialize word embedding'
        self._params['W'] = theano.shared(init_mtx, name=p(self._name, 'W'))

    def get_emb(self):
        return self._params['W']

    def get_emb_nobe(self):
        return self._params['W'][:-2]

    def get_layer(self, x_in):
        n_timesteps = x_in.shape[0]
        window_size = x_in.shape[1] if x_in.ndim > 1 else 1
        y_out = self._params['W'][x_in.flatten()].reshape(
            [n_timesteps, window_size * self._ydim])
        return y_out


class SimpleLayer(TheanoNeuralLayer):

    def __init__(self, name, x_dim, y_dim):
        TheanoNeuralLayer.__init__(self, name, x_dim, y_dim)
        self._params['U'] = theano.shared(
            0.01 * randn_weight(self._xdim, self._ydim), name=p(self._name, 'U'))
        self._params['b'] = theano.shared(
            np.zeros((1, self._ydim)).astype(config.floatX), name=p(self._name, 'b'))

    def get_layer(self, x_in, op=lambda x: x):
        b_ = T.addbroadcast(self._params['b'], 0)
        ret = op(T.dot(x_in, self._params['U']) + b_)
        return ret


class LocalClassifier():
    # here y_dim is the hidden-layer dimension for parameter 'U'

    def __init__(self, name, x_dim1, x_dim2, y_dim):
        self._name = name
        self._xdim1 = x_dim1
        self._xdim2 = x_dim2
        self._ydim = y_dim
        self._params = OrderedDict()
        self._params['U1'] = theano.shared(
            0.01 * rand_weight(self._xdim1, self._ydim), name=p(self._name, 'U1'))
        self._params['U2'] = theano.shared(
            0.01 * rand_weight(self._xdim2, self._ydim), name=p(self._name, 'U2'))
        self._params['b'] = theano.shared(
            rand_weight(1, self._ydim), name=p(self._name, 'b'))
        self._params['v0'] = theano.shared(
            np.random.rand(self._ydim).astype(config.floatX), name=p(self._name, 'v0'))

    def reg_params(self, tparams):
        for kk, pp in self._params.iteritems():
            tparams[p(self._name, kk)] = self._params[kk]

    def get_layer(self, x_in, C_in, ty_i):  # op,

        n_steps = C_in.shape[0]

        def __logsumexp(x, axis=None):
            xmax = x.max(axis=axis, keepdims=True)
            xmax_ = x.max(axis=axis)
            return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

        def __step(_C, _x):
            #scores = T.dot( T.dot(_x, self._params['U']) + self._params['b'], self._params['v0'])
            scores = T.dot(T.nnet.sigmoid(T.dot(_x, self._params[
                           'U1']) + T.dot(_C, self._params['U2']) + self._params['b']), self._params['v0'])
            return scores.flatten()

        y_out, _ = theano.scan(
            __step, sequences=C_in, non_sequences=x_in, name='classification_layer', n_steps=n_steps)
        norm_y = y_out.flatten() - __logsumexp(y_out)
        f_lc_debug = theano.function(
            [x_in, C_in, ty_i], [y_out, norm_y, norm_y[ty_i]])
        return norm_y[ty_i], T.argmax(norm_y), f_lc_debug


class Crf(TheanoNeuralLayer):

    def __init__(self, name, x_dim, viterbi=False):
        TheanoNeuralLayer.__init__(self, name, x_dim, 1)
        self._viterbi = viterbi

    def score(self, x_in, y):

        def __step(o, y, p_, y_):
            p = p_ + o[y_, y]
            return p, y

        assert x_in[0].ndim == 2
        p0 = x_in[0, -2, y[0]]
        y0 = y[0]

        [rval, _], _ = theano.scan(__step, outputs_info=[p0, y0],
                                   sequences=[x_in[1:, :-2], y[1:]])
        return rval[-1]

    def get_layer(self, x_in):

        def __logsumexp(x, axis=None):
            xmax = x.max(axis=axis, keepdims=True)
            xmax_ = x.max(axis=axis)
            return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

        def __step(o_, p_):
            p_ = p_.dimshuffle(0, 'x')
            f_ = p_ + o_
            p = f_.max(axis=0) if self._viterbi else __logsumexp(
                f_, axis=0)
            y = f_.argmax(axis=0)
            return p, y

        assert x_in[0].ndim == 2
        p0 = x_in[0, -2]
        [rval, bp], _ = theano.scan(
            __step, outputs_info=[p0, None], sequences=x_in[1:, :-2])
        yn = rval[-1].argmax(axis=0)
        path_tmp, _ = theano.scan(
            lambda p, y: p[y], outputs_info=yn, sequences=bp, go_backwards=True)
        path = T.concatenate([reverse(path_tmp), yn.dimshuffle('x')])
        if self._viterbi:
            return rval[-1].max(axis=0), path
        else:
            return __logsumexp(rval[-1], axis=0), path

def reset():
    random.seed(0)
    np.random.seed(0)
    return
def allclose(testname, *args):
    try:
        for (a,b) in zip(args[::2], args[1::2]):
            a = a.get_value()
            b = b.get_value()
            assert np.allclose(a, b, atol=1e-6), '%s and %s mismatch'%(a, b)
        print testname, 'passed'
    except Exception as e:
        print testname, 'failed'
        print e
        exit(1)

if __name__ == '__main__':
    import pdb
    getsimplobj = lambda : StackConfig(dict(simpl_obj_out_dim=100))
    # Test Wemb
    reset()
    core_emb = Wemb('neural_core', 5, 100).get_emb()
    reset()
    dummy = theano.tensor.ivector('dummy')
    namespace = getsimplobj()
    Embedding('simpl_obj', namespace).prepend(Start(5+2, dummy))
    simpl_T = namespace['tparam_simpl_obj_T_1']
    allclose('Test Wemb', core_emb, simpl_T)

    # Test Lstm
    reset()
    core_lstm = Lstm('lstm_core', 5, 100, go_backwards=False)._params
    core_W, core_U, core_b = [core_lstm[e] for e in ['W', 'U', 'b']]
    reset()
    dummy = theano.tensor.fvector('dummy')
    namespace = getsimplobj()
    namespace['simpl_obj_go_backwards']=False
    LSTM('simpl_obj', namespace).prepend(Start(5, dummy))
    simpl_W = namespace['tparam_simpl_obj_W_3']
    simpl_U = namespace['tparam_simpl_obj_U_4']
    simpl_b = namespace['tparam_simpl_obj_b_5']
    allclose('Test Lstm', core_W, simpl_W, core_U, simpl_U, core_b, simpl_b)
    # Test SimpleLayer / Biased Linear
    reset()
    core_simplelayer = SimpleLayer('core_simplelayer', 5, 100)._params
    core_U, core_b = [core_simplelayer[e] for e in ['U', 'b']]
    reset()
    namespace = getsimplobj()
    BiasedLinear('simpl_obj', namespace).prepend(Start(5, dummy))
    simpl_N = namespace['tparam_simpl_obj_N_8']
    simpl_b = namespace['tparam_simpl_obj_b_10']
    allclose('Test SimpleLayer', core_U, simpl_N, core_b, simpl_b)
    # Test CRF
    reset()
    core_crf = Crf('core_crf', 10, viterbi=False)
    dummy = np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12]],
                         [[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12]]]).astype('float32').transpose(1, 2, 0)
    dummy_tv = theano.tensor.tensor3('dummy')
    core_crf_partition_tv, core_crf_path_tv = core_crf.get_layer(
        dummy_tv)
    tmp_f = theano.function([dummy_tv],
                            [core_crf_partition_tv,
                             core_crf_path_tv])
    [core_crf_partition, core_crf_path] = tmp_f(dummy)
    namespace = getsimplobj()
    _tmp = OrderOnePathMax('simpl_obj', namespace).prepend(Start(0, dummy_tv))
    simpl_crf_partition_tv = _tmp._partition_val
    simpl_crf_path_tv = _tmp.output_tv
    [simpl_crf_partition, simpl_crf_path] = theano.function(
        [dummy_tv], [simpl_crf_partition_tv, simpl_crf_path_tv])(dummy)
    assert np.allclose(core_crf_partition, simpl_crf_partition)
    assert np.allclose(core_crf_path, simpl_crf_path)
    print 'Test CRF passed'
    reset()
    print 'all test pass'
