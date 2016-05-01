'''
| Filename    : class_serializable_lambda.py
| Description : Serializable Lambda
| Author      : Pushpendre Rastogi
| Created     : Wed Oct 21 18:44:05 2015 (-0400)
| Last-Updated: Sat Mar 26 16:35:57 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 6
'''
import theano


class SerializableLambda(object):

    def __init__(self, s):
        self.s = s
        self.f = eval(s)
        return

    def __repr__(self):
        return "SerializableLambda('%s')" % self.s

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


RELU_FN = SerializableLambda('lambda x: x + theano.tensor.abs_(x)')
SIGMOID_FN = SerializableLambda('lambda x: theano.tensor.nnet.sigmoid(x)')
TANH_FN = SerializableLambda('lambda x: 2*theano.tensor.nnet.sigmoid(2*x) - 1')
