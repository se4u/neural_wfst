'''
| Filename    : transducer_wrapper.py
| Description : A wrapper for the cython transducer implementation.
| Author      : Pushpendre Rastogi
| Created     : Fri Dec  4 19:18:12 2015 (-0500)
| Last-Updated: Wed Jan 20 23:02:47 2016 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 8
'''
class TransducerWrapper(object):
    def __init__(self, _transducer, sampling_decoding=0, crunching=0):
        ''' Cython object dont support attribute assignments and they
        can't be pickled (at least not that I know of.) Both of these
        together are a problem. So I just create a wrapper of the
        Params
        ------
        _transducer :
        Returns
        -------
        '''
        assert not(sampling_decoding and crunching)
        self._transducer = _transducer
        self.sd = sampling_decoding
        self.crunching = crunching

    def func(self, *args):
        return self._transducer.func(*args)

    def grad(self, *args):
        return self._transducer.grad(*args)

    def decode(self, *args):
        if self.sd > 0:
            params = list(args) + [self.sd]
            return self._transducer.sample(*params)
        elif self.crunching > 0:
            params = list(args) + [self.crunching]
            return self._transducer.crunch(*params)
        else:
            return self._transducer.decode(*args)

    def __getattr__(self, k):
        return self.__dict__[k]

    def __setattr__(self, k, v):
        self.__dict__[k] = v

    def __getstate__(self):
        '''
        From https://docs.python.org/3.1/library/pickle.html#pickle-inst

        Classes can further influence how their instances are pickled;
        if the class defines the method __getstate__(), it is called
        and the returned object is pickled as the contents for the
        instance, instead of the contents of the instance's
        dictionary.

        If __getstate__() returns a false value, the __setstate__()
        method will not be called upon unpickling, otherwise if the
        class defines __setstate__() upon unpickling, it is called with
        the unpickled state.
        '''
        return False

    def __str__(self):
        return 'Class:%s Object. Config=%s'%(
            self.__class__, str(self.__dict__))
