import theano
from ..util_theano import th_choose
import numpy
np_x = numpy.random.rand(3,2,5).astype(numpy.float32)
np_y = numpy.array([[0, 2], [1, 3], [4, 1]]).astype(numpy.int32)
x = theano.tensor.tensor3('x')
y = theano.tensor.imatrix('y')
f2 = theano.function([x, y], theano.tensor.inc_subtensor(th_choose(x, y, reshape=False), -1).reshape((3,2,5)))
o2 = f2(np_x, np_y)
nz = numpy.transpose(numpy.nonzero(np_x - o2))
for e in nz:
    (i, j, k) = e
    assert np_y[i, j] == k
