# Filename: test_known_grads.py
# Description:
# Author: Pushpendre Rastogi
# Created: Sun Apr 19 16:57:25 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 110

# Commentary:
#
import theano
import numpy as np
import theano.tensor as T
## On Scalars
x_np = np.float32(2.0)
alpha_np = np.float32(1.0)
np_dy = np.float32(1e-4) # Setting this to 1 seems to work. The thing is that it is multiplying whatever it sees.
ff = lambda alpha, x: alpha * alpha * alpha * x

x = T.fscalar('x')
alpha = theano.shared(alpha_np)
y = ff(alpha, x)
# yy = y # This naive trick doesn't work
dw = T.grad(None,
            wrt=[alpha],
            known_grads = {y: theano.shared(np_dy)})
dw_np = theano.function([x], dw)(x_np)[0]
true_deriv = (dw_np/np_dy)
print "(Derivative * dy): %g, True_Derivative: %g"%(dw_np, true_deriv)
"Now In order to make y change by dy amount. I need to change alpha by dy/(dw_np/dy)"
y_before =  ff(alpha_np, x_np)
delta_alpha = (1/true_deriv) * np_dy
print "delta_alpha: ", delta_alpha
alpha_np += delta_alpha
y_after = ff(alpha_np, x_np)
delta_achieved = (y_after - y_before)
print "delta_achieved: %g, Delta Desired: %g"%(delta_achieved, np_dy)
assert abs(delta_achieved - np_dy) < 1e-8

#############################################
# On vectors when the complete graph is known
x_np = np.array([1.0, 1.0, 1.0])
U_np = np.random.randn(3,3)
V_np = np.random.randn(3,3)
[U, V] = [theano.shared(e) for e in [U_np, V_np]]
U.name = 'U'
V.name = 'V'
x = T.dvector('x')
y = T.dot(U + U, x)
y.name = 'y'
z = T.dot(V, T.dot(V, y))
z.name = 'z'
cost = z.sum()
cost.name = 'cost'
params = [U, V]
true_grads = theano.function([x], T.grad(cost, params))
true_grads = true_grads(x_np)
for var, params in zip([z, y], [(U, V), (U,)]):
    print var, params
    grad = T.grad(cost, var)
    grad_f = theano.function([var], grad)
    print "grad", grad_f(np.array([1, 1, 1]).astype(np.float64))
    full = theano.function([x],
                           T.grad(
                               cost=None,
                               known_grads={var: grad},
                               wrt=params))
    full = full(x_np)
    print len(true_grads), len(full)
    for a, b in zip(true_grads, full):
        if not np.allclose(a, b):
            print var, params
#############################################
# On vectors when the complete graph is not known
x_np = np.array([1.0, 1.0, 1.0])
U_np = np.random.randn(3,3)
V_np = np.random.randn(3,3)
[U, V] = [theano.shared(e) for e in [U_np, V_np]]
U.name = 'U'
V.name = 'V'
x = T.dvector('x')
y = T.dot(U + U, x)
y.name = 'y'
z = T.dot(V, T.dot(V, y))
z.name = 'z'
cost = z.sum()
cost.name = 'cost'
params = [U, V]
true_grads = theano.function([x], T.grad(cost, params))
true_grads = true_grads(x_np)
for var, params in zip([z, y], [(U, V), (U,)]):
    print var, params
    grad_np = np.zeros(1)
    disconnected_grad = theano.shared(grad_np, borrow=True)
    del grad
    full = theano.function([x],
                           T.grad(
                               cost=None,
                               known_grads={var: disconnected_grad},
                               wrt=params))
    grad = T.grad(cost, var)
    grad = theano.function([var], grad)(np.array([1, 1, 1]).astype(np.float64))
    print "grad", grad

    # Update grad_np with the correct value
    grad_np.resize(grad.shape, refcheck=False)
    np.copyto(grad, grad_np)

    full = full(x_np)

    print len(true_grads), len(full)
    for a, b in zip(true_grads, full):
        if not np.allclose(a, b):
            print var, params
