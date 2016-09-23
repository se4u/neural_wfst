'''
| Filename    : lstm_seqlabel_optimizer.py
| Description : Library of train time parameter update algorithms.
| Author      : Pushpendre Rastogi
| Created     : Sun Apr 19 09:13:08 2015 (-0400)
| Last-Updated: Fri Sep 23 16:05:28 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 231

'''
import theano
import numpy
from theano import tensor
from util_lstm_seqlabel import np_floatX
from rasengan import flatten, Namespace

def get_cautious_update_f(updates, lr, x, y, cost):
    ''' Sometimes the theano optimizer may layout the updates improperly.
    so that some parameters get updated in place before their effects on
    other gradients and updates have been computed. This can happen during
    nested scans for example. One simple strategy to overcome this problem
    is to stage the `troublemaking` variables into their own staging area.
    Update the staging area and then to copy the updates over. This way updates
    won't suffer from a race condition.

    Params
    ------
    updates : The update expression list of tuples, First element of tuple
      is the paraemter to be updated and then second one is the update
      expression.
    lr      : The learning rate.
    x       : The input sentence tv.
    y       : The gold output tv.
    cost    : The cost tv.
    Returns
    -------
    A function for updating the variables.
    '''
    print 'Using Cautious Updates'
    params = [e[0] for e in updates]
    updates = [e[1] for e in updates]
    staging_area = [theano.shared(e.get_value()) for e in params]
    update_stage_1 = theano.function(
        flatten([lr, x, y]),
        cost,
        updates=zip(staging_area, updates),
        name='f_update_stage_1')
    update_stage_2 = theano.function(
        [], [], updates=zip(params, staging_area),
        name='f_update_stage_2')

    # Instead of using the lambda notation one can define a sequential function.
    def f_update(p1, p2, p3):
        update_stage_1(p1, p2, p3)
        return update_stage_2()

    # f_update = (lambda p1, p2, p3:
    #             (lambda _: update_stage_2())(
    #                 update_stage_1(p1, p2, p3)))
    f_update.name = 'f_update'
    return f_update

def compile_update_fn(x, y, lr, cost, updates,
                      stack_config, grads):
    ''' Compile the inputs and outputs to produce an update functions.
    Params
    ------
    x       :
    y       :
    lr      :
    cost    :
    updates :
    '''
    f_to_print = None
    if y is not None:
        f_cost = theano.function(flatten([x, y]), cost, name='f_cost')
        f_grad = theano.function(flatten([x, y]), grads, name='f_grad')
    else:
        v2p = stack_config.stack_ns.debug_tv_list[4]
        print "THE VARIABLE TO PRINT IS", v2p
        f_to_print = theano.function(inputs=flatten([x]), outputs=v2p, name='f_v2p')
        f_intermediate = theano.function(
            inputs=flatten([x]),
            outputs=stack_config.stack_ns.pred_y,
            name='f_intermediate')

        def f_cost(str1, str2):
            '''
            The stack_config['endpoint'] contains a func method that has three
            inputs. it takes in a
            1. stringA
            2. stringB
            3. tensor.
            Params
            ------
            *args : The args right now are designed to be just a tuple
                    of two integer sequences. args[0] is the left integer
                   sequence. representing a string.
            Returns
            -------
            '''
            assert str1.ndim == 2
            cols = (str1.shape[1]-1)
            assert cols % 2 == 0
            assert str2.ndim == 1
            print f_to_print(str1)
            intermediate_tensor = f_intermediate(str1).astype('float64')
            return stack_config['endpoint'].func(
                str1[1:, cols/2], str2, intermediate_tensor)

        f_grad_intermediate = theano.function(
            inputs=flatten([x, cost]),
            outputs=grads,
            name='f_grad_intermediate')

        def f_grad(str1, str2):
            assert str1.ndim == 2
            cols = (str1.shape[1]-1)
            assert cols % 2 == 0
            assert str2.ndim == 1
            print f_to_print(str1)
            intermediate_tensor = f_intermediate(str1).astype('float64')
            intermediate_grad = numpy.array(
                stack_config['endpoint'].grad(
                    str1[1:, cols/2], str2, intermediate_tensor),
                dtype='float32')
            return f_grad_intermediate(str1, intermediate_grad)
        pass

    if stack_config['cautious_update']:
        f_update = get_cautious_update_f(updates, lr, x, y, cost)
    else:
        on_unused_input=('ignore' if updates == [] else 'raise')
        if y is not None:
            f_update = theano.function(
                # Input is the learning rate, and supervised example.
                flatten([lr, x, y]),
                # Output is the cost/loss that has to be minimized.
                cost,
                updates=updates,
                name='f_update',
                on_unused_input=on_unused_input)
        else:
            f_update_intermediate = theano.function(
                flatten([lr, x, cost]),
                [],
                updates=updates,
                name='f_update_intermediate',
                on_unused_input=on_unused_input)

            def f_update(lr, str1, str2):
                ''' f_update in this case receives a tuple of
                Params
                ------
                *args :
                Returns
                -------
                '''
                assert str1.ndim == 2
                cols = (str1.shape[1]-1)
                assert cols % 2 == 0
                assert str2.ndim == 1
                print f_to_print(str1)
                intermediate_tensor = f_intermediate(str1).astype(
                    'float64')
                intermediate_grad = numpy.array(
                    stack_config['endpoint'].grad(
                        str1[1:, cols/2], str2, intermediate_tensor),
                    dtype='float32')
                f_update_intermediate(lr, str1, intermediate_grad)
                return (intermediate_tensor, intermediate_grad)

    if y is not None:
        f_classify = theano.function(
            inputs=flatten([stack_config.stack_ns.absolute_input_tv]),
            outputs=stack_config.stack_ns.pred_y,
            name='f_classify')
        f_update.individual_updates = {}
        for (_, u) in updates:
            f_update.individual_updates[u.wrt_name] = theano.function(
                flatten([lr, x, y]), u, name='f_update')
    else:
        def f_classify(in_str):
            assert in_str.ndim == 2
            cols = (in_str.shape[1]-1)
            assert cols % 2 == 0
            # ONLY KEEP THIS IF YOU JUST WANT to print the vectors
            # during decoding (which I call classification)
            print f_to_print(str1)
            return stack_config['endpoint'].decode(
                in_str[1:, cols/2], f_intermediate(in_str).astype('float64'))
    nsf = Namespace()
    nsf.f_cost = f_cost
    nsf.f_update = f_update
    nsf.f_classify = f_classify
    nsf.f_grad = f_grad
    return nsf

###################################################################
# Large decaying learning rates"
#  Momentum needs to be high.
# High Momentum"
###################################################################
def clip_gradients(stack_config, grad_param):
    ''' TODO  Gradients need to be clipped while updating.
    Params
    ------
    stack_config :
    grads        :
    params       :
    '''
    threshold = stack_config['clipping_value']
    print 'clip_gradients threshold', threshold
    if threshold > 0:
        gradients_to_clip = []
        gradients_not_to_clip = []
        for (g, p) in grad_param:
            if (hasattr(p, 'clip_gradient') and p.clip_gradient):
                gradients_to_clip.append((g, p))
                print p.name, 'gradient is being clipped in optimizer.clip_gradients'
            else:
                gradients_not_to_clip.append((g, p))

        if len(gradients_to_clip) == 0:
            return grad_param

        total_grad_norm = tensor.sqrt(tensor.sum(
            [tensor.sum(g * g) for (g, _) in gradients_to_clip]))
        grad_norm_gt_threshold = tensor.ge(total_grad_norm, threshold)
        grad_thresholder = lambda _g: (tensor.switch(
            grad_norm_gt_threshold,
            _g * (threshold / total_grad_norm),
            _g))
        clipped_grad_param = []
        for (g, p) in gradients_to_clip:
            cg = grad_thresholder(g)
            cg.wrt_name = g.wrt_name
            clipped_grad_param.append((cg, p))
        clipped_grad_param += gradients_not_to_clip
        return clipped_grad_param
    else:
        return grad_param

def project_parameters_l2(stack_config, unprojected_param_update):
    '''
    # The dropout paper talks about the importance of
    # "constraining the norm of the incoming weight vector at each
    # hidden unit to be upper bounded by a fixed constant c. ...
    # The neural network was optimized under the constraint ||w|| <= c"
    # The updates need to be projected onto a norm ball.
    Params
    ------
    stack_config :
    unprojected_param_update :
    '''
    threshold = stack_config['projection_threshold']
    print 'parameter projection threshold', threshold
    if threshold > 0:
        projected_param_update = []
        for p, u in unprojected_param_update:
            if hasattr(p, 'l2_project') and p.l2_project:
                # Either the norm of a 'row' is less than threshold,
                # which means that the weight_vector_norm is going to
                # be higher than 1.
                # OR, it is higher than threshold, in which case the
                # norm of a 'row' is less than 1.
                weight_vector_norms = (
                    threshold / u.norm(2, axis=p.l2_projection_axis))
                ones_per_weight_vector = theano.tensor.ones_like(
                    weight_vector_norms)
                # We want to rescale the update `u` so that if a `row`
                # of `u` has higher norm than the threshold, then that
                # `row` gets scaled down. Therefore we choose the
                # minimum between 1 and the weight vector norm. When
                # weight vector norm is higher than threshold, we
                # would end up choosing the multiplier that would push
                # the norm of the row to equal to the threshold.
                multipliers = theano.tensor.minimum(
                    weight_vector_norms, ones_per_weight_vector)
                tmp = range(u.ndim - 1)
                tmp.insert(p.l2_projection_axis, 'x')
                u_ = multipliers.dimshuffle(tmp) * u
                u_.wrt_name = u.wrt_name
                projected_param_update.append((p, u_))
            else:
                projected_param_update.append((p, u))
        return projected_param_update
    else:
        return unprojected_param_update

def assert_alignment_of_grad_param(grad_param, reverse=False):
    for (a, b) in grad_param:
        if reverse:
            assert a.name == b.wrt_name
        else:
            assert a.wrt_name == b.name
    return

def sgd(stack_config):
    ''' The venerable `Stochastic Gradient Descent` procedure enhanced with
    projection and gradient clipping.

    The dropout paper "Dropout: A Simple Way to Prevent Neural Networks from
    Overfitting" talks about additional heuristics beyond dropout that are
    important for training neural networks. such as constraining the
    norm of the weight vectors for each node and using large momentum.

    The dropout-retention frequency should be high in the start of the network and
    decay towards the output.
    Specifically they say that
    " Using dropout along with:
    A) max-norm regularization : Typical values of the bound range from 3 to 4.
    B) large decaying learning rates :
    C) high momentum : While momentum values of 0.9 are common for standard nets,
       with dropout we found that values around 0.95 to 0.99 work quite a lot better.
    provides a significant boost over just using dropout.
    Params
    ------
    stack_config :
    stack_ns     :
    '''
    stack_ns = stack_config.stack_ns
    x = stack_ns.absolute_input_tv
    y = stack_ns.gold_y
    lr = theano.tensor.scalar('lr')
    grad_param = [(g, p)
                  for (g, p)
                  in zip(stack_ns.grads, stack_config.differentiable_parameters())
                  if not p.block_update]
    assert_alignment_of_grad_param(grad_param)
    clipped_grad_param = clip_gradients(stack_config, grad_param)
    assert_alignment_of_grad_param(clipped_grad_param)
    unprojected_param_update = []
    for (g, p) in clipped_grad_param:
        u = p - lr * g
        u.wrt_name = g.wrt_name
        unprojected_param_update.append((p, u))
    projected_param_update = project_parameters_l2(
        stack_config, unprojected_param_update)
    assert_alignment_of_grad_param(projected_param_update, reverse=True)
    return compile_update_fn(
        x, y, lr, stack_ns.cost_or_known_grads_tv, projected_param_update,
        stack_config, [e[0] for e in clipped_grad_param])


def sgd_momentum(stack_config, stack_ns):
    '''
    Phase 1: Initialize the averaged gradients as zeros.
    Phase 2: create unprojected updates for the params.
        Note that we use the averaged gradients for this purpose.
        And then we project the updates to the L2 ball.
    Phase 3: create update expressions for the avg_grads.
        First we clip the original gradients.
        and then we do an online averaging to update the momentum terms.
    Phase 4: total update expressions are created by concatenating
        the `projected_param_updates` and `avg_grad_updates`.

    Params
    ------
    stack_config :
    stack_ns     :
    build_cost_f : (default True)
    '''
    grads = stack_ns.grads
    x = stack_ns.absolute_input_tv
    y = stack_ns.gold_y
    lr = theano.tensor.scalar('lr')
    params = stack_config.differentiable_parameters()
    # Phase 1
    avg_grads = [theano.shared(p.get_value() * 0., name=str(p) + '_avg_grads')
                for p in params]
    # Phase 2
    unprojected_updates = [(p - lr * ag)
                           for (p, ag)
                           in zip(params, avg_grads)]
    projected_updates = project_parameters_l2(
        stack_config, params, unprojected_updates)
    # Phase 3
    grad_clip = clip_gradients(stack_config, grads, params)
    avg_grad_updates = [(ag * stack_config.momentum_coeff
                         + gc * (1 - stack_config.momentum_coeff))
                        for (ag, gc)
                        in zip(avg_grads, grad_clip)]
    # Phase 4
    updates = projected_updates + avg_grad_updates
    return compile_update_fn(x, y, lr, stack_ns.cost_or_known_grads_tv, updates)


def adadelta(stack_config, stack_ns):
    '''

    Params
    ------
    stack_config :
    stack_ns     :
    build_cost_f : (default True)
    '''
    raise NotImplementedError
    l2_regularizer_coeff=1
    lr = theano.tensor.scalar('lr')
    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.), name='%s_grad' % k)
                    for k, p in stack_config]
    running_up2 = [theano.shared(p.get_value() * np_floatX(0.), name='%s_rup2' % k)
                   for k, p in stack_config]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.), name='%s_rgrad2' % k)
                      for k, p in stack_config]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    updir = [-theano.tensor.sqrt(ru2 + 1e-6) / theano.tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p * l2_regularizer_coeff + ud)
                for p, ud in zip([p for (k, p) in stack_config], updir)]
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')
    (f_cost_input, f_cost_output) = (([x, y], cost)
                                     if y is not None
                                     else ([x, cost], []))
    f_cost = theano.function(f_cost_input,
                             f_cost_output,
                             updates=zgup + rg2up,
                             name='adadelta_f_cost')
    return f_cost, f_update


def adaM(stack_config, stack_ns):
    pass

def adaGrad(stack_config, stack_ns):
    pass

def nesterov(stack_config, stack_ns):
    pass
