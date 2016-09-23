'''
| Filename    : lstm_seqlabel_circuit_compilation.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Sun Nov 15 03:29:29 2015 (-0500)
| Last-Updated: Fri Sep 23 15:43:37 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 85
'''
import rasengan
import util_lstm_seqlabel
import lstm_seqlabel_training
import lstm_seqlabel_validation
import numpy
import re
import pickle
import class_stack_config
import lstm_seqlabel_circuit
import theano
import contextlib
import os


def compile_args(args):
    stack_config = class_stack_config.StackConfig(args)
    stack_ns = lstm_seqlabel_circuit.metaStackMaker(
        args.chips,
        stack_config)
    stack_config.stack_ns = stack_ns
    try:
        print 'THE THEANO VARIABLE TO PRINT', (stack_config).stack_ns.debug_tv_list[4]
    except:
        pass
    model = args.optimizer(stack_config)
    model.stack_config = stack_config
    return model

def get_train_test_namespace(args):
    if args.perform_training:
        print 'Compiling train_model'
        train_model = compile_args(args)

    set_dropout_to_zero(args)
    print 'Compiling test_model'
    test_model = compile_args(args)
    # Prepare the `ttns` namespace by adding train and test prefixes.
    ttns = rasengan.Namespace('ttns')
    if args.perform_training:
        ttns = ttns.update_and_append_prefix(
            train_model, 'train_')
    ttns = ttns.update_and_append_prefix(
        test_model, 'test_')
    return ttns

def print_pklfn_performance(args, add_newline=True):
    if ((not hasattr(args, 'pretrained_param_pklfile'))
        or (args.pretrained_param_pklfile is None)):
        return
    pkl_fn = args.pretrained_param_pklfile
    if  not os.path.exists(pkl_fn):
        print 'The pretrained param file', pkl_fn, 'does not exist.'
        return
    pkl = pickle.load(open(pkl_fn))
    pa_best_epoch = pkl['best_epoch_id']
    pa_training_f1 = pkl['training_result'][pa_best_epoch]['f1']
    pa_validation_f1 = pkl['validation_result'][pa_best_epoch]['f1']
    arr = ['pklfn', pkl_fn, 'Training F1', pa_training_f1,
           'Validation F1', pa_validation_f1, 'Best Epoch', pa_best_epoch,]
    for e in arr:
        print e,
    if add_newline:
        print
    return pkl, arr

def load_params_from_pklfile_to_stack_config(pkl_fn, stack_config):
    '''
    Params
    ------
    pkl_fn       : The path of the pickle file that contains the parameters.
    stack_config : The stack_config object that contains handles to theano
      shared variabes to be updated.
    Returns
    -------
    '''
    # Check which parameters of `ttns` need to be loaded
    # from pklfile.
    parameters_required = [str(k) for k in
                           stack_config.differentiable_parameters()]
    # Read required parameters from pklfile.
    # Upto the last id, the parameter names would be similar.
    parameters_available_pkl = pickle.load(open(pkl_fn))
    parameters_available = [k for k in parameters_available_pkl
                            if k.startswith('tparam_')]
    print 'Loading pretrained parameters from pklfn:', pkl_fn, ' to model'
    print 'parameters_required', parameters_required
    print 'parameters_available', parameters_available

    req_to_ava_map = {}
    for p in parameters_required:
        lemma = re.match(r'(.*)_\d+', p).group(1)
        l = [e for e in parameters_available
             if e.startswith(lemma)]
        if len(l) == 1:
            l = l[0]
        elif len(l) == 0:
            l = None
        else:
            raise Exception(str(parameters_available) + p)
        req_to_ava_map[p] = l
    # Update `ttns`
    for p in parameters_required:
        pretrained_param = req_to_ava_map[p]
        if pretrained_param is not None:
            required_shape = stack_config[p].shape.eval()
            available_shape = parameters_available_pkl[pretrained_param].shape
            pp_val = parameters_available_pkl[pretrained_param]
            shape_check = lambda s1, s2 : (
                len(s1) == 2 and len(s2) == 2 and s1[1] == s2[1])
            if shape_check(required_shape, available_shape):
                row_req = required_shape[0]
                row_ava = available_shape[0]
                pad_width = (row_req - row_ava)/2
                pp_val = numpy.pad(
                    pp_val, ((pad_width, pad_width), (0, 0)), mode='constant')
                print 'Padded', pretrained_param, 'with', pad_width, 'zero rows'
            if (all(a == b for (a,b) in zip(required_shape, available_shape))
                or shape_check(required_shape, available_shape)):
                stack_config[p].set_value(pp_val)
                print 'Setting value of', p, 'with', pretrained_param, \
                    'Shape:', pp_val.shape
            else:
                print 'Error:', p, 'requires shape ', required_shape, \
                    'but', pretrained_param, 'has shape', available_shape
                raise NotImplementedError
                pass
            pass
        pass
    return

def load_params_from_pklfile(ttns, args):
    if ((not hasattr(args, 'pretrained_param_pklfile'))
        or (args.pretrained_param_pklfile is None)):
        return
    pkl_fn = args.pretrained_param_pklfile
    if  not os.path.exists(pkl_fn):
        print 'The pretrained param file', pkl_fn, 'does not exist.'
        return
    if args.perform_training:
        print 'Loading params from pklfile to train stack config'
        load_params_from_pklfile_to_stack_config(
            pkl_fn, ttns.train_stack_config)
    print 'Loading params from pklfile to test stack config'
    load_params_from_pklfile_to_stack_config(pkl_fn, ttns.test_stack_config)
    return

def perform_training_and_testing(training_stage, args, data):
    '''
    Returns
    -------
    The validation error. A quantity that we want to minimize.
    '''
    stats = None
    with rasengan.tictoc(training_stage):
        with rasengan.debug_support():
            if args.perform_training or args.perform_testing:
                with rasengan.tictoc("Circuit Compilation"):
                    ttns = get_train_test_namespace(args)
                with rasengan.tictoc("Loading Parameters"):
                    load_params_from_pklfile(ttns, args)
                pass
            rasengan.decrease_print_indent()
            print_pklfn_performance(args)
            rasengan.increase_print_indent()
            # Train
            if args.perform_training:
                with rasengan.tictoc("Training"):
                    stats = lstm_seqlabel_training.training(args, data, ttns)
            # Test (IF asked)
            if args.perform_testing:
                with rasengan.tictoc("Testing"):
                    stats = lstm_seqlabel_validation.testing(args, data, ttns)
                    return (100 - stats)
    if stats is None:
        return 100
    else:
        best_epoch_id = stats['best_epoch_id']
        return (100 - stats['validation_result'][best_epoch_id]['f1'])


def set_dropout_to_zero(args):
    ''' Set all the parameters that have specify dropout
    in a layer to zero.
    Params
    ------
    args : A namespace object.
    Returns
    -------
    A handle to the inplace modified namespace object.
    '''
    for attr in args.__dict__:
        if attr.endswith('do_dropout'):
            args.__dict__[attr] = 0
    return args

@contextlib.contextmanager
def make(args, force=False, pipeline=False):
    ''' In each training run we have to check whether a trained pickle
    file `saveto` for that stage already exists. If it exists then we
    skip the training. Otherwise we load the trained parameters and
    when we leave then we set the pretrained_param_pklfile as the
    pkl file that we just saved parameters to (or that already exists).

    Also we restore the state of perform_training and saveto to defaults.
    Params
    ------
    args  :
    saveto :
    force  : (default False)
    Returns
    -------
    '''
    saveto = os.path.join(args.folder, args.pkl_name)
    with rasengan.tictoc('Making ' + saveto):
        rasengan.ensure_dir(args.folder, verbose=1, treat_as_dir=1)
        if hasattr(args, 'saveto'):
            assert args.saveto == saveto, str((args.saveto, saveto))
        else:
            args.saveto = saveto
            print 'Set args.saveto=', args.saveto
        # Check whether we need to do any training unless forced
        # explicitly.
        pt = args.perform_training
        if not force and os.path.exists(args.saveto):
            args.perform_training = 0
        rasengan.increase_print_indent()
        #----#
        yield
        #----#
        rasengan.decrease_print_indent()
        args.perform_training = pt
        if pipeline:
            # Set the pretrained_param_pklfile field to a value after saving
            # parameters to that location.
            args.pretrained_param_pklfile = args.saveto
            # Reset args.saveto to null
            args.saveto = None
