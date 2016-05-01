#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
| Filename    : transducer_score.py
| Description : A script to produce scores for the transducer.
| Author      : Pushpendre Rastogi
| Created     : Thu Nov 12 00:48:02 2015 (-0500)
| Last-Updated: Fri Apr  8 11:50:48 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 533
USAGE: ./transducer_score.py

OR

python -c 'import transducer_score; transducer_score.main(pretrained_param_pklfile=<filename>)'
'''
import lstm_seqlabel_circuit
import lstm_seqlabel_circuit_compilation
import lstm_seqlabel_optimizer
import lstm_seqlabel_training
import lstm_seqlabel_validation
import rasengan
import transducer_circuit
import transducer_wrapper
from transducer.src.transducer import Transducer
import transducer_data
import util_lstm_seqlabel

LSTM_PROPERTIES = ['out_dim', 'do_dropout', 'dropout_retention_freq',
                       'clip_gradient', 'l2_project', 'add_bias']
FORWARD_LSTM_PROPERTIES = ['forward_'+e for e in LSTM_PROPERTIES]
BACKWARD_LSTM_PROPERTIES = ['backward_'+e for e in LSTM_PROPERTIES]
ARABIC_TO_ROMAN_MAP = {
    0:'0', 1:'I', 2:'II', 3:'III', 4:'IV', 5:'V', 6:'VI', 7:'VII', 8:'VIII'}
def util_add_bilstm_prop(args, bilstm_name):
    for invariant in FORWARD_LSTM_PROPERTIES:
        args.copy_invariant_is_suffix(invariant, 'bilstm', bilstm_name)
    for prop_src, prop_dest in zip(
            FORWARD_LSTM_PROPERTIES, BACKWARD_LSTM_PROPERTIES):
        # Copy Backward LSTM property from forward part
        args.copy_invariant_is_prefix(bilstm_name, prop_src, prop_dest)
    return args

def args_creation_part1(args):
    ''' Add data/task related parameters to args.
    '''
    args.task = 'transduction'
    args.limit_corpus = 0
    args.mix_validation_into_training = 0
    args.replace_validation_by_training = 0
    args.jump_to_validation = 0
    # Full window size. Used in the numerize function to window the sequence
    args.win = 3
    #----------------#
    # Save Locations #
    #----------------#
    args.folder = r'../../results/transducer'
    args.pkl_name = r'transducer.pkl'
    # A flag to partition the dev data into dev/test. (default:0)
    # In non zero then determines the samples kept in dev. (ex:500)
    args.partition_dev_into_test = 0
    # A flag to partition the dev data into train/dev. (default:0)
    # If non zero then determines the samples kept in dev. (ex:500)
    args.partition_dev_into_train = 0
    args.use_0bl=0
    args.use_1bl=0
    args.use_8bl=0
    args.use_6bl=0
    args.use_1l=0
    args.use_4l=0
    args.sampling_decoding=0
    args.crunching=0
    args.bilstm_stagger_schedule = 'extended'
    args.bilstm_externalandcopyatmax = 0
    args.bilstm_runbilstmseparately = 0
    args.wemb1_out_dim = 10
    args.bilstm_forward_out_dim = 15
    args.penalty_do_dropout = 0
    args.penalty_dropout_retention_freq = -1
    args.penalty_tensor_decomp_ta_h_prod = 0
    args.penalty_tensor_decomp_ta_h_prodrelu = 0
    args.penalty_tensor_decomp_t_a_h_prod = 0
    args.penalty_simple_decomp_jason = 0
    args.penalty_full_decomp_jason = 0
    args.penalty_my_decomp = 0
    args.penalty_my_decomp_h_dim=-1
    return args

def args_creation_part2(args, data):
    if args.penalty_full_decomp_jason:
        assert args.use_1bl
        assert args.bilstm_stagger_schedule == 'external'

    if (args.partition_dev_into_test
        and args.partition_dev_into_train):
        rasengan.warn('NOTE: You are pilfering from dev into both train and test')
    #------------------------#
    # Add Topology Arguments #
    #------------------------#
    args.in_dim = (data.vocsize + 2)
    args.wemb1_win_size = args.win
    args.penalty_vocsize = data.vocsize
    args.penalty_mid_col = (args.wemb1_win_size - 1)/2
    if args.use_0bl:
        bilstm_stack = []
    elif args.use_1bl:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm')]
    elif args.use_1l:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm'),]
    elif args.use_6bl:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIV'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmV'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmVI')]
    elif args.use_8bl:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIV'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmV'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmVI'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmVII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmVIII')]
    elif args.use_4l:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIV')]
    else:
        bilstm_stack = [
            (lstm_seqlabel_circuit.BiLSTM, 'bilstm'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIII'),
            (lstm_seqlabel_circuit.BiLSTM, 'bilstmIV')]
    args.chips = (
        [(lstm_seqlabel_circuit.Embedding, 'wemb1')]
        + bilstm_stack
        +[(transducer_circuit.Penalty, 'penalty')])
    #---------------------------------------------#
    # Learning Rates, Optimizers, Epoch, EndPoint #
    #---------------------------------------------#
    args.optimizer = lstm_seqlabel_optimizer.sgd
    args.perform_training = 1
    args.perform_testing = (0 or args.partition_dev_into_test)
    args.lr = 0.4
    args.lr_drop = 0.9
    args.nepochs = 1000
    args.train_f = lstm_seqlabel_training.train_transducer
    args.validate_predictions_f = (
        lstm_seqlabel_validation.validate_predictions_transducer)
    args.verbose = 2
    args.skip_validation = 0
    INSERTION_LIMIT = 3
    args.endpoint = transducer_wrapper.TransducerWrapper(
        Transducer(data.vocsize, INSERTION_LIMIT),
        sampling_decoding=args.sampling_decoding,
        crunching=args.crunching)
    args.endpoint.dont_pickle = 1
    print args.endpoint
    #-------------------------------------------------------#
    # Dropout, Gradient Clipping, L2 Projection for *Wemb1* #
    #-------------------------------------------------------#
    args.wemb1_do_dropout = 1
    args.wemb1_dropout_retention_freq = .8
    args.wemb1_clip_gradient = 1
    args.wemb1_l2_project = 1
    #---------------------------------------------------------#
    # Dropout, Gradient Clipping, L2 Projection for *Penalty* #
    #---------------------------------------------------------#
    rasengan.warn('We DONT DO DROPOUT ON PENALTY !!')
    args.penalty_clip_gradient = 1
    args.penalty_l2_project = 1
    args.penalty_tie_copy_param = 1
    args.penalty_vocsize = data.vocsize
    #-----------------#
    # LSTM parameters #
    #-----------------#
    # Set the forward LSTM of the first LSTM by hand.
    # Forward LSTM
    if args.bilstm_externalandcopyatmax:
        args.bilstm_stagger_schedule = 'external'
        pass

    if args.bilstm_runbilstmseparately:
        args.bilstm_stagger_schedule = 'external'
        pass

    args.bilstm_do_backward_pass = not (args.use_1l or args.use_4l)
    args.bilstm_forward_do_dropout = 1
    args.bilstm_forward_dropout_retention_freq = 0.8
    args.bilstm_forward_clip_gradient = 1
    args.bilstm_forward_l2_project = 1
    args.bilstm_forward_add_bias = 1
    for prop_src, prop_dest in zip(
            FORWARD_LSTM_PROPERTIES, BACKWARD_LSTM_PROPERTIES):
        # Copy Backward LSTM property from the forward part
        args.copy_invariant_is_prefix('bilstm', prop_src, prop_dest)
    #------------------------------------------------------------#
    # Settings for later BiLSTMs : bilstmII, bilstmIII, bilstmIV #
    # These settings are simply copied over.                     #
    # There is no need to remove properties, since properties    #
    # that are not needed would simply not be compiled.          #
    #------------------------------------------------------------#
    for bilstm_height in range(2, len(bilstm_stack)+1):
        at_top = (bilstm_height == len(bilstm_stack))
        bilstm_height = ARABIC_TO_ROMAN_MAP[bilstm_height]
        if args.bilstm_externalandcopyatmax or args.bilstm_runbilstmseparately:
            if at_top:
                bl_name = ('bilstm%s_forcefully_copy_embedding'
                           '_to_output'%bilstm_height)
                setattr(args, bl_name, 1)
            pass
        if args.bilstm_runbilstmseparately:
            setattr(args, 'bilstm%s_segregate_bilstm_inputs'%bilstm_height,
                    args.bilstm_runbilstmseparately)

        setattr(args, 'bilstm%s_stagger_schedule'%bilstm_height,
                args.bilstm_stagger_schedule)
        setattr(args, 'bilstm%s_do_backward_pass'%bilstm_height,
                args.bilstm_do_backward_pass)
        args = util_add_bilstm_prop(args, 'bilstm%s'%bilstm_height)
    #----------------------------------------------#
    # The clipping Value and Projection Threshold. #
    #----------------------------------------------#
    args.clipping_value = 10
    args.projection_threshold = 7
    #------------------------------------------#
    # Settings for blocking updates to layers. #
    #------------------------------------------#
    args.wemb1_block_update = 0
    args.bilstm_forward_block_update = 0
    args.bilstm_backward_block_update = 0
    args.bilstmII_forward_block_update = 0
    args.bilstmII_backward_block_update = 0
    args.penalty_block_update = 0
    #----------------------------#
    # Learning Rate Controllers. #
    #----------------------------#
    args.decay = 0
    args.decay_epochs = 0
    args.minimum_lr = 1e-5
    # The learning rate decay exponent.
    args.lr_decay_exponent = 0
    #-------------------------#
    # Loading Pretrained PKL. #
    #-------------------------#
    rasengan.warn('NOTE: I have set pretrained_param_pklfile to None')
    args.pretrained_param_pklfile = None
    return args

def update_args(args, kwargs):
    for k, v in kwargs.items():
        print ("Setting args.%s ="%k), v
        setattr(args, k, v)
    return args

def main(*_fold_info, **kwargs):
    args = rasengan.Namespace()
    args = args_creation_part1(args)
    args = update_args(args, kwargs)
    data = transducer_data.main(args)
    args = args_creation_part2(args, data)
    args = update_args(args, kwargs)
    args.diff_kwargs = kwargs


    # Start each run in a fresh directory to avoid interference
    # with other running processes and avoiding overwriting results of
    # previous processes.
    import os
    idx = 0
    while os.path.exists(args.folder + '_' + str(idx)):
        idx += 1
    args.folder = args.folder + '_' + str(idx)
    print 'Set args.folder to', args.folder
    if __name__ != '__main__':
        # We are probably running from the hpolibrary.
        # In non-interactive batch runs, it is also important to
        # disable debug_support so that we dont call post-mortem in
        # case of an exception.
        rasengan.disable_debug_support()
    else:
        #rasengan.warn('NOTE: I am using pretrained pkl')
        # args.pretrained_param_pklfile = (args.folder + r'/' + args.pkl_name)
        pass
    with lstm_seqlabel_circuit_compilation.make(args, force=True):
        error = lstm_seqlabel_circuit_compilation.perform_training_and_testing(
            "", args, data)
    return error

if __name__ == '__main__':
    main()
