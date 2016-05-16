'''
| Filename    : lstm_seqlabel_training.py
| Description : Functions that allows us to train circuits.
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 20:06:03 2015 (-0400)
| Last-Updated: Wed Dec 23 17:52:06 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 213
'''
import util_lstm_seqlabel
import time
import sys
import numpy
import contextlib
import lstm_seqlabel_callbacks
import zlib
try:
    from dependency_parser.RectangleDependencyParser_cython \
        import dp_insideOutside, dp_find_LogZ
except ImportError:
    print 'Could not import dependency parser. Dont try to use it.'
import re
import rasengan
import contextlib
try:
    from scipy.optimize import fmin_l_bfgs_b as lbfgs
except ImportError:
    print 'Couldnt import scipy.optimize.fmin_l_bfgs_b'
import scipy.optimize

def train_transducer_lbfgs(
        train_lex, train_y, args, ttns, training_stats, batch_size=None):
    ''' This function completes a training epoch by doing one run of LBFGS.
    `ts` abbreviates `train_stack` in entire function

    Params
    ------
    train_lex      : A list of input_strings (the strings are represented as np arrays)
    train_y        : A list of output strings
    batch_size     : UNUSED : (default None)
    '''
    assert args.clipping_value < 0
    assert args.projection_threshold < 0

    ts_param_name = [
        str(e) for e in ttns.train_stack_config.updatable_parameters()]
    print 'The following params will be trained by lbfgs', ts_param_name
    ts_param_shape_list = [ttns.train_stack_config[name].get_value().shape
                           for name in ts_param_name]
    ts_param_shape_map = dict(zip(ts_param_name, ts_param_shape_list))

    total_param = sum(numpy.prod(shape)
                      for shape
                      in ts_param_shape_map.values())

    def set_entries_in_ttns(param_vec):
        ''' Set entries in ttns.train_stack_config
        with corresponding values in param_vec.
        '''
        param_vec = param_vec.astype('float32')
        offset = 0
        for name in ts_param_name:
            shape = ts_param_shape_map[name]
            numel = numpy.prod(shape)
            ttns.train_stack_config[name].set_value(
                param_vec[offset:offset + numel].reshape(shape))
            offset += numel
            pass
        return

    def vectorize(param_list, dtype='float32'):
        param_vec = numpy.zeros((total_param,), dtype=dtype)
        offset = 0
        for idx, param in enumerate(param_list):
            shape = param.shape
            assert shape == ts_param_shape_list[idx]
            numel = numpy.prod(shape)
            param_vec[offset:offset + numel] = param.reshape((numel,)).astype(dtype)
            offset += numel
            pass
        return param_vec

    def get_entries_in_ttns():
        ''' Set entries in ttns.train_stack_config
        with corresponding values in param_vec.
        '''
        return vectorize(
            [ttns.train_stack_config[name].get_value()
             for name
             in ts_param_name])

    def loss_over_corpus(param_vec):
        ''' Compute the loss value over the entire corpus.
        '''
        set_entries_in_ttns(param_vec)
        corpus_cost = 0
        for idx in range(len(train_lex)):
            input_string = train_lex[idx]
            output_string = train_y[idx]
            corpus_cost += ttns.train_f_cost(input_string, output_string)
        return corpus_cost / len(train_lex)

    def gradient_over_corpus(param_vec):
        set_entries_in_ttns(param_vec)
        corpus_grad = numpy.zeros((total_param,), dtype='float64')
        for idx in range(len(train_lex)):
            input_string = train_lex[idx]
            output_string = train_y[idx]
            tmp_grad = ttns.train_f_grad(input_string, output_string)
            corpus_grad += vectorize(tmp_grad, 'float64')
        return corpus_grad / len(train_lex)

    with rasengan.tictoc("Training %d epoch"%training_stats['epoch_id']):
        init_param = get_entries_in_ttns()
        rasengan.warn('Skipped FD Check')
        # print 'Check grad output: Error=', scipy.optimize.check_grad(func=loss_over_corpus, grad=gradient_over_corpus, x0=init_param)
        opt_param = scipy.optimize.fmin_l_bfgs_b(
            loss_over_corpus, init_param,
            fprime=gradient_over_corpus, disp=2, maxiter=1000)[0]
        set_entries_in_ttns(opt_param)
    return

def train_transducer(train_lex, train_y, args, ttns, training_stats, batch_size=None):
    ''' This function is completes an epoch of training for the transducer.
    The primary function of this method is to facilitate as the numpy.array
    interface between the actual data and the thenao code.
    '''
    epoch_id = training_stats['epoch_id']
    shuf_idx_lst = util_lstm_seqlabel.get_shuffling_index_sorted_by_length(
        train_lex, shuffle=False)
    total_tokens = sum(e * len(v) for (e, v) in shuf_idx_lst)
    tic = time.time()
    tokens_done = 0
    epoch_cost = 0

    for expected_len, idx_list in shuf_idx_lst:
        if expected_len <= 1:
            continue
        for i, idx in enumerate(idx_list):
            input_string = train_lex[idx]
            output_string = train_y[idx]
            epoch_cost += ttns.train_f_cost(input_string, output_string)
            ttns.train_f_update(
                training_stats['clr'], input_string, output_string)
            tokens_done += expected_len
            percentage_complete = float(tokens_done) / total_tokens * 100
            if args.verbose >= 3 and i % 10 == 0:
                util_lstm_seqlabel.print_progress(
                    percentage_complete, tic, epoch_id=epoch_id,
                    carriage_return=False)
                pass
            pass
        pass
    training_stats['epoch_cost'].append(epoch_cost)
    print
    print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic)
    print '>> Epoch Cost, ', epoch_cost, '<<'
    return

def train_seq(train_lex, train_y, args, ttns, training_stats):
    ''' This function is called from the main method. and it is primarily
    responsible for updating the parameters. Because of the way that
    create_circuit works that creates f_cost, f_update etc. this function
    needs to be flexible and can't be put in a lib.
    Look at lstm_dependency_parsing_simplification.py for more pointers.
    '''
    batch_size = args.batch_size
    epoch_id = training_stats['epoch_id']
    shuf_idx_lst = util_lstm_seqlabel.get_shuffling_index_sorted_by_length(
        train_lex)
    total_tokens = sum(e * len(v) for (e, v) in shuf_idx_lst)
    tokens_done = 0
    tic = time.time()
    epoch_cost = 0
    for expected_len, idx_list in shuf_idx_lst:
        if expected_len == 0:
            continue
        for i, idx_batch in enumerate(rasengan.batch_list(idx_list, batch_size)):
            x_batch = [train_lex[e] for e in idx_batch]
            y_batch = [train_y[e] for e in idx_batch]
            words = numpy.array(
                [util_lstm_seqlabel.conv_x(sentence, args.win, args.vocsize)
                 for sentence in x_batch],
                dtype=numpy.int32)
            labels = numpy.array(
                [util_lstm_seqlabel.conv_y(sentence)
                 for sentence in y_batch],
                dtype=numpy.int32)
            # Words should be a bunch of 3D tensors of shape
            # n_sentences, n_tokens, window
            if not args.batch_input:
                for (ww, ll) in zip(words, labels):
                    epoch_cost += ttns.train_f_update(
                        training_stats['clr'], ww, ll)
            else:
                try:
                    epoch_cost += ttns.train_f_update(
                        training_stats['clr'] * len(idx_batch), words, labels)
                    for p in ttns.train_stack_config.differentiable_parameters():
                        rasengan.validate_np_array(
                            p.get_value(), name=p.name,
                            describe=args.describe_training)
                        pass
                except (NotImplementedError, ValueError):
                    if expected_len == 1:
                        # Desperate hack to overcome theano frustrations.
                        # Theano does not make it easy to handle
                        # corner cases. One such corner case is when
                        # the scan code receives zero length sequences
                        # which happens when the sentence has length 1
                        # because then the score of an order 1 model
                        # does not require a recursion. In any case to
                        # actually provide a prediction in this edge
                        # case we can just duplicate the sentence once in
                        # the right dimension and go on.
                        ttns.train_f_update(
                            training_stats['clr'],
                            util_lstm_seqlabel.duplicate_middle_word(words),
                            util_lstm_seqlabel.duplicate_label(labels))
                    else:
                        sys_exc_info = sys.exc_info()
                        raise sys_exc_info[0], sys_exc_info[1], sys_exc_info[2]

            tokens_done += expected_len * len(idx_batch)
            percentage_complete = float(tokens_done) / total_tokens * 100
            if args.verbose >= 3 and i % 10 == 0:
                util_lstm_seqlabel.print_progress(
                    percentage_complete, tic, epoch_id=epoch_id)
                pass
            pass
        pass
    training_stats['epoch_cost'].append(epoch_cost)
    print
    print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic)
    print '>> Epoch Cost, ', epoch_cost, '<<'
    return


def train_pdp(train_lex, train_y, args, f_cost, f_update, f_classify, epoch_id,
              learning_rate):
    '''
    # Inputs #
    train_wsp : It is a list of ([word], [arc]) tuples.
                Every arc is a tuple (parent_idx, child_idx, label)
    args : Contains training parameters, like window size and vocabulary size
           and verbosity
    f_cost : It is empty. Just used for side effects. Its input is the
             absolute_input
    f_update : It is also empty. Its input is the learning rate.
    f_y_pred : It takes absolute input and produces y_pred.
    epoch_id : Its an int
    learning_rate : Its a theano variable.

    # What Happens #
    Now, I will reify a function to create y_pred.
    I will feed those y_pred (which are actually arc scores) to the
    dependency parser's neg-ll-grad-func and get resulting gradient *G*

    I will copy *G* to gradarr numpy array. (resize, copyto)
    I will call optimization functions and they would simply do their job
    and update the class_chips parameters.
    '''
    util_lstm_seqlabel.shuffle([train_lex, train_y])

    def get_arcscore_tensor_grad_from_dp(gold_parse, arc_scores, n, num_labels):
        efficient_arc_count = dp_insideOutside(arc_scores.astype('float64'))
        for (parent, child, label) in gold_parse:
            efficient_arc_count[parent, child, label] -= 1
        return efficient_arc_count

    tic = time.time()
    time_arc_scores = 0.0
    time_nll_grad = 0.0
    time_updates = 0.0
    number_of_sentences_skipped = 0
    for i, (sentence, arcs) in enumerate(zip(train_lex, train_y)):
        sentence_len = len(sentence)
        if sentence_len < 2 or sentence_len > args.sentence_maxlength:
            continue
        # Convert words to a window of words.
        words = util_lstm_seqlabel._conv_x(sentence, args.win, args.vocsize)
        tic_tmp = time.time()
        arc_scores = f_classify(words)
        time_arc_scores += (time.time() - tic_tmp)
        arc_scores_orig_dim = arc_scores.ndim
        if arc_scores_orig_dim == 2:
            arc_scores = numpy.expand_dims(arc_scores, 2)

        find_lp = lambda _scores, _arcs: \
            sum([_scores[e] for e in _arcs]) - dp_find_LogZ(_scores)
        if __debug__:
            pass
        else:
            util_lstm_seqlabel.print_domination(arc_scores)
            LogProbofGoldParseBefore = find_lp(arc_scores, arcs)
        tic_tmp = time.time()
        nll_grad_wrt_arc_score = get_arcscore_tensor_grad_from_dp(
            arcs, arc_scores, sentence_len, 1).astype('float32')
        time_nll_grad += (time.time() - tic_tmp)
        if util_lstm_seqlabel.is_invalid(nll_grad_wrt_arc_score):
            number_of_sentences_skipped += 1
            print "Skipped idx: ", i, "Consecutively Skipped: ", number_of_sentences_skipped
            if number_of_sentences_skipped >= 100:
                print "Skipped 100 sentences, no point training any further"
                break
            continue
        else:
            number_of_sentences_skipped = 0
        tic_tmp = time.time()
        f_cost(words,
               (nll_grad_wrt_arc_score
                if arc_scores_orig_dim == 3
                else nll_grad_wrt_arc_score.squeeze()))
        f_update(learning_rate)
        time_updates += (time.time() - tic_tmp)
        if __debug__:
            pass
        else:
            arc_scores_after_update = numpy.expand_dims(f_classify(words), 2)
            print 'Delta Log Prob of Gold Parse (Should be +ve) : ', find_lp(arc_scores_after_update, arcs) - LogProbofGoldParseBefore, 'sentence_len: ', sentence_len
            if sentence_len == 6:
                # , arc_scores_after_update.squeeze()
                print arc_scores.squeeze(), nll_grad_wrt_arc_score.squeeze()
        if args.verbose >= 3:
            print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / args.nsentences),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    total_time = time.time() - tic
    print '\n', 'Time training 1 epoch with %d sentence' % len(train_lex), total_time, [e / total_time for e in [time_arc_scores, time_nll_grad, time_updates]], '\n'
    return


@contextlib.contextmanager
def setup_training_environment(args, ttns, training_stats):
    train_stack_param_names = [
        str(e) for e in ttns.train_stack_config.updatable_parameters()]

    test_stack_param_names = [
        str(e) for e in ttns.test_stack_config.updatable_parameters()]

    # Assert that train_stack and test_stack are perfectly aligned so
    # that zipping them would do the right thing.
    for trv, tev in map(lambda x, y: (x, y), train_stack_param_names, test_stack_param_names):
        assert tev.startswith(re.match(r'^(.*_)\d+$', trv).group(1))

    # Populate two maps.
    train_test_map = dict(zip(train_stack_param_names, test_stack_param_names))
    train_hash_map = dict((e, zlib.adler32(ttns.train_stack_config[e].get_value()))
                          for e in train_stack_param_names)
    print 'Epoch:', training_stats['epoch_id'], \
        'Learning Rate', training_stats['clr']
    #---#
    yield
    #---#
    if args.verbose >= 2:
        param_copy_str = ', '.join('%s -> %s'%(str(trv), str(tev))
                                   for trv, tev
                                   in train_test_map.items())
        print 'Copying parameters', param_copy_str
    # Based on the trained parameters. Update the parameters of the
    # testing model.
    for trv, tev in train_test_map.items():
        # Assert that trained parameters actually changed.
        trv_new_hash = zlib.adler32(ttns.train_stack_config[trv].get_value())
        assert trv_new_hash != train_hash_map[trv], (
            'This assertion checks that during each training epoch, the neural '
            'network parameters actually change. If the neural net parameters '
            'do not change in one epoch, then we raise as assertion error '
            'to stop the training loop.')
        
        if hasattr(ttns.train_stack_config[trv], 'dropout_retention_freq'):
            retention_freq = ttns.train_stack_config[trv].dropout_retention_freq
            ttns.test_stack_config[tev].set_value(
                ttns.train_stack_config[trv].get_value()
                * retention_freq)
        else:
            ttns.test_stack_config[tev].set_value(
                ttns.train_stack_config[trv].get_value())
            assert (zlib.adler32(ttns.test_stack_config[tev].get_value())
                    == trv_new_hash)
    pass

def training(args, data, ttns):
    '''
    Params
    ------
    args    : The namespace that contains training - related parameters.
    data    : The data namespace used to train the neural network. It contains
              train_lex, train_y, valid_lex, valid_y, words_valid
    ttns    : The train_test namespace which contains compiled functions and
              associated stack configuration namespaces.
    '''
    # Training is through SGD by passing through the data for certrain epochs.
    # `training_stats` is  a namespace of globals that I will maintain
    # during the training run.
    training_stats = dict(
        clr=args.lr,         # Current Learning Rate.
        best_epoch_id=0,     # Id of best epoch.
        best_f1=-numpy.inf,  # F1 of best epoch.
        epoch_id=-1,         # Current Epoch Id.
        epoch_cost=[])       # Holder of all epoch costs.

    while training_stats['epoch_id'] + 1 < args.nepochs:
        training_stats['epoch_id'] += 1

        #--------------------#
        # Train the circuit. #
        #--------------------#
        if not args.jump_to_validation:
            # setup_training_environment copies parameters from the
            # training circuit to the testing circuit.
            with setup_training_environment(args, ttns, training_stats):
                args.train_f(
                    data.train_lex,
                    data.train_y,
                    args,
                    ttns,
                    training_stats)

        #-------------------------------------------------------#
        # Perform validation if conditions to skip are not met. #
        #-------------------------------------------------------#
        skip_validation = (args.skip_validation != 0
                           and training_stats['epoch_id'] % args.skip_validation != 0)
        if skip_validation:
            validation_result = dict(f1=-1)
        else:
            with rasengan.announce_ctm("Calculating Training Performance"):
                training_result = args.validate_predictions_f(
                    rasengan.sample_from_list(data.train_lex, 1000),
                    data.idx2label,
                    args,
                    ttns.test_f_classify,
                    util_lstm_seqlabel.convert_id_to_word(
                        rasengan.sample_from_list(data.train_y, 1000),
                        data.idx2label),
                    rasengan.sample_from_list(data.words_train, 1000),
                    fn='/current.train.txt')

            with rasengan.announce_ctm("Calculating Validation Performance"):
                validation_result = args.validate_predictions_f(
                    data.valid_lex,
                    data.idx2label,
                    args,
                    ttns.test_f_classify,
                    data.valid_y,
                    data.words_valid,
                    fn='/current.valid.txt')

            print ('epoch_id:', training_stats['epoch_id'],
                   'Training F1:', training_result['f1'],
                   'Validation F1:', validation_result['f1'])

        #---------------------------------------------------------------------#
        # Callbacks to update the training statistics, saved model pickle etc #
        #---------------------------------------------------------------------#
        lstm_seqlabel_callbacks.validate_validation_result(
            validation_result)
        lstm_seqlabel_callbacks.update_training_stats(
            validation_result, training_result, training_stats)
        lstm_seqlabel_callbacks.update_saved_parameters(
            training_stats, ttns.test_stack_config, args)
        lstm_seqlabel_callbacks.update_saved_predictions(
            training_stats, args)
        lstm_seqlabel_callbacks.update_learning_rate(
            training_stats, args)
        if training_stats['clr'] < args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break
        pass
    best_epoch_id = training_stats['best_epoch_id']
    print 'Best Epoch', best_epoch_id,\
        'Validation F1', training_stats['validation_result'][best_epoch_id]['f1']
    return training_stats
