'''
| Filename    : lstm_seqlabel_validation.py
| Description : Functions that provide validation facility.
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 20:04:44 2015 (-0400)
| Last-Updated: Sun May  1 09:07:58 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 64
'''
import util_lstm_seqlabel
import numpy
import time
import rasengan
import lstm_seqlabel_load_save_model
import functools
import codecs
try:
    from dependency_parser.RectangleDependencyParser import DependencyParser
    dp_viterbi_parse = DependencyParser().viterbi_parse
except:
    rasengan.warn(
        'You dont have the depenedency parser. Dont worry if you just want the transducer')


def get_conlleval_for_task(args):
    if args.task == 'slu':
        from data.atis import conlleval
    elif args.task == 'chunking':
        from data.conll2003_ner import conlleval
    elif args.task == 'ner':
        from data.conll2003_ner import conlleval
    elif args.task == 'postag':
        from data.conll_postag import eval_pwa as conlleval
    else:
        raise NotImplementedError
    return conlleval


def tolerant_y_predictor(args, f_classify, words, expected_len_of_sentence):
    ''' Because of the need to support non-batch and batched inputs and
    some error handling the logic to produce predictions from the inputs
    is a little concoluted and this functions handles that.

    Params
    ------
    args                     :
    f_classify               :
    words                    :
    expected_len_of_sentence :
    Returns
    -------
    '''
    if not args.batch_input:
        y_pred = numpy.array([f_classify(e) for e in words])
    else:
        if expected_len_of_sentence == 1:
            try:
                y_pred = f_classify(words)
            except NotImplementedError as ni_err:
                y_pred = util_lstm_seqlabel.deduplicate_label(
                    f_classify(
                        util_lstm_seqlabel.duplicate_middle_word(words)))
        else:
            y_pred = f_classify(words)
    return y_pred


def wrap_float(f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        results = f(*args, **kwargs)
        if isinstance(results, float) or isinstance(results, int):
            tmp = results
            results = {}
            results['f1'] = tmp
            results['p'] = tmp
            results['r'] = tmp
        return results
    return g


@wrap_float
def validate_predictions_transducer(
        test_lex, idx2label, args, f_classify, groundtruth_valid,
        words_valid, fn='/current.valid.txt', conv_x_to_batch=False):
    ''' Validate the predictions made by the transducer.
    '''
    correct = 0.0
    print 'Writing predictions to ', (args.folder + fn)
    with codecs.open(args.folder + fn, mode="w", encoding="utf8") as f:
        f.write('input prediction goldOutput\n')
        for (x, y) in zip(test_lex, groundtruth_valid):
            y = ''.join(list(y))
            prediction = ''.join([idx2label[e] for e in f_classify(x)])
            assert x.shape[1] % 2 == 1
            mid_column = (x.shape[1] - 1) / 2
            x = ''.join([idx2label[e] for e in x[:, mid_column]])
            f.write('%s %s %s\n' % (x, prediction, y))
            correct += (prediction == y)
    return 100 * float(correct) / len(groundtruth_valid)


@wrap_float
def validate_predictions_seq(
        test_lex, idx2label, args, f_classify, groundtruth_valid,
        words_valid, fn='/current.valid.txt', conv_x_to_batch=True):
    ''' On the validation set predict the labels using f_classify.
    Compare those labels against groundtruth.
    The folder is for storing the evaluation script results so we can stare at them.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    batch_size = 100
    shuf_idx_lst = util_lstm_seqlabel.get_shuffling_index_sorted_by_length(
        test_lex, shuffle=False)
    total_tokens = sum(len(e) for e in test_lex)
    tokens_done = 0
    conlleval = get_conlleval_for_task(args)
    y_pred_list = [None] * len(test_lex)
    tic = time.time()
    for expected_len, idx_list in shuf_idx_lst:
        for i, idx_batch in enumerate(rasengan.batch_list(idx_list, batch_size)):
            x_batch = [test_lex[e] for e in idx_batch]
            if conv_x_to_batch:
                words = numpy.array(
                    [util_lstm_seqlabel.conv_x(sentence, args.win, args.vocsize)
                     for sentence in x_batch])
            else:
                words = numpy.array(x_batch)
            y_pred = tolerant_y_predictor(
                args, f_classify, words, expected_len)
            for i_y_pred, idx in enumerate(idx_batch):
                y_pred_list[idx] = y_pred[i_y_pred]
            if args.verbose == 2 and i % 100 == 0:
                tokens_done += expected_len * len(idx_batch)
                percentage_complete = float(tokens_done) / total_tokens * 100
                util_lstm_seqlabel.print_progress(
                    percentage_complete, tic)
                pass
            pass
        pass
    if args.folder is None:
        assert idx2label is None
        assert words_valid is None
        results_arr = [((p == g).sum(), p.shape[0])
                       for (p, g)
                       in zip(y_pred_list, groundtruth_valid)]
        results = (float(sum(e[0] for e in results_arr))
                   / sum(e[1] for e in results_arr))
    else:
        predictions_valid = util_lstm_seqlabel.convert_id_to_word(
            y_pred_list, idx2label)
        results = conlleval(
            predictions_valid, groundtruth_valid, words_valid,
            args.folder + fn, args.folder)
    return results


def validate_predictions_pdp(valid_lex, _1, args, f_classify, valid_y, _4, _5):
    ''' On the validation set predict the labels using f_classify.
    Compare those labels against groundtruth.
    The folder is for storing the evaluation script results so we can stare at them.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    def evaluate_predicted_parse(viterbi_parse, true_parse):
        return (len([e for e in viterbi_parse if e in true_parse]), len(true_parse))

    results = []
    for sentence, true_parse in zip(valid_lex, valid_y):
        arc_scores = f_classify(
            util_lstm_seqlabel.conv_x(sentence, args.win, args.vocsize))
        if arc_scores.ndim == 2:
            arc_scores = numpy.expand_dims(arc_scores, 2)
        viterbi_score, viterbi_parse = dp_viterbi_parse(arc_scores)
        if __debug__:
            pass
        else:
            util_lstm_seqlabel.print_domination(arc_scores)
            print ' VALIDATION: parents', [e[0] for e in viterbi_parse]
            print ' TRUE: parents', [e[0] for e in true_parse]
        results.append(evaluate_predicted_parse(viterbi_parse, true_parse))
    correct_arcs = sum(e[0] for e in results)
    total_arcs = sum(e[1] for e in results)
    tmp = float(correct_arcs) / total_arcs * 100
    results = {}
    for e in ['f1', 'p', 'r']:
        results[e] = tmp
    return results


def testing(args, data, ttns):
    rasengan.warn('NOTE: this function presupposes that the parameters were '
                  'loaded inside the circuit already')
    test_result = args.validate_predictions_f(
        data.test_lex,
        data.idx2label,
        args,
        ttns.test_f_classify,
        data.test_y,
        data.words_test,
        fn='/current.test.txt')
    print('Test F1', test_result['f1'])
    return test_result['f1']
