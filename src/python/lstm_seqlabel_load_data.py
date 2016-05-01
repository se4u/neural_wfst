'''
| Filename    : lstm_seqlabel_load_data.py
| Description : The load data function used in lstm_seqlabel.py
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 19:36:47 2015 (-0400)
| Last-Updated: Tue Nov 24 03:57:11 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 25
'''
from collections import defaultdict
import functools
from rasengan import Namespace
import util_lstm_seqlabel


def create_vocabulary(train_lex, valid_lex, test_lex):
    vocab = {}
    for row in (train_lex + valid_lex + test_lex):
        for word in row:
            vocab[word] = None
    return vocab


def get_loaddata_for_task(args):
    if args.task == 'slu':
        from data.atis import loaddata
    elif args.task == 'chunking':
        from data.conll2003_ner import loaddata_chunking
        loaddata = functools.partial(loaddata_chunking,
                                     lc=args.lower_case_input,
                                     oov_thresh=args.chunking_oovthresh,
                                     d2z=args.digit_to_zero,
                                     also_add_test_file=True,
                                     care_about_OOV=False)
    elif args.task == 'ner':
        from data.conll2003_ner import loaddata_ner
        loaddata = functools.partial(loaddata_ner,
                                     lc=args.lower_case_input,
                                     oov_thresh=args.ner_oovthresh,
                                     d2z=args.digit_to_zero,
                                     also_add_test_file=True,
                                     care_about_OOV=False)
    elif args.task == 'postag':
        from data.conll_postag import conll_pos
        loaddata = functools.partial(conll_pos,
                                     lc=args.lower_case_input,
                                     oov=args.pos_oovthresh)
    elif args.task == 'pdp':
        from data.conll_pdp import loaddata as loaddata_pdp
        loaddata = functools.partial(loaddata_pdp,
                                     binary_arclabel=args.binary_arclabel,
                                     size_limit=(None
                                                 if args.limit_corpus == 0
                                                 else args.limit_corpus))
    else:
        raise NotImplementedError
    return loaddata


def load_data(args, summarize=False):
    ''' This function loads data according to `args` and fills the data
    namespace.
    '''
    loaddata = get_loaddata_for_task(args)
    if args.task in ['slu', 'postag']:
        train_set, valid_set, test_set, dic = loaddata()
        word2idx = dic['words2idx']
        label2idx = dic['labels2idx']
        train_lex, train_y = train_set
        valid_lex, valid_y = valid_set
        test_lex, test_y = test_set
    elif args.task in ['chunking', 'ner', 'pdp']:
        train_lex, train_y, valid_lex, valid_y, test_lex, test_y, word2idx, label2idx = loaddata()
    else:
        raise NotImplementedError

    # Reverse dictionaries from indices to words.
    idx2word = dict((k, v) for v, k in word2idx.iteritems())
    idx2label = dict((k, v) for v, k in label2idx.iteritems())

    # Delete slice of data, in case we want to run fast.
    if args.limit_corpus:
        for e in [train_lex, train_y, valid_lex, valid_y, test_lex, test_y]:
            del e[(args.limit_corpus + 1):]

    vocsize = len(word2idx)
    nsentences = len(train_lex)
    nclasses = len(label2idx)

    if args.task != 'pdp':
        valid_y = util_lstm_seqlabel.convert_id_to_word(valid_y, idx2label)
        test_y = util_lstm_seqlabel.convert_id_to_word(test_y, idx2label)

    #--------------------------------------------------------------------------#
    # We mix validation into training in case we want to train on all the data #
    # and skip validation and instead train for a fixed number of epochs       #
    #--------------------------------------------------------------------------#
    if args.mix_validation_into_training:
        print '\n', "We'll mix validation into training data", '\n'
        train_lex = train_lex + valid_lex
        train_y = train_y + valid_y

    #---------------------------------------------------------------------#
    # Some times it is nice to validate on the entire training set, so we #
    # may replace the validation set by the entire set in this portion    #
    #---------------------------------------------------------------------#
    if args.replace_validation_by_training:
        valid_lex = train_lex
        valid_y = train_y
    #----------------------------------------------#
    # Update args and data  from the loaded data. #
    #----------------------------------------------#
    data = Namespace()
    data.nsentences = nsentences
    data.vocsize = vocsize
    data.nclasses = nclasses
    data.train_lex = train_lex
    data.train_y = train_y
    data.valid_lex = valid_lex
    data.valid_y = valid_y
    data.test_lex = test_lex
    data.test_y = test_y
    data.idx2label = idx2label
    data.idx2word = idx2word
    data.words_train = util_lstm_seqlabel.convert_id_to_word(
        train_lex, idx2word)
    data.words_valid = util_lstm_seqlabel.convert_id_to_word(
        valid_lex, idx2word)
    data.words_test = util_lstm_seqlabel.convert_id_to_word(
        test_lex,   idx2word)
    if summarize:
        print 'Data summary', 'vocsize', vocsize, 'nclasses', nclasses, \
            'len(train_lex)', len(train_lex), 'len(valid_lex)', len(valid_lex)
    return data
