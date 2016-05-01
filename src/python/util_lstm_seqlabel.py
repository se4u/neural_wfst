'''
| Filename    : util_lstm_seqlabel.py
| Description : Utility functions for the lstm_seqlabel.py file.
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 20:01:22 2015 (-0400)
| Last-Updated: Wed Dec 16 03:49:16 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 44
'''
import collections
import contextlib
import numpy
import random
import rasengan
import re
import sys
import theano
import time

def set_seed(seed):
    ''' Set the seed in both numpy and random module
    '''
    numpy.random.seed(seed)
    random.seed(seed)


def is_invalid(arr):
    return any([f(arr).any() for f in [numpy.isinf, numpy.isnan, numpy.isneginf]])


def is_there_a_dominating_row(mat):
    di = None
    for i in range(mat.shape[0]):
        if all(all(mat[i] > mat[j])
               for j in range(mat.shape[0])
               if i != j):
            di = i
    return di


def print_domination(arc_scores):
    print 'Dominating Row: ', is_there_a_dominating_row(arc_scores.squeeze())
    return


def convert_id_to_word(corpus, idx2label):
    return [[idx2label[word] for word in sentence]
            for sentence
            in corpus]


def conv_x(x, window_size, vocsize):
    x = list(x)
    x = [vocsize] + x + [vocsize + 1]
    cwords = contextwin(x, window_size)
    words = numpy.ndarray((len(x), window_size)).astype('int32')
    for i, win in enumerate(cwords):
        words[i] = win
    return words[1:-1]


def conv_y(y):
    return y


def pprint_per_line(d, l):
    ''' Pretty print the entries in a dictionary/list based on the
    indices / keys contained in the list.
    Params
    ------
    d : A dict or a list.
    l : A list of keys or indexes
    '''
    for k in l:
        print (k, d[k])
    return


def shuffle(lol):
    '''
    shuffle inplace each list in the same order by ensuring that we
    use the same state for every run of shuffle.

    lol :: list of list as input
    '''
    state = random.getstate()
    for l in lol:
        random.setstate(state)
        random.shuffle(l)


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def np_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)



def get_shuffling_index_sorted_by_length(X, seed=None, shuffle=True):
    '''
    together. Shuffle them by default but allow for not shuffling as well.
    Params
    ------
    X    : X is a list of sequences.
    seed : (default 10)
    Returns
    -------
    Return a list of tuples where each tuple contains (length:l, list of indices:lst)
    such that if the sequence X was sorted corresponding to lst then all the length l
    elements from original X would come together.
    '''
    dd = collections.defaultdict(list)
    for i, x in enumerate(X):
        dd[len(x)].append(i)
    if shuffle:
        for k in dd:
            with rasengan.reseed_ctm(seed):
                random.shuffle(dd[k])
    shuffled_idx = [(k, dd[k]) for k in sorted(dd.keys())]
    return shuffled_idx


def print_progress(percentage_complete, tic, epoch_id=None, carriage_return=True):
    '''
    Params
    ------
    epoch_id            : The current Epoch
    percentage_complete :
    tic                 :
    Returns
    -------
    '''
    eol = '\r' if carriage_return else '\n'
    print ('[Testing] >> %2.2f%%' % (percentage_complete)
           if epoch_id is None
           else
           '[learning] epoch %i >> %2.2f%%' % (epoch_id, percentage_complete)),
    print 'completed in %.2f (sec) <<%s' % (time.time() - tic, eol),
    sys.stdout.flush()
    return

def duplicate_middle_word(words):
    d = words.ndim
    assert d == 3
    return numpy.concatenate(
        (words, words), axis=d-2)

def duplicate_label(labels):
    d = labels.ndim
    assert d == 2
    return numpy.concatenate(
        (labels, labels), axis=d-1)

def deduplicate_label(labels):
    d = labels.ndim
    assert d == 2 and labels.shape[d-1] == 2
    return labels[:, :1]

def remove_int_at_end(s):
    try:
        return re.match('(.*)_\d+', s).group(1)
    except AttributeError:
        return s

@contextlib.contextmanager
def config_overide(msg, args):
    assert ' ' not in msg
    args.folder = args.folder + '_' + msg
    rasengan.warn('NOTE: I set args.folder to ' + args.folder)
    yield
    pass
