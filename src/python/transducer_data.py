'''
| Filename    : transducer_data.py
| Description : Functions that return the data fe to the transducer.
| Author      : Pushpendre Rastogi
| Created     : Tue Dec  8 17:50:51 2015 (-0500)
| Last-Updated: Thu Dec 31 01:08:44 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 22
'''
import codecs
import numpy
import string
import rasengan
import util_lstm_seqlabel
import warnings

BOS_CHAR = '^'
def read_data(file_name):
    """
    Helper function
    """
    lst = []
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            one, two = line.split("\t")
            lst.append((one, two))
    return lst

def numerize(lst, Sigma, win):
    " Takes the string-valued training data and interns it "
    lst_prime = []
    bos_idx = len(Sigma)
    for one, two in lst:
        one_prime = numpy.asarray(
            util_lstm_seqlabel.conv_x(
                [Sigma[x] for x in one], win, bos_idx),
            dtype=numpy.int32)
        two_prime = numpy.asarray(
            [Sigma[x] for x in two],
            dtype=numpy.int32)
        lst_prime.append((one_prime, two_prime))
    return lst_prime

def int2str(lst, Sigma_inv):
    " Converts a list of integers to a string "
    _string = ""
    for x in lst:
        _string += Sigma_inv[x]
    return _string

def get_lst_char(data_tuple_list):
    lst_char = list(set(reduce(
            lambda x, y: x + y[0] + y[1], data_tuple_list, '')))
    for e in list(set(string.letters.lower())):
        e = unicode(e)
        if e not in lst_char:
            lst_char.append(e)
    assert BOS_CHAR not in lst_char
    lst_char.insert(0, BOS_CHAR)
    return lst_char

def add_bos(data_tuple_list):
    '''
    The BOS_CHAR is added to the left portion of the data, that is transduced
    so that my LSTM can produce (1 + length) dimensional tensor, which is then
    used by the cython transducer.
    '''
    return [(BOS_CHAR + a, b) for a,b in data_tuple_list]

def main(args):
    with rasengan.debug_support():
        with rasengan.tictoc("Loading Data"):
            data_list = rasengan.namespacer(
                read_data(args.train_fn))
            val_data_list = rasengan.namespacer(
                read_data(args.dev_fn))
            if args.partition_dev_into_train > 0:
                lim = args.partition_dev_into_test
                data_list.extend(val_data_list[lim:])
                val_data_list = val_data_list[:lim]

            if args.partition_dev_into_test > 0:
                lim = args.partition_dev_into_test
                test_data_list = val_data_list[lim:]
                val_data_list = val_data_list[:lim]
            else:
                test_data_list = rasengan.namespacer(
                    read_data(args.test_fn))

            # data_list = val_data_list = [(u'jason', u'eisner')]
            lst_char = get_lst_char(data_list
                                    + val_data_list
                                    + test_data_list)
            data_list = add_bos(data_list)
            val_data_list = add_bos(val_data_list)
            test_data_list = add_bos(test_data_list)
            warnings.warn('''
            NOTE: While preparing sigma, we add 1 to the index
            returned by enumerate because the transducer unit that
            Ryan wrote uses index 0 as the index for the epsilon
            symbol. So essentially the epsilon symbol and the
            integer 0 are reserved symbols that cannot appear in the
            vocabulary.

            ALSO, we need to add 1 to the vocsize because of that.
            ''')
            # sigma :: char -> int
            sigma = dict((b, a+1) for (a,b) in enumerate(lst_char))

            # sigma_inv :: int -> char
            sigma_inv = dict((a+1, b) for (a,b) in enumerate(lst_char))

            if args.limit_corpus > 0:
                data_list = data_list[:args.limit_corpus]

            train_data = numerize(data_list, sigma, args.win)
            val_data = numerize(val_data_list, sigma, args.win)
            test_data = numerize(test_data_list, sigma, args.win)

            data = rasengan.Namespace()

            #-------------------------------------------------------------#
            # Add sets that would be used by the tensorflow seq2seq       #
            # model. See~$PY/tensorflow/models/rnn/translate/translate.py #
            #-------------------------------------------------------------#
            data.train_data = data_list
            data.val_data = val_data_list
            data.test_data = test_data_list

            data.train_set = train_data
            data.dev_set = val_data
            data.test_set = test_data

            data.vocsize = len(sigma) + 1
            data.idx2label = sigma_inv
            data.label2idx = sigma

            data.train_lex = [e[0] for e in train_data]
            data.train_y = [e[1] for e in train_data]

            data.valid_lex = [e[0] for e in val_data]
            data.valid_y = util_lstm_seqlabel.convert_id_to_word(
                [e[1] for e in val_data], data.idx2label)

            data.test_lex = [e[0] for e in test_data]
            data.test_y = util_lstm_seqlabel.convert_id_to_word(
                [e[1] for e in test_data], data.idx2label)

            data.words_train = []
            data.words_valid = []
            data.words_test = []
    return data
