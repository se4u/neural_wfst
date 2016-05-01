import numpy as np
import numpy.random as npr
from numpy import zeros, zeros_like, ones, empty, exp, log
from transducer import Transducer
from argparse import ArgumentParser
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import codecs, sys
import cProfile, pstats
from features import Features


def read_data(file_name):
    """
    Helper function
    """
    lst = []
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            one, two = line.split("\t")
            lst.append((one, two))
    return lst

def numerize(lst, Sigma):
    " Takes the string-valued training data and interns it "
    lst_prime = []
    for one, two in lst:
        one_prime = np.asarray([ Sigma[x] for x in list(one) ], dtype=np.int32)
        two_prime = np.asarray([ Sigma[x] for x in list(two) ], dtype=np.int32)
        lst_prime.append((one_prime, two_prime))
    return lst_prime


def int2str(lst, Sigma_inv):
    " Converts a list of integers to a string "
    string = ""
    for x in lst:
        string += Sigma_inv[x]
    return string

if __name__ == '__main__':
    # p = ArgumentParser()
    # p.add_argument('--train', required=True)
    # p.add_argument('--test')
    # p.add_argument('--dev')
    # parse the arguments
    # args = p.parse_args()


    # read in the data
    # size = 80
    # data = read_data_seg(args.train)
    # data = read_data(args.train)

    # npr.shuffle(data)
    # data = data[0:size]
    # train_str = data[:int(size*0.8)]
    # test_str = data[int(size*0.8):]

    train_str = [(u'jason', u'eisner')]
    #import sys; sys.exit(0)
    # alphabet
    Sigma = { "" : 0 }
    for one, two in train_str:
        for c in list(one):
            if c not in Sigma:
                Sigma[c] = len(Sigma)
        for c in list(two):
            if c not in Sigma:
                Sigma[c] = len(Sigma)

    Sigma_inv = {}
    for x, y in Sigma.items():
        Sigma_inv[y] = x


    # test training data
    train = numerize(train_str, Sigma)

    # number of total insertions per string
    INSERTION_LIMIT = 3
    
    # transducer
    t = Transducer(len(Sigma), INSERTION_LIMIT)
    string1 = train[0][0]
    string2 = train[0][1]


    features = Features(Sigma, Sigma_inv)
    for upper, lower in train_str:
        #print upper, lower, len(features.features)
        features.extract(upper, URC=0, ULC=0, create=True)

    # get tensor
    # This is equivalent to the earlier tensor.
    # tensor_features is a list of sparse W tensors.
    # Every element of tensor_feature is 5 dimensional
    # where the first 4 are the same as the W tensor.
    # And the last dimension is feature_index into set of features.
    # To embed these features I would get their embeddings and add
    # them.
    # for k, v in features.features.items(): print k, v
    # SUB(o,e)|LLC(j) 432
    # SUB(n,) 470
    # INS(a)|LLC(n) 27
    # COPY|LLC(e) 117
    # DEL(j)|LLC(e) 98
    # INS(n)|LLC(o) 56
    # SUB(n,i)|LLC(e) 537
    tensor_features = []

    extracted = features.extracted
    for i in xrange(len(train_str)):
        tensor_feature = np.zeros((len(train_str[i][0])+1, len(Sigma), len(Sigma), 3, features.num_features))
        for j in xrange(len(train_str[i][0])+1):
            for x in xrange(len(Sigma)):
                for y in xrange(len(Sigma)):
                    for a in xrange(3):
                        for feat in extracted[i][j][x][y][a]:
                            tensor_feature[j, x, y, a, feat] = 1.0


    w = np.zeros(((len(string1)+1)*len(Sigma)*len(Sigma)*3))
    
    def f(w):
        W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
        return t.func(string1, string2, W)

    def g(w):
        W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
        W_grad = np.asarray(t.grad(string1, string2, W))
        return W_grad.reshape(((len(string1)+1)*len(Sigma)*len(Sigma)*3))
        
    
    w, _, _ = lbfgs(f, w, fprime=g, disp=2, maxiter=1000)

    W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
    print int2str(string1, Sigma_inv), "->", int2str(t.decode(string1, W), Sigma_inv)
