import numpy as np
import numpy.random as npr
from numpy import zeros, zeros_like, ones, empty, exp, log
from transducer import Transducer
from features import Features
from argparse import ArgumentParser
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import codecs, sys
import cProfile, pstats

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


def read_data_seg(file_name):
    """
    Helper function
    """
    lst = []
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            one, two = line.split("\t")
            two = two.replace(" ", "")
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
    p = ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test')
    p.add_argument('--dev')
    # parse the arguments
    args = p.parse_args()


    # read in the data
    size = 80
    #data = read_data_seg(args.train)
    data = read_data(args.train)

    #npr.shuffle(data)
    data = data[0:size]
    train_str = data[:int(size*1.0)]
    test_str = data[int(size*1.0):]

    #train_str = [(u'jason', u'eisner')]

    print len(train_str)
    print len(test_str)
    #import sys; sys.exit(0)
    # alphabet
    Sigma = { "" : 0 }
    for one, two in data:
        for c in list(one):
            if c not in Sigma:
                Sigma[c] = len(Sigma)
        for c in list(two):
            if c not in Sigma:
                Sigma[c] = len(Sigma)

    Sigma_inv = {}
    for x, y in Sigma.items():
        Sigma_inv[y] = x

    # features
    features = Features(Sigma, Sigma_inv)
    for upper, lower in train_str:
        print upper
        features.extract(upper, create=True)
    for upper, lower, in test_str:
        features.extract(upper, create=False)
    """
    seen = set([])
    for i in xrange(len("jason")+1):
        for j in xrange(len(Sigma)):
            dell = features.extracted[0][i][j][0][1]

<<<<<<< HEAD
# parse the arguments
args = p.parse_args()


# read in the data
size = 320
#data = read_data_seg(args.train)
data = read_data(args.train)

#npr.shuffle(data)
data = data[0:size]
train_str = data[:int(size*0.8)]
test_str = data[int(size*0.8):]

#train_str = [(u'jason', u'eisner')]

print len(train_str)
print len(test_str)
#import sys; sys.exit(0)
# alphabet
Sigma = { "" : 0 }
for one, two in data:
    for c in list(one):
        if c not in Sigma:
            Sigma[c] = len(Sigma)
    for c in list(two):
        if c not in Sigma:
            Sigma[c] = len(Sigma)

Sigma_inv = {}
for x, y in Sigma.items():
    Sigma_inv[y] = x

# features
features = Features(Sigma, Sigma_inv)
for upper, lower in train_str:
    print upper
    features.extract(upper, create=True)
for upper, lower, in test_str:
    print upper
    features.extract(upper, create=False)
"""
seen = set([])
for i in xrange(len("jason")+1):
    for j in xrange(len(Sigma)):
        dell = features.extracted[0][i][j][0][1]
        
        #if len(dell) == 0:
        #    print i, j
        #    print dell

        for k in xrange(len(Sigma)):
            sub = features.extracted[0][i][j][k][0]
            inv = features.extracted[0][i][j][k][2]

        if len(inv) == 0:
            print i, j, k
            print inv


#print features.extracted
import sys; sys.exit(0)
"""
# test training data
print "numerizing"
train = numerize(train_str, Sigma)
test = numerize(test_str, Sigma)


# number of total insertions per string
INSERTION_LIMIT = 3

# transducer
t = Transducer(len(Sigma), INSERTION_LIMIT, features)
#string1 = train[0][0]
#string2 = train[0][1]

theta = ones((features.num_features)) * -.1 
theta[0] = 1.0
#theta = npr.rand(features.num_features)

threshold = 100
def f(theta):
    val = 0.0
    for i, (x, y) in enumerate(train):
        val += t.func_features(x, y, i, theta, features, threshold)
    return val
    #return np.asarray(t.func_features(string1, string2, 0, theta, features))
=======
            #if len(dell) == 0:
            #    print i, j
            #    print dell
>>>>>>> 76148e8887cbf535c1574c441fac8eecd4f467d5

            for k in xrange(len(Sigma)):
                sub = features.extracted[0][i][j][k][0]
                inv = features.extracted[0][i][j][k][2]

<<<<<<< HEAD
    for i, (x, y) in enumerate(train):
        t.forward_tropical_features(x, i, theta, features, 100000000000000000)
=======
            if len(inv) == 0:
                print i, j, k
                print inv
>>>>>>> 76148e8887cbf535c1574c441fac8eecd4f467d5


    #print features.extracted
    import sys; sys.exit(0)
    """
    # test training data
    train = numerize(train_str, Sigma)
    test = numerize(test_str, Sigma)

<<<<<<< HEAD
def g(theta):
    theta_g = zeros_like(theta)
    for i, (x, y) in enumerate(train):
        t.grad_features(x, y, i, theta, theta_g, features, threshold)
    return theta_g
=======
>>>>>>> 76148e8887cbf535c1574c441fac8eecd4f467d5

    # number of total insertions per string
    INSERTION_LIMIT = 5

    # transducer
    t = Transducer(len(Sigma), INSERTION_LIMIT, features)
    #string1 = train[0][0]
    #string2 = train[0][1]

    theta = zeros((features.num_features))
    theta[0] = 10.0
    #theta = npr.rand(features.num_features)

    def f(theta):
        val = 0.0
        for i, (x, y) in enumerate(train):
            val += t.func_features(x, y, i, theta, features, 20)
        return val
        #return np.asarray(t.func_features(string1, string2, 0, theta, features))

    def f_tropical(theta):

<<<<<<< HEAD
#for i in xrange(100):
#    theta -= 0.001 * g(theta)
#    print f(theta)

#cProfile.runctx("lbfgs(f, theta, fprime=g, disp=2, maxiter=10)", globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

#cProfile.runctx("f_tropical(theta)", globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

# train

def adagrad():
    theta = ones((features.num_features)) * -.1 
    theta[0] = 1.0


    master_stepsize = 1e-1 #for example
    fudge_factor = 1e-6 #for numerical stability
    historical_grad = 0

    for iteration in xrange(10):
        print "iteration", iteration
        for i, (x, y) in enumerate(train):
            print i
            theta_g = zeros_like(theta)
            t.grad_features(x, y, i, theta, theta_g, features, threshold)
            historical_grad += theta_g * theta_g
            adjusted_grad = theta_g / (fudge_factor + np.sqrt(historical_grad))
            
            theta -= master_stepsize * adjusted_grad
            
        print f(theta)

    return theta
#cProfile.runctx("adagrad()", globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

theta = adagrad()
#theta, _, _ = lbfgs(f, theta, fprime=g, disp=2, maxiter=10)

#import sys; sys.exit(0)
#w, _, _ = lbfgs(f, w, fprime=g, disp=2, maxiter=100)
#W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
=======
        for i, (x, y) in enumerate(train):
            t.forward_tropical_features(x, i, theta, features, 0.0)


>>>>>>> 76148e8887cbf535c1574c441fac8eecd4f467d5

    def g(theta):
        theta_g = zeros_like(theta)
        for i, (x, y) in enumerate(train):
            t.grad_features(x, y, i, theta, theta_g, features)
        return theta_g


    """
    w = np.zeros(((len(string1)+1)*len(Sigma)*len(Sigma)*3))

    def f(w):
        W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
        return t.func(string1, string2, W)

    def g(w):
        W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))
        W_grad = np.asarray(t.grad(string1, string2, W))
        return W_grad.reshape(((len(string1)+1)*len(Sigma)*len(Sigma)*3))
    """

    #f_tropical(theta)
    #import sys; sys.exit(0)

    cProfile.runctx("f(theta)", globals(), locals(), '.prof')
    s = pstats.Stats('.prof')
    s.strip_dirs().sort_stats('time').print_stats(30)

    cProfile.runctx("f_tropical(theta)", globals(), locals(), '.prof')
    s = pstats.Stats('.prof')
    s.strip_dirs().sort_stats('time').print_stats(30)

    import sys; sys.exit(0)
    theta, _, _ = lbfgs(f, theta, fprime=g, disp=2, maxiter=100)
    #w, _, _ = lbfgs(f, w, fprime=g, disp=2, maxiter=100)
    #W = w.reshape((len(string1)+1, len(Sigma), len(Sigma), 3))

<<<<<<< HEAD
for i, (x, y) in enumerate(train):
    score, decoded = t.decode_features(x, i, theta, features, True)
    guess = int2str(decoded, Sigma_inv)
    print int2str(x, Sigma_inv), "->", guess, int2str(y, Sigma_inv) , guess == int2str(y, Sigma_inv)
=======
    """
    for iteration in xrange(30):
        print iteration
        #npr.shuffle(train)
        for i, (x, y) in enumerate(train):
            theta_g = zeros_like(theta)

            t.grad_features(x, y, i, theta, theta_g, features)
            theta -= theta_g
            #theta -= np.asarray(theta_g)
    """

    print "DECODING"

    for i, (x, y) in enumerate(train):
        score, decoded = t.decode_features(x, i, theta, features, True)
        print int2str(x, Sigma_inv), "->", int2str(decoded, Sigma_inv)
>>>>>>> 76148e8887cbf535c1574c441fac8eecd4f467d5


    print "TESTING"
    correct = 0
    total = 0
    for i, (x, y) in enumerate(test):
        score, decoded = t.decode_features(x, i+len(train), theta, features, True)
        #int2str(x, Sigma_inv), "->",
        guess = int2str(decoded, Sigma_inv)
        truth = int2str(y, Sigma_inv)

        if guess == truth:
            correct += 1
        else:
            print guess, truth
        total += 1

    print "ACC", float(correct) / total

        #print int2str(x, Sigma_inv), "->", int2str(t.decode(x, W), Sigma_inv)
    #samples = t.sample(string1, Sigma_inv, W, 100)
    #for s, p in samples:
    #    print int2str(s, Sigma_inv), p
    #import sys; sys.exit(0)



    # theta = npr.rand(features.num_features)
    # print t.func_features(string1, string2, 0, theta, features)
    # theta_g = t.grad_features(string1, string2, 0, theta, features)
    # theta_g_fd = zeros((features.num_features))


    # EPS = 0.001
    # for i in xrange(features.num_features):
    #     theta[i] += EPS
    #     val1 = t.func_features(string1, string2, 0, theta, features)
    #     theta[i] -= 2*EPS
    #     val2 = t.func_features(string1, string2, 0, theta, features)
    #     theta[i] += EPS

    #     theta_g_fd[i] = (val1 - val2) / (2 * EPS)

    # for i in xrange(features.num_features):
    #     print theta_g[i], theta_g_fd[i]
    # print np.allclose(theta_g, theta_g_fd, atol=0.1)
    # import sys; sys.exit(0)

    # parameter tensor
    # W = np.zeros((len(string1)+1, len(Sigma), len(Sigma), 3))
    # W = np.random.rand(len(string1)+1, len(Sigma)+1, len(Sigma)+1, 3)

    # unit tests
    # t.fd_check(string1, string2, W)
    # t.enumerate_func(string1, string2, W)

    # check training
