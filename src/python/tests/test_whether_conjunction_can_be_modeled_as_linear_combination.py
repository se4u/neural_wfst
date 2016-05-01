# Filename: test_whether_conjunction_can_be_modeled_as_linear_combination.py
# Description:
# Author: Pushpendre Rastogi
# Created: Thu May 21 22:29:26 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 1
f = open('/export/projects/prastogi/nmcr/store/absmo_chunking_embeddings-senna.txt.filtered.scaled~100~1_2015-05-13-01-44-21-191332614913219.parameters.pickle'); import cPickle as pickle; d = pickle.load(f)
y = d['tparam_conjunction_T_20'][:23, :].reshape((529,))
import numpy as np
A = np.zeros((529, 46))
import itertools
for r,(i,j) in enumerate(itertools.product(range(23), range(23))):
  A[r, i]=1
  A[r, 23+j]=1
x, residuals, rank, s = np.linalg.lstsq(A, m, rcond=1e-10)
np.linalg.norm(np.dot(A, x) - y)
# 24.565864873085285
np.linalg.norm(y)
# 45.649647
P = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
np.linalg.norm(np.dot(P, y) - y)
# 30.6414214414154
np.linalg.norm(y)
# 45.649647
np.linalg.det(np.linalg.inv(np.dot(A.T, A)))
# 2.267165461018736e-47
np.linalg.det(np.linalg.inv(np.dot(A.T, A) + 1e-8 * np.eye(46)))
# 2.637651281942231e-54
np.linalg.det(np.linalg.inv(np.dot(A.T, A) + 1e-4 * np.eye(46)))
# 2.6371410384222165e-58
(np.linalg.det(np.dot(A.T, A)/23 + 1e-4 * np.eye(46)))
# 0.00020089193874658396
np.linalg.svd(np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)/23 + 1e-8 * np.eye(46))/23, A.T)))
y_hat = np.dot((np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)/23 + 1e-8 * np.eye(46))/23, A.T))) , y)
np.linalg.norm(y_hat - y)
# 24.565864873085292
np.linalg.norm(y_hat)
# 38.476067053816017
np.linalg.norm(y)
# 45.649647

# Since the norm of the vector orthogonal to the subspace is large
# therefore, we can't hope to approximate these 529 numbers by just a linear combination of 46 numbers.
