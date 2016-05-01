# Filename: test_depparse_grad_speedup.py
# Description: This script basically imports the RectangleDependencyParser and uses it to calculate gradients for given arc_score tensors and then prints the time taken.
# Author: Pushpendre Rastogi
# Created: Mon May 18 15:01:21 2015 (-0400)
# Last-Updated:
#           By:
#     Update #: 1

def main():
    import sys
    from depparse import RectangleDependencyParser
    dp_insideOutside = RectangleDependencyParser.DependencyParser().insideOutside
    n = int(sys.argv[1])
    arc_scores = np.random.randn(n,n,1)

    efficient_arc_count = dp.insideOutside(arc_scores)
