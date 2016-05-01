"""
Feature extractor for the transduction
"""
from arsenal.alphabet import Alphabet
from libcpp.vector cimport vector

cdef class Features(object):
    """ 
    Extract features for transducer
    """

    cdef object Sigma, Sigma_inv, features, str2int
    cdef int num_extracted, counter
    cdef vector[vector[vector[vector[vector[vector[int]]]]]] extracted

    def __init__(self, Sigma, Sigma_inv):
        self.Sigma = Sigma
        self.Sigma_inv = Sigma_inv
        self.features = Alphabet()
        self.str2int = Alphabet()
        self.num_extracted = 0
        self.counter = 0

        # TODO: macro to clean up
        self.extracted = vector[vector[vector[vector[vector[vector[int]]]]]]()
        self.features.add("COPY")


    def _left_context(self, i, string, size):
        " gets the right context "
        return string[max(0, i-size):i]


    def _right_context(self, i, string, size, offset=0):
        " gets the left context "
        return string[i+offset:min(i+size+offset, len(string)+offset)]


    def extract(self, string_in, ULC=0,  URC=0, create=False):
        " Extract the features "

        cdef int strid, i, j, k, featid
        self.str2int.add(string_in)
        strid = self.str2int[string_in]

        # augment feature data structure
        self.extracted.push_back(vector[vector[vector[vector[vector[int]]]]]())

        for i in xrange(len(string_in)+1):
            self.extracted[strid].push_back(vector[vector[vector[vector[int]]]]())
            for j in xrange(len(self.Sigma)):
                self.extracted[strid][i].push_back(vector[vector[vector[int]]]())
                for k in xrange(len(self.Sigma)):
                    self.extracted[strid][i][j].push_back(vector[vector[int]]())
                    for l in xrange(3):
                        self.extracted[strid][i][j][k].push_back(vector[int]())

        for i in xrange(len(string_in)+1):
            rcs, lcs = [], []

            for urc in xrange(1, URC):
                rc = self._right_context(i, string_in, urc, offset=1)
                rcs.append(rc)
                
            for ulc in xrange(1, URC):
                lc = self._left_context(i, string_in, ulc)
                lcs.append(lc)

            # deletion
            if i > 0:
                upper = string_in[i-1]
                feat = "DEL(" + upper + ")"
                if create:
                    self.features.add(feat)
                to_add = [ self.features[feat] ]

                for rc in rcs:
                    featn = feat + "|" + "URC(" + rc + ")"
                    if create:
                        self.features.add(featn)
                    to_add.append(self.features[featn])

                for lc in lcs:
                    featn = feat + "|" + "ULC(" + lc + ")"
                    if create:
                        self.features.add(featn)
                    to_add.append(self.features[featn])

                for j in xrange(len(self.Sigma)):
                    x = self.Sigma_inv[j]
                    featn = feat + "|" + "LLC(" + x + ")"
                    if create:
                        self.features.add(featn)
                
                    for featid in to_add:
                        self.extracted[strid][i][j][0][1].push_back(featid)
                    featid = self.features[featn]
                    self.extracted[strid][i][j][0][1].push_back(featid)

                    #self.extracted[strid][i][j][0][1].push_back(self.counter)
                    #self.counter += 1

                
                for k in xrange(len(self.Sigma)):
                    y = self.Sigma_inv[k]
                    # substitution   
                    feat = "SUB(" + upper + "," + y + ")"
                    if upper == y:
                        feat = "COPY"

                    if create:
                        self.features.add(feat)
                    to_add = [ self.features[feat] ]

                    for rc in rcs:
                        featn = feat + "|" + "URC(" + rc + ")"
                        if create:
                            self.features.add(featn)
                        to_add.append( self.features[featn] )

                    for lc in lcs:
                        featn = feat + "|" + "ULC(" + lc + ")"
                        if create:
                            self.features.add(featn)
                        to_add.append( self.features[featn] )


                    for j in xrange(len(self.Sigma)):

                        x = self.Sigma_inv[j]
                        featn = feat + "|" + "LLC(" + x + ")"
                        if create:
                            self.features.add(featn)
                    
                        # add to extracted
                        for featid in to_add:
                            self.extracted[strid][i][j][k][0].push_back(featid)
                        featid = self.features[featn]
                        self.extracted[strid][i][j][k][0].push_back(featid)

                        #self.extracted[strid][i][j][k][0].push_back(self.counter)
                        #self.counter += 1


            for k in xrange(len(self.Sigma)):
                y = self.Sigma_inv[k]

                # insertion
                feat = "INS(" + y + ")"
                if create:
                    self.features.add(feat)
                to_add = [ self.features[feat] ]

                for rc in rcs:
                    featn = feat + "|" + "URC(" + rc + ")"
                    if create:
                        self.features.add(featn)
                    to_add.append(self.features[featn])

                for lc in lcs:
                    featn = feat + "|" + "ULC(" + lc + ")"
                    if create:
                        self.features.add(featn)
                    to_add.append(self.features[featn])

                for j in xrange(len(self.Sigma)):
                    x = self.Sigma_inv[j]
                    featn = feat + "|" + "LLC(" + x + ")"
                    if create:
                        self.features.add(featn)
                   
                    # add to extracted
                    for featid in to_add:
                        self.extracted[strid][i][j][k][2].push_back(featid)
                    featid = self.features[featn]
                    self.extracted[strid][i][j][k][2].push_back(featid)

                    #self.extracted[strid][i][j][k][2].push_back(self.counter)
                    #self.counter += 1

                
    property features:
        def __get__(self):
            return self.features

    property num_features:
        def __get__(self):
            #return self.counter
            return len(self.features)

    property extracted:
        def __get__(self):
            return self.extracted

    property num_extracted:
        def __get__(self):
            return self.num_extracted

    property str2int:
        def __get__(self):
            return self.str2int
