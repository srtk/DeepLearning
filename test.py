#!/usr/bin/env python

import cPickle
import gzip
import os
import sys
import time
from pprint import pprint as pp

import numpy
import theano
import theano.tensor as T
from theano.printing import pprint as tpp

sys.path.append('tutorial_code')

from showPklGz import showDataset, theanoTensor2NumpyArray as tt2na

if __name__ == '__main__':
    theano.config.compute_test_value = 'warn'
    v1 = numpy.asarray([[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=theano.config.floatX)
    s1 = theano.shared(value=v1, name="s1", borrow=True)
    v2 = numpy.asarray([[1,0,1,0],[1,0,1,0],[1,0,0,1]],dtype=theano.config.floatX)
    s2 = theano.shared(value=v2, name="s2", borrow=True)
    t3 = T.dot(s1, s2)

    t4 = T.dmatrix('W')
    t5 = T.lvector('i')
    t6 = T.dot(t4, t5)
    f = theano.function([t4, t5], t6)
    print('end')
    print('end2')
