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
from theano.printing import debugprint as tdp

sys.path.append('tutorial_code')

from my_theano_util import theanoTensor2NumpyArray as tt2na

if __name__ == '__main__':
    #theano.config.compute_test_value = 'warn'
    v1 = numpy.asarray([[1,2,3,4],[1,2,3,4],[1,2,3,4]],dtype=theano.config.floatX)
    s1 = theano.shared(value=v1, name="s1", borrow=True)
    v2 = numpy.asarray([[1,0,1,0],[1,0,1,0],[1,0,0,1]],dtype=theano.config.floatX)
    s2 = theano.shared(value=v2, name="s2", borrow=True)
    #t3 = T.dot(s1, s2)

    #t4 = T.dmatrix('W')
    #t4.tag.test_value = numpy.random.rand(5, 10).astype(theano.config.floatX)
    #t5 = T.lvector('i')
    #t5.tag.test_value = numpy.random.rand(5, 10).astype(theano.config.floatX)
    #t6 = T.dot(t4, t5)

    x = T.dmatrix('x')
    theano_rng = theano.tensor.shared_randomstreams.RandomStreams()
    rand = theano_rng.binomial(n=1, p=0.5, size=x.shape, dtype=theano.config.floatX)

    rx = rand * (x ** 2) * 0.5
    c = rx.sum()
    dy = T.grad(c, x)
    pdy = theano.printing.Print()(dy)
    m = T.max(pdy)
    f2 = theano.function([x], m)
    input = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    out2 = f2(input)
    pp(out2)
    print('end')
    print('end2')
