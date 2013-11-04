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
    x = T.dscalar('x')
    y = x ** 2.
    dy = T.grad(y, x)
    f = theano.function([x], dy)
    print('end')
    print('end2')
