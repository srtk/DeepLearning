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

from showPklGz import showDataset, theanoTensor2NumpyArray as tt2na


#theano.function(inputs, output)(input_value)でOK

#def printValue(inputs, output, input_value):
#    #out_name = output.name if output.name else 'printValue'
#    #p = theano.printing.Print(out_name)(output)
#    pf = theano.function(inputs, output)
#    result = pf(input_value)
#    pp(result)
