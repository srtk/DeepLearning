#!/usr/bin/env python

import cPickle
import gzip
import os
import sys
import math
from optparse import OptionParser
from pprint import pprint as pp

import numpy
import theano
from theano.printing import pprint as tpp
from theano.printing import debugprint as tdp

sys.path.append('tutorial_code')

from logistic_sgd import load_data
from my_theano_util import theanoTensor2NumpyArray


def minifyDataset(input, output, minify_rate=0.01):
    usage = 'usage: minify_dataset.py <input_file> <output_file>'
    if not input:
        print(usage)
        return False
    if not os.path.exists(input):
        print('file ' + input + ' not found')
        return False
    if not output:
        print(usage)
        return False
    if os.path.exists(output):
        print('output file ' + output + ' already exists')
        return False
    print('read ' + input)
    datasets = load_data(input)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    orig_train_len = len(train_set_x.get_value(borrow=True))
    orig_valid_len = len(valid_set_x.get_value(borrow=True))
    orig_test_len = len(test_set_x.get_value(borrow=True))

    mini_train_len = int(math.floor(orig_train_len * minify_rate))
    mini_valid_len = int(math.floor(orig_valid_len * minify_rate))
    mini_test_len = int(math.floor(orig_test_len * minify_rate))

    minified_train_x = theanoTensor2NumpyArray(train_set_x[0:mini_train_len])
    minified_valid_x = theanoTensor2NumpyArray(valid_set_x[0:mini_valid_len])
    minified_test_x = theanoTensor2NumpyArray(test_set_x[0:mini_test_len])
    
    minified_train_y = theanoTensor2NumpyArray(train_set_y[0:mini_train_len])
    minified_valid_y = theanoTensor2NumpyArray(valid_set_y[0:mini_valid_len])
    minified_test_y = theanoTensor2NumpyArray(test_set_y[0:mini_test_len])
    
    data = ((minified_train_x, minified_train_y),(minified_valid_x, minified_valid_y), (minified_test_x, minified_test_y))

    f = gzip.open(output, 'wb')
    cPickle.dump(data, f, -1)
    f.close()

    print('output to ' + output)
    return True


if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    input = args[0] if args[0] else None
    output = args[1] if args[1] else None
    minifyDataset(input=input, output=output)

