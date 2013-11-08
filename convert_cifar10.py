#!/usr/bin/env python

import cPickle
import gzip
import sys
import os.path
from pprint import pprint

import numpy
import theano

sys.path.append('tutorial_code')

from logistic_sgd import load_data
from showPklGz import showDataset, theanoTensor2NumpyArray

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def convert_cifar10(output='data/cifar10_dlt_compatible.pkl.gz'):
    input_dir = 'data/cifar-10-batches-py'
    datafiles = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    for datafile in datafiles:
        input_path = os.path.join(input_dir, datafile)
        print('read ' + input_path)
    #datasets = load_data(input)
    #
    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]
    #
    #train_len = 500
    #valid_len = 100
    #test_len = 100
    #
    #minified_train_x = theanoTensor2NumpyArray(train_set_x[0:train_len:1])
    #minified_valid_x = theanoTensor2NumpyArray(valid_set_x[0:valid_len:1])
    #minified_test_x = theanoTensor2NumpyArray(test_set_x[0:test_len:1])
    #
    #minified_train_y = theanoTensor2NumpyArray(train_set_y[0:train_len:1])
    #minified_valid_y = theanoTensor2NumpyArray(train_set_y[0:valid_len:1])
    #minified_test_y = theanoTensor2NumpyArray(train_set_y[0:test_len:1])
    #
    #data = (minified_train_x, minified_train_y),(minified_valid_x, minified_valid_y), (minified_test_x, minified_test_y)
    #
    #f = gzip.open(output, 'wb')
    #cPickle.dump(data, f, -1)
    #f.close()
    #
    print('output to ' + output)


if __name__ == '__main__':
    convert_cifar10()
