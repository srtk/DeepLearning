#!/usr/bin/env python

import cPickle
import gzip
import sys
import os.path
import cPickle
from pprint import pprint


import numpy
import theano

sys.path.append('tutorial_code')

from logistic_sgd import load_data
from showPklGz import showDataset, theanoTensor2NumpyArray

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#convert to the format loadable with logistic_sgd.load_data
def convert_cifar10(output='data/cifar10_for_dlt.pkl.gz'):
    input_dir = 'data/cifar-10-batches-py'
    if os.path.exists(output):
        print(output + ' already exists. remove it manually for safety')
        return

    def dir_from_file(file):
        input_path = os.path.join(input_dir, file)
        print('read ' + input_path)
        if not os.path.exists(input_path):
            print(input_path + ' not exists')
            exit(1)
        return unpickle(input_path)

    datas = []
    labels = []
    datafiles = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    for datafile in datafiles:
        data_dir = dir_from_file(datafile)
        datas.extend(data_dir['data'])
        labels.extend(data_dir['labels'])

    test_dir = dir_from_file('test_batch')

    print('reformating datas...')

    #description in logistic_sgd.py

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    n_data = 50000
    n_train_set = 41666 #same proportion as mnist 50000 / (50000+10000)

    #MNIST pixel datas vary from 0 to (255.0/256.0), CIFAR-10 from 0 to 255


    train_x = numpy.divide(numpy.asarray(datas[:n_train_set]), 256.)
    valid_x = numpy.divide(numpy.asarray(datas[n_train_set:]), 256.)
    test_x = numpy.divide(numpy.asarray(test_dir['data']), 256.)

    train_y = numpy.asarray(labels[:n_train_set])
    valid_y = numpy.asarray(labels[n_train_set:])
    test_y = numpy.asarray(numpy.asarray(test_dir['labels']))

    train_set = (train_x, train_y)
    valid_set = (valid_x, valid_y)
    test_set = (test_x, test_y)
    to_pickle = (train_set, valid_set, test_set)

    print('start pickling...')
    f = gzip.open(output, 'wb')
    cPickle.dump(to_pickle, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    print('output to ' + output)


if __name__ == '__main__':
    convert_cifar10()
