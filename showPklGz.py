#!/usr/bin/env python

import cPickle
import gzip
import sys
from pprint import pprint

import numpy
import theano

sys.path.append('tutorial_code')

from logistic_sgd import load_data
from my_theano_util import theanoTensor2NumpyArray

def showDataset(dataset='data/mnist.pkl.gz'):
    """
    :type dataset: string
    :param dataset: path the the pickled dataset
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #type(train_set_x) => <class 'theano.tensor.sharedvar.TensorSharedVariable'>
    np_train_x = train_set_x.get_value(borrow=True)
    np_valid_x = valid_set_x.get_value(borrow=True)
    np_test_x = test_set_x.get_value(borrow=True)

    #type(train_set_y) => <class 'theano.tensor.basic.TensorVariable'>
    np_train_y = theanoTensor2NumpyArray(train_set_y)
    np_valid_y = theanoTensor2NumpyArray(valid_set_y)
    np_test_y = theanoTensor2NumpyArray(test_set_y)

    for varName in "np_train_x[0] np_train_y np_valid_x[0] np_valid_y np_test_x[0] np_test_y".split(" "):
        var = eval(varName)
        print(varName)
        print("len:" + str(len(var)))
        pprint(var)


if __name__ == '__main__':
    input="data/mnist.pkl.gz"
    if(len(sys.argv) >= 2):
        input=sys.argv[1]
    showDataset(input)
