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
from my_python_util import getVarNames

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


    np_train_x0 = np_train_x[0]
    np_valid_x0 = np_valid_x[0]
    np_test_x0 = np_test_x[0]

    varNames = "np_train_x[0] np_train_y np_valid_x[0] np_valid_y np_test_x[0] np_test_y".split(" ")
    for varName in varNames:
        var = eval(varName)
        print(varName)
        print("len:" + str(len(var)))
        pprint(var)

    for varName in varNames: #print again for convenience
        var = eval(varName)
        print(varName + " len:" + str(len(var)))

    vars = [np_train_x0, np_train_y, np_valid_x0, np_valid_y, np_test_x0, np_test_y]
    var_names = locals()
    for var in vars:
        name = getVarNames(var, var_names)
        print(str(name) + ' start')
        pprint(var)
        print("len:%d max:%f min:%f"%(len(var), numpy.max(var), numpy.min(var)))
        print(str(name) + ' end')

    pairs = [(np_train_x, np_train_y), (np_valid_x, np_valid_y), (np_test_x, np_test_y)]
    for pair in pairs:
        if not (len(pair[0]) == len(pair[1])):
            name_x = getVarNames(pair[0], locals())
            name_y = getVarNames(pair[1], locals())
            print("WARNING: the lengths of %s & %s are different" % (name_x, name_y))



if __name__ == '__main__':
    input="data/mnist.pkl.gz"
    if(len(sys.argv) >= 2):
        input=sys.argv[1]
    showDataset(input)
