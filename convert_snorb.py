#!/usr/bin/env python

import cPickle
import gzip
import sys
import os.path
import cPickle
from pprint import pprint
import struct
import operator

import numpy as np
import theano

def check_file(file):
    print('reading data from  ' + file)
    if not os.path.exists(file):
        print(file + ' not exists')
        exit(1)


def open_file(file):
    check_file(file)
    return gzip.open(file, 'rb')

def read_header(file):
    magic, ndim = struct.unpack('<ii', file.read(4 * 2))

    n_dim_info = max([ndim, 3]) #see the quotation above
    dims = []
    for i in range(0, n_dim_info):
        dims.extend(struct.unpack('<i', file.read(4)))
    dims = dims[:ndim]
    return magic, ndim, dims


def read_from_datafile(file):
    """
    Each "-dat" file stores 24,300 image pairs
    (5 categories, 5 instances, 6 lightings, 9 elevations, and 18 azimuths).
    The corresponding "-cat" file contains 24,300 category labels
    (0 for animal, 1 for human, 2 for plane, 3 for truck, 4 for car).
    ...
    Note that when the matrix has less than 3 dimensions,
    say, it's a 1D vector, then dim[1] and dim[2] are both 1.
    When the matrix has more than 3 dimensions,
    the header will be followed by further dimension size information.
    """

    #header

    f = open_file(file)
    magic, ndim, dims = read_header(f)

    #data

    n_values_in_data = reduce(operator.mul, dims[1:]) if (len(dims) > 2) else 1
    value_size = 1
    unpack_datatype = 'B'
    unpack_format = '<%d%s' % (n_values_in_data, unpack_datatype)
    datas = []
    #for data_index in range(0, dim[0]):
    for data_index in range(0, 3):
        #the values stored as uint8
        tmp = struct.unpack(unpack_format, f.read(value_size * n_values_in_data))
        datas.append(list(tmp))

    f.close()
    return datas

def read_from_categoryfile(file):
    #header
    f = open_file(file)
    magic, ndim, dims = read_header(f)

    #data
    n_values_in_data = reduce(operator.mul, dims[1:]) if (len(dims) > 2) else 1
    value_size = 4
    unpack_datatype = 'i'
    unpack_format = '<%d%s' % (n_values_in_data, unpack_datatype)
    datas = []
    for data_index in range(0, dims[0]):
        #the values stored as uint32
        tmp = struct.unpack(unpack_format, f.read(value_size * n_values_in_data))
        datas.append(list(tmp))

    f.close()
    return datas

"""
 convert to the format loadable with logistic_sgd.load_data

 although the files have extensions 'mat' they can't be loaded with scipy.io.loadmat
 actually their format is not matlab's but an original one
 see http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/
"""
def convert_small_norb(output='data/smallnorb_for_dlt.pkl.gz'):
    if os.path.exists(output):
        print(output + ' already exists. remove it manually for safety')
        return

    input_dir = 'data'
    train_dat = os.path.join(input_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz')
    train_cat = os.path.join(input_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz')
    train_info = os.path.join(input_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz')
    test_dat = os.path.join(input_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz')
    test_cat = os.path.join(input_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz')
    test_info = os.path.join(input_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz')

    arr_train_dat = read_from_datafile(train_dat)
    arr_train_cat = read_from_categoryfile(train_cat)
    print('bp placehoder')

    exit(0)

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
    convert_small_norb()
