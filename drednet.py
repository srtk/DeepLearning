#!/usr/bin/env python

import cPickle
import gzip
import sys
from pprint import pprint

import numpy
import theano
import theano.tensor as T

sys.path.append('tutorial_code')

from logistic_sgd import load_data
from showPklGz import showDataset, theanoTensor2NumpyArray
from convolutional_mlp import LeNetConvPoolLayer

class DredNetLayer(object):
    def __init__(self, rng, input):
        self.input = input
        self.output = input
        self.params = []

def test_drednet(learning_rate=0.1, n_epochs=200, input='data/mnist_100.pkl.gz', nkerns=[20, 50], batch_size=50):
    print('read ' + input)
    rng = numpy.random.RandomState(23455)

    datasets = load_data(input)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    ishape = (28, 28)  # this is the size of MNIST images

    layer0_input = x.reshape((batch_size, 1, 28, 28))
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
                                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
    layer1 = DredNetLayer(rng, input=layer0.output)
    pprint(layer1.output)



if __name__ == '__main__':
    test_drednet()
