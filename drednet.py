#!/usr/bin/env python

import cPickle
import gzip
import sys
from pprint import pprint

import numpy
import theano

sys.path.append('tutorial_code')

from logistic_sgd import LogisticRegression, load_data


if __name__ == '__main__':
    print('a')
