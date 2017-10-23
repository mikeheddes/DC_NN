import numpy as np

from .optimizers import *
from .regularizers import *
from .__init__ import *


class Layer(object):
    """docstring for Layer."""

    def __init__(self, arg):
        pass


class Dense(object):
    """Most basic layer"""

    def __init__(self, output_shape,
                 activation=None,
                 W_init=normal(),
                 b_init=zeros(),
                 input_shape=None):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.activation = activation
        self.W_init = W_init
        self.b_init = b_init

    def compile(self):
        if not self.input_shape:
            raise ValueError('No input_shape specified')
        self.W = self.W_init.fn((self.input_shape, self.output_shape))
        self.b = self.b_init.fn((1, self.output_shape))

    def fn(self, X):
        return np.dot(X, self.W) + self.b

    def prime(self, error, activation_1, W1):
        deltaW = np.dot(activation_1.T, error)
        return error, deltaW, self.W

    def get_config(self):
        return{}


class Input(object):

    def __init__(self, shape):
        self.output_shape = shape
        self.input_shape = shape
