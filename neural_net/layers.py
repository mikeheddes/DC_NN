import numpy as np

from .optimizers import *
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

    def fn(self, x, **kwarg):
        return np.dot(x, self.W) + self.b

    def prime(self, error, activation_1, W1):
        deltaW = np.dot(activation_1.T, error)
        return error, deltaW, self.W

    def get_config(self):
        return{}


class Dropout(object):
    """docstring for Dropout."""

    def __init__(self, rate=0.5):
        self.output_shape = None
        self.input_shape = None
        self.rate = min(1., max(0., rate))
        self.options = {'TRAIN': self.train_fn,
                        'PREDICT': self.predict_fn}

    def fn(self, x, flag):
        return self.options[flag](x)

    def train_fn(self, x):
        return x / self.rate * np.less(np.random.rand(*x.shape), self.rate)

    def predict_fn(self, x):
        return x

    def prime(self, error, activation_1, W1):
        return error, None, W1

    def get_config(self):
        return{}


class Input(object):

    def __init__(self, shape):
        self.output_shape = shape
        self.input_shape = shape
