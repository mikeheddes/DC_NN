import numpy as np

from .optimizers import *
from .regularizers import *


class Dense(object):
    """Most basic layer"""

    def __init__(self, output_size,
                 activation=None,
                 weight_init=normal(),
                 bias_init=zeros(),
                 dropout=None):
        self.output_size = output_size
        self.activation = activation
        self.W_init = weight_init
        self.b_init = bias_init
        self.dropout = dropout
        self.b = self.b_init.fn((self.output_size, 1))
        self.Z = np.zeros((self.output_size, 1))
        self.A = np.zeros((self.output_size, 1))

    def fn(self, X):
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation.fn(self.Z)
        # Moet niet hier in komen maar als apparte laag
        # if self.dropout is not None:
        #     self.A = Dropout(self.A, self.dropout)

    def prime(self, E, W1, A_1):
        error = np.dot(W1.T, E) * self.activation.prime(self.Z)
        change_W = np.dot(error, A_1.T)
        return error, change_W

    def get_config(self):
        return{}


class Dropout(object):
    """docstring for Dropout."""

    def __init__(self, p):


class Input(object):
    def __init__(self, size):
        self.output_size = size
        self.A = np.zeros((size, 1))
        self.b = np.zeros(1)
        self.W = np.zeros(1)
