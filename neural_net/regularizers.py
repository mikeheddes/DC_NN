import numpy as np


class Dropout(object):
    """docstring for Dropout."""

    def __init__(self, p):
        dropout_layer = (np.random.rand(*X.shape) < p) / p
        return X * dropout_layer

    def get_config(self):
        pass


class L2_norm(object):
    """docstring for L2_norm."""

    def __init__(self, arg):
        L2 = (1 - eta * lmbda / dataset_size)
        return W * L2

    def get_config(self):
        pass
