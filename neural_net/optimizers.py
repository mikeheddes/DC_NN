import numpy as np


class uniform(object):
    def __init__(self, max_value=1, min_value=0):
        self.max_value = max_value
        self.min_value = min_value

    def fn(self, size):
        return (self.max_value - self.min_value) * np.random.rand(*sizes) + self.min_value


class normal(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.standard_deviation = std

    def fn(self, size):
        return self.standard_deviation * np.random.randn(*size) + self.mean


class variance(object):
    def __init__(self, mean=0.):
        self.mean = mean

    def fn(self, size):
        return np.random.randn(*size) * np.sqrt(2. / np.sum(size)) + self.mean


class zeros(object):

    @staticmethod
    def fn(size):
        return np.zeros(size)


class learning_rate(object):
    def __init__(self, begin, func='CONSTANT', to=None):
        self.begin = begin
        self.to = to
        self.func = func
        self.options = {'CONSTANT': self.constant,
                        'LINEAR': self.linear,
                        'QUADRATIC': self.quadratic}

    def fn(self, epoch, epochs):
        self.epoch = epoch
        self.epochs = epochs
        return self.options[self.func]()

    def constant(self):
        return self.begin

    def linear(self):
        return self.begin - (self.begin - self.to) * (self.epoch / (self.epochs - 1. + 1.0e-10))

    def quadratic(self):
        return self.begin - (self.begin - self.to) * (self.epoch / (self.epochs - 1. + 1.0e-10))**2.

    def get_config(self):
        pass


class L1L2(object):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = float(l1)
        self.l2 = float(l2)

    def fn(self, x):
        regularization = 0.
        if self.l1:
            regularization += np.sum(self.l1 * np.absolute(x))
        if self.l2:
            regularization += np.sum(self.l2 * np.square(x))
        return regularization

    def prime(self, x, LR, dataset_size):
        x_ = x
        if self.l1:
            x_ -= LR * self.l1 / dataset_size * np.greater_equal(x, 0)
        if self.l2:
            x_ -= LR * self.l2 / dataset_size * x
        return x_

    def get_config(self):
        return {'l1': self.l1,
                'l2': self.l2}


def L1(l=5.):
    return L1L2(l1=l)


def L2(l=5.):
    return L1L2(l2=l)


def L1_L2(l1=5., l2=5.):
    return L1L2(l1=l1, l2=l2)
