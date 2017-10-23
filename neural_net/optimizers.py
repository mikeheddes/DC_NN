import numpy as np


class uniform(object):
    def __init__(self, max_value=1, min_value=0):
        self.max_value = max_value
        self.min_value = min_value

    def fn(self, size):
        self.sizes = size
        return (self.max_value - self.min_value) * np.random.random_sample(size=self.sizes) + self.min_value


class normal(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.standard_deviation = std

    def fn(self, size):
        self.sizes = size
        return self.standard_deviation * np.random.standard_normal(size=self.sizes) + self.mean


class zeros(object):

    @staticmethod
    def fn(size):
        return np.zeros(size)


class learning_rate(object):
    def __init__(self, begin, func='CONSTANT', to=None):
        self.begin = begin
        self.to = to
        self.func = func

    def fn(self, epoch, epochs):
        self.epoch = epoch
        self.epochs = epochs
        options = {'CONSTANT': self.constant,
                   'LINEAR': self.linear,
                   'QUADRATIC': self.quadratic}
        return options[self.func]()

    def constant(self):
        return self.begin

    def linear(self):
        return self.begin - (self.begin - self.to) * (self.epoch / (self.epochs - 1 + 1.0e-10))

    def quadratic(self):
        return self.begin - (self.begin - self.to) * (self.epoch / (self.epochs - 1 + 1.0e-10))**2

    def get_config(self):
        pass
