"""
Helper module to provide cost functions to the network.
Two types of cost functions with their derivates are available:

- Quadratic
- Cross Entropy
"""
import numpy as np


class quadratic(object):
    @staticmethod
    def fn(y, a):
        return np.power(y - a, 2) / 2

    @staticmethod
    def prime(y, a, z, ac):
        return (a - y) * (ac).prime(z)


class cross_entropy(object):
    @staticmethod
    def fn(y, a):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def prime(y, a, z, ac):
        # print(y, a)
        return a - y
