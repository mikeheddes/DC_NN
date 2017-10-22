"""
Helper module to provide activation to network layers.
Four types of activations with their derivates are available:

- Sigmoid
- Softmax
- Tanh
- ReLU
- Leaky ReLU
"""
import numpy as np


class sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return np.exp(-z) / np.power(1 + np.exp(-z), 2)


class softmax(object):
    @staticmethod
    def fn(z):
        z -= np.amax(z, axis=0, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def prime(z):
        return softmax.fn(z) * (1 - softmax.fn(z))


class tanh(object):
    @staticmethod
    def fn(z):
        return np.tanh(z)

    @staticmethod
    def prime(z):
        return 1 - np.power(np.tanh(z), 2)


class relu(object):
    @staticmethod
    def fn(z):
        return np.maximum(z, 0)

    @staticmethod
    def prime(z):
        return np.greater_equal(z, 0)


class leaky_relu(object):
    @staticmethod
    def fn(z):
        return np.maximum(z, 0) + 0.01 * np.minimum(z, 0)

    @staticmethod
    def prime(z):
        return np.greater_equal(z, 0) + 0.01 * np.less(z, 0)
