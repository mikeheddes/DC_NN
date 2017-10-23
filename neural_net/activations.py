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


class Activation(object):
    """docstring for Activation."""

    def __init__(self):
        self.input_shape = None
        self.output_shape = None


class sigmoid(Activation):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return np.exp(-z) / np.power(1 + np.exp(-z), 2)


class softmax(Activation):
    @staticmethod
    def fn(z):
        # print(z)
        z -= np.amax(z, axis=1, keepdims=True)
        # print(z)
        # print(np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True), '\n')
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    @staticmethod
    def prime(z):
        return softmax.fn(z) * (1 - softmax.fn(z))


class tanh(Activation):
    @staticmethod
    def fn(z):
        return np.tanh(z)

    @staticmethod
    def prime(z):
        return 1 - np.power(np.tanh(z), 2)


class relu(Activation):
    @staticmethod
    def fn(z):
        return np.maximum(z, 0)

    @staticmethod
    def prime(z):
        return np.greater_equal(z, 0)


class leaky_relu(Activation):
    @staticmethod
    def fn(z):
        return np.maximum(z, 0) + 0.01 * np.minimum(z, 0)

    @staticmethod
    def prime(E, z, W1):
        dz = np.greater_equal(z, 0) + 0.01 * np.less(z, 0)
        error = np.dot(E, W1.T) * dz
        return error, None, W1
