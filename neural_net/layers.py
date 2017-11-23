import numpy as np

from .optimizers import *


class Layer(object):
    """docstring for Layer."""

    def __init__(self):
        pass


class Conv2D(Layer):
    """docstring for Conv2D."""

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 f_init=normal(),
                 b_init=zeros(),
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 input_shape=None,
                 **kwargs):
        self.fshape = fshape
        self.strides = strides
        self.filters = filter_init(self.fshape)
        self.activation = activation

        self.output_shape = None
        self.input_shape = input_shape
        self.size = [*kernel_size, filters]
        self.activation = activation
        self.optimizer = optimizer
        self.f_init = W_init
        self.b_init = b_init

    def compile(self):
        if not self.input_shape:
            raise ValueError('No input shape specified')
        self.output_shape =
        self.fmap = np.zeros((self.input_shape[0], s, s, self.fshape[-1]))
        s = (x.shape[1] - self.fshape[0]) / self.strides + 1
        self.b = self.b_init.fn((1, self.output_shape))

    def fn(self, x, **kwarg):
        s = (x.shape[1] - self.fshape[0]) / self.strides + 1
        fmap = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = np.sum(x[:, j * self.strides:j * self.strides + self.fshape[0], i *
                                            self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return fmap

    def prime(self, error, activation_1, learning_rate):
        _E = np.dot(error, self.W.T)
        self.b -= learning_rate * np.sum(error, axis=0)
        deltaW = np.dot(activation_1.T, error)
        self.W -= learning_rate * (deltaW + self.optimizer.prime(self.W))
        return _E

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err * self.activation.deriv(z)

    def backward(self, layer_err):
        bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2]))
        s = (backwarded_fmap.shape[1] - self.fshape[0]) / self.strides + 1
        for j in xrange(s):
            for i in xrange(s):
                backwarded_fmap[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides +
                                self.fshape[1]] += np.sum(self.filters[np.newaxis, ...] * layer_err[:, j:j + 1, i:i + 1, np.newaxis, :], axis=4)
        return backwarded_fmap

    def get_grad(self, inputs, layer_err):
        total_layer_err = np.sum(layer_err, axis=(0, 1, 2))
        filters_err = np.zeros(self.fshape)
        s = (inputs.shape[1] - self.fshape[0]) / self.strides + 1
        summed_inputs = np.sum(inputs, axis=0)
        for j in xrange(s):
            for i in xrange(s):
                filters_err += summed_inputs[j * self.strides:j * self.strides + self.fshape[0],
                                             i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_err * total_layer_err

    def update(self, grad):
        self.filters -= grad

    def get_config(self):
        return{}


class Dense(Layer):
    """Most basic layer"""

    def __init__(self, output_shape,
                 activation=None,
                 optimizer=None,
                 W_init=normal(),
                 b_init=zeros(),
                 input_shape=None):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.activation = activation
        self.optimizer = optimizer
        self.W_init = W_init
        self.b_init = b_init

    def compile(self):
        if not self.input_shape:
            raise ValueError('No input shape specified')
        print(self.input_shape, self.output_shape)
        self.W = self.W_init.fn((self.input_shape, self.output_shape))
        self.b = self.b_init.fn((1, self.output_shape))

    def fn(self, x, **kwarg):
        return np.dot(x, self.W) + self.b

    def prime(self, error, activation_1, learning_rate):
        _E = np.dot(error, self.W.T)
        self.b -= learning_rate * np.sum(error, axis=0)
        deltaW = np.dot(activation_1.T, error)
        self.W -= learning_rate * (deltaW + self.optimizer.prime(self.W))
        return _E

    def get_config(self):
        return{}


class Dropout(Layer):
    """docstring for Dropout."""

    def __init__(self, pKeep=0.5):
        self.output_shape = None
        self.input_shape = None
        self.rate = min(1., max(0., pKeep))
        self.options = {'TRAIN': self.train_fn,
                        'PREDICT': self.predict_fn}

    def fn(self, x, flag):
        return self.options[flag](x)

    def train_fn(self, x):
        self.mask = np.less(np.random.rand(*x.shape), self.rate) / self.rate
        return x * self.mask

    def predict_fn(self, x):
        return x

    def prime(self, error, activation_1, **kwarg):
        return error * self.mask

    def get_config(self):
        return{}


class Input(Layer):

    def __init__(self, shape):
        self.output_shape = shape
        self.input_shape = shape
