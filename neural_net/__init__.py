from .layers import *
from .activations import *
from .losses import *
from .optimizers import *
from .regularizers import *

import random
import os
import json
import time


class Model():

    def __init__(self, layers=None, name=None):
        self.layers = []  # Stack of layers.
        self.model = None  # Internal Model instance.
        self.inputs = []  # List of input tensors
        self.outputs = []  # List of length 1: the output tensor (unique).
        self._trainable = True
        self._initial_weights = None

        # Model attributes.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False

        # Set model name.
        if not name:
            name = "Model"
        self.name = name

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.
        # Arguments
            layer: layer instance.
        # Raises
            ValueError: In case the `layer` argument does not
                know its input shape.
        """
        # Check if there are known outputs and if added layer is Input layer
        if not self.outputs and not isinstance(layer, Input):
            # Create an Input layer
            if not hasattr(layer, 'input_shape'):
                raise ValueError('The first layer in a '
                                 'Sequential model must '
                                 'get an `input_shape` argument.')
            # Instantiate the input layer.
            input_layer = Input(layer.input_shape)
            self.layers.append(input_layer)

        if not layer.input_shape:
            layer.input_shape = self.outputs[-1]
        if not layer.output_shape:
            layer.output_shape = self.outputs[-1]
        self.inputs.append(layer.input_shape)
        self.outputs.append(layer.output_shape)
        self.layers.append(layer)
        if hasattr(layer, 'activation'):
            if layer.activation:
                self.add(layer.activation())
        self.built = False

    def compile(self):
        for layer in self.layers:
            if hasattr(layer, 'compile'):
                layer.compile()

    def forward(self, X):
        self.calc = [X]
        for l in range(1, len(self.layers)):
            self.calc.append(self.layers[l].fn(self.calc[-1]))

    def train(self, training_data, loss=None, batch_size=1, epochs=0, test_data=None, learning_rate=None):
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dataset_size = len(training_data)

        for epoch in range(self.epochs):
            np.random.shuffle(training_data)
            training_batches = [list(zip(*training_data[k:k + self.batch_size]))
                                for k in range(0, self.dataset_size, self.batch_size)]
            self.LR = self.learning_rate.fn(epoch, self.epochs)
            for batch in training_batches:
                X = np.array(batch[0])
                Y = np.array(batch[1])
                self.forward(X)
                d_b, d_W = self.backprop(Y)
                for l in range(1, len(self.layers)):
                    if d_W[l] is not None:
                        self.layers[l].b -= self.LR / \
                            self.batch_size * np.sum(d_b[l], axis=0)
                        self.layers[l].W -= self.LR / self.batch_size * d_W[l]
            if test_data:
                self.evaluate(test_data, epoch=epoch)
            else:
                print("Epoch", epoch + 1, "complete")

    def backprop(self, Y):
        l = self.layers
        E = [self.loss.prime(Y, self.calc[-1], self.calc[-2], self.layers[-1])]
        W1 = 0
        delta_W = [None]
        for i in range(len(self.layers) - 2, 0, -1):
            err, dW, W = l[i].prime(E[-1], self.calc[i - 1], W1)
            E.append(err)
            delta_W.append(dW)
            W1 = W
        E.append(None)
        E.reverse()
        delta_W.append(None)
        delta_W.reverse()
        return E, delta_W

    def evaluate(self, test_data, epoch=0):
        data = list(zip(*test_data))
        X = np.array(data[0])
        Y = np.array(data[1])
        testPrediction = self.predict(X)
        good = np.sum(np.equal(testPrediction, Y))
        print("Epoch", epoch + 1, " Test accuracy", good, "/", len(test_data))

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.calc[-1], axis=1)
