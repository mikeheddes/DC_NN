from .layers import *
from .activations import *
from .losses import *
from .optimizers import *

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

    def Add(self, layer):
        """Adds a layer instance on top of the layer stack.
        # Arguments
            layer: layer instance.
        # Raises
            ValueError: In case the `layer` argument does not
                know its input shape.
        """
        # Check if layer is layer
        if not isinstance(layer, Layer):
            raise ValueError('%s is not a layer' % (layer.__class__.__name__))
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

    def Compile(self, optimizer=L1L2()):
        for layer in self.layers:
            if hasattr(layer, 'compile'):
                layer.compile()
            if hasattr(layer, 'optimizer'):
                if not layer.optimizer:
                    layer.optimizer = optimizer

    def forward(self, X, flag):
        self.calc = [X]
        addCalc = self.calc.append
        for l in range(1, len(self.layers)):
            addCalc(self.layers[l].fn(self.calc[-1], flag=flag))

    def Train(self, training_data, loss=None, batch_size=1, epochs=0, test_data=None, learning_rate=None):
        self.loss = loss
        dataset_size = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            training_batches = [list(zip(*training_data[k:k + batch_size])) for k in range(0, dataset_size, batch_size)]
            self.LR = learning_rate.fn(epoch, epochs) / batch_size
            for batch in training_batches:
                X = np.array(batch[0])
                Y = np.array(batch[1])
                self.forward(X, 'TRAIN')
                self.backprop(Y)
            if test_data:
                self.evaluate(test_data, epoch=epoch)
            else:
                print("Epoch %s complete" % (epoch + 1))

    def backprop(self, Y):
        lay = self.layers
        E = self.loss.prime(Y, self.calc[-1], self.calc[-2], lay[-1])
        for i in range(len(lay) - 2, 0, -1):
            E = lay[i].prime(E, self.calc[i - 1], learning_rate=self.LR)

    def evaluate(self, test_data, epoch=0):
        data = list(zip(*test_data))
        X = np.array(data[0])
        Y = np.array(data[1])
        testPrediction = self.predict(X)
        good = np.sum(np.equal(testPrediction, Y))
        print("Epoch %s: Test accuracy %s / %s" % (epoch + 1, good, len(test_data)))

    def predict(self, X):
        self.forward(X, 'PREDICT')
        return np.argmax(self.calc[-1], axis=1)
