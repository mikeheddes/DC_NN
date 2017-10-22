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

    def __init__(self, input_size):
        self.model = {"version": 0,
                      "layers": [],
                      "config": []}
        self.layers = self.model["layers"]
        self.layers += [Input(input_size)]
        self.num_layers = len(self.layers)

    def add(self, layer):
        layer.W = layer.W_init.fn(
            (layer.output_size, self.layers[-1].output_size))
        self.layers += [layer]
        self.num_layers += 1

    def forward(self, X):
        self.layers[0].A = X
        for l in range(1, self.num_layers):
            self.layers[l].fn(self.layers[l - 1].A)

    def train(self, training_data, loss=None, batch_size=1, epochs=0, test_data=None, learning_rate=None):
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dataset_size = len(training_data)

        for epoch in range(self.epochs):
            random.shuffle(training_data)
            training_batches = [training_data[k:k + self.batch_size]
                                for k in range(0, self.dataset_size, self.batch_size)]
            self.LR = self.learning_rate.fn(epoch, self.epochs)
            for batch in training_batches:
                nabla_b = [np.zeros(l.b.shape) for l in self.layers]
                nabla_w = [np.zeros(l.W.shape) for l in self.layers]
                for point in batch:
                    self.forward(point[0])
                    d_nabla_b, d_nabla_w = self.backprop(point[1])
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, d_nabla_w)]
                for l in range(1, self.num_layers):
                    self.layers[l].b -= self.LR / self.batch_size * nabla_b[l]
                    self.layers[l].W -= self.LR / self.batch_size * nabla_w[l]
            if test_data:
                self.evaluate(test_data, epoch=epoch)
            else:
                print("Epoch", epoch + 1, "complete")

    def backprop(self, Y):
        L = self.layers[-1]
        l = self.layers
        error = self.loss.prime(Y, L.A, L.Z, L.activation)
        nabla_b = [error]
        nabla_w = [np.dot(error, l[-2].A.T)]
        for i in range(self.num_layers - 2, 0, -1):
            error, change_W = l[i].prime(error, l[i + 1].W, l[i - 1].A)
            nabla_b.insert(0, error)
            nabla_w.insert(0, change_W)
        nabla_b.insert(0, 0)
        nabla_w.insert(0, 0)
        return nabla_b, nabla_w

    def evaluate(self, test_data, epoch=0):
        good = 0
        '''Can be more efficent by computing the prediction in matrix form instead of one-by-one'''
        for test in test_data:
            testPrediction = self.predict(test[0])
            good += np.equal(testPrediction, test[1])
        print("Epoch", epoch + 1, " Test accuracy", good, "/", len(test_data))

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.layers[-1].A)
