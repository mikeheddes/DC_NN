import numpy as np
import mnist
import neural_net as nn
import time


training_data, validation_data, test_data = mnist.load_data_wrapper()

m = nn.Model()
m.add(nn.Input(784))
m.add(nn.Dense(200, activation=nn.leaky_relu, W_init=nn.normal(std=0.1)))
m.add(nn.Dense(80, W_init=nn.normal(std=0.1)))
m.add(nn.leaky_relu())
m.add(nn.Dense(10, activation=nn.softmax, W_init=nn.normal(std=0.1)))
m.compile()
LR = nn.learning_rate(0.1, to=0.001, func="QUADRATIC")
m.train(list(training_data),
        loss=nn.cross_entropy,
        batch_size=10,
        epochs=50,
        learning_rate=LR,
        test_data=list(test_data))
