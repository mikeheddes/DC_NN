import numpy as np
import mnist
import neural_net as nn


training_data, validation_data, test_data = mnist.load_data_wrapper()

m = nn.Model(784)
m.add(nn.Dense(200, activation=nn.leaky_relu, weight_init=nn.normal(std=0.1)))
m.add(nn.Dense(80, activation=nn.leaky_relu, weight_init=nn.normal(std=0.1)))
m.add(nn.Dense(10, activation=nn.softmax, weight_init=nn.normal(std=0.1)))
m.train(list(training_data),
        loss=nn.cross_entropy,
        batch_size=10,
        epochs=3,
        learning_rate=nn.learning_rate(0.1, to=0.001, func="QUADRATIC"),
        test_data=list(test_data))
