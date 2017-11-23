import mnist
import numpy as np
import neural_net as nn

training_data, validation_data, test_data = mnist.load_data_wrapper()

m = nn.Model()
W = nn.variance()
m.Add(nn.Input(784))
m.Add(nn.Dense(200, W_init=W))
m.Add(nn.leaky_relu())
m.Add(nn.Dropout(pKeep=0.75))
m.Add(nn.Dense(80, W_init=W))
m.Add(nn.leaky_relu())
m.Add(nn.Dropout(pKeep=0.75))
m.Add(nn.Dense(10, W_init=W))
m.Add(nn.softmax())

m.Compile(optimizer=nn.L2())

LR = nn.learning_rate(0.1, to=0.001, func="STEP")
m.Train(list(training_data),
        loss=nn.cross_entropy,
        batch_size=10,
        epochs=50,
        learning_rate=LR,
        test_data=list(test_data))
