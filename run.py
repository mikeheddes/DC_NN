'''
http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
https://stackoverflow.com/questions/41635737/is-this-the-correct-way-of-whitening-an-image-in-python
https://grzegorzgwardys.wordpress.com/2016/04/22/8/
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
http://cs231n.github.io/neural-networks-2/
https://github.com/llSourcell/Convolutional_neural_network/blob/master/convolutional_network_tutorial.ipynb
'''
import mnist
import numpy as np
import neural_net as nn

training_data, validation_data, test_data = mnist.load_data_wrapper()

m = nn.Model()
W = nn.variance()
m.Add(nn.Input(784))
m.Add(nn.Dense(200, W_init=W))
m.Add(nn.leaky_relu())
# m.add(nn.Dropout(rate=0.5))
m.Add(nn.Dense(80, W_init=W))
m.Add(nn.leaky_relu())
# m.add(nn.Dropout(rate=0.5))
m.Add(nn.Dense(10, W_init=W))
m.Add(nn.softmax())

m.Compile()

LR = nn.learning_rate(0.1, to=0.001, func="QUADRATIC")
m.Train(list(training_data),
        loss=nn.cross_entropy,
        batch_size=10,
        epochs=50,
        learning_rate=LR,
        optimizer=nn.L2(),
        test_data=list(test_data))
