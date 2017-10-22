import numpy as np


def Dropout(X, p):
    dropout_layer = (np.random.rand(*X.shape) < p) / p
    return X * dropout_layer


def L2_norm(arg):
    L2 = (1 - eta * lmbda / dataset_size)
    return W * L2
