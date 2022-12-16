import numpy as np

def relu(Z):
    return np.maximum(Z,0)

def relu_derivative(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    return(np.exp(Z)/np.exp(Z).sum())


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2