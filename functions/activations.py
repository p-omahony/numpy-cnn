import numpy as np

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)

def softmax_prime(x):
    soft = softmax(x)                                
    diag_soft = soft*(1- soft)
    return diag_soft  

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2