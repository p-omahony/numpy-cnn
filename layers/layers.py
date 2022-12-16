import numpy as np
from scipy import signal


class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class ConvLayer:
    def __init__(self, num_kernels, kernel_size) -> None:
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.kernel_shape = (num_kernels, *kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        

    def forward_propagation(self, input):
        output = []
        self.input = input
        self.input_shape = input.shape
        for j in range(self.num_kernels):
            kernel = self.kernels[j]
            correlated = signal.correlate2d(np.squeeze(input), kernel, mode='valid')
            output.append(correlated)
        return np.array(output)

    def backward_propagation(self, output_error, learning_rate):
        kernels_gradient = np.zeros((self.num_kernels, *self.kernel_size))
        input_gradient = np.squeeze(np.zeros(self.input_shape))

        inputs = np.squeeze(self.input)
        for i in range(self.num_kernels):
            kernels_gradient[i] = signal.correlate2d(inputs, output_error[i], mode='valid')
            input_gradient += signal.convolve2d(output_error[i], self.kernels[i], "full")


        self.kernels -= learning_rate * kernels_gradient
        #self.biases -= learning_rate * output_error
        return input_gradient
        

class ReshapeLayer:
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagation(self, input):
        return np.reshape(input, self.output_shape)

    def backward_propagation(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)