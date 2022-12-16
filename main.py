import numpy as np

from models import Network
from layers import FCLayer, ActivationLayer, ConvLayer, ReshapeLayer
from functions.activations import tanh, tanh_prime
from functions.losses import mse, mse_prime

def main():
    data = np.load('./data/mnist.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']


    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np.eye(10)[y_train]

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np.eye(10)[y_test]

    net = Network()
    net.add(ConvLayer(5,(3,3)))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(ReshapeLayer(input_shape=(5,26,26), output_shape=(1,5*26*26)))
    net.add(FCLayer(5*26*26, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 10))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.01)

    out = net.predict(x_test[0:3])
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(np.argmax(y_test[0:3]))

if __name__ == '__main__':
    main()