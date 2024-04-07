from dense import Dense
from activations import Tahn
from losses import mse, mse_prime
import numpy as np 

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tahn(),
    Dense(3, 1),
    Tahn()
]

epochs = 40000
learning_rate = 0.001

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error += mse(y, output)

        #backward
        grad = mse_prime(y, output)
        
        #print(grad)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(x)
    print('%d/%d, error=%f' % (e +1, epochs, error))

