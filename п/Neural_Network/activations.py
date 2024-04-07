from activation import Activation
import numpy as np

class Tahn(Activation): 
    def __init__(self):
        tahn = lambda x: np.tanh(x) ** 2
        tahn_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tahn, tahn_prime)
        