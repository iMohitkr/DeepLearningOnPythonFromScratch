import numpy as np

from Layers import Base


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.fx = None

    def forward(self, input_tensor):
        self.fx = 1 / (1 + np.exp(-1 * input_tensor))
        return self.fx

    def backward(self, error_tensor):
        return error_tensor * self.fx * (1 - self.fx)
