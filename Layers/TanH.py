import numpy as np

from Layers import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        super(TanH, self).__init__()
        self.fx = None

    def forward(self, input_tensor):
        self.fx = np.tanh(input_tensor)
        return self.fx

    def backward(self, error_tensor):
        return error_tensor * (1 - np.power(self.fx,2))
