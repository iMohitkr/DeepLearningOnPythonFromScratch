import numpy as np

from Layers import Base


class ReLU(Base.BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.last_input = None

    def forward(self, input_tensor):
        self.last_input = input_tensor
        return input_tensor * (input_tensor > 0)

    def backward(self, error_tensor):
        return error_tensor * (self.last_input > 0)
