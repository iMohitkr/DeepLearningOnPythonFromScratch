from numpy import ndarray

from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.input_shape = None

    def forward(self, input_tensor: ndarray):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor: ndarray):
        return error_tensor.reshape(self.input_shape)
