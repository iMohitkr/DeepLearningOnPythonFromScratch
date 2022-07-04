import numpy as np

from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.y_hat = None
        self.last_input = None

    def forward(self, input_tensor):
        self.last_input = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)
        self.y_hat = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1).reshape((input_tensor.shape[0], 1))
        return self.y_hat

    def backward(self, error_tensor):
        error_tensor_next = self.y_hat * (error_tensor - np.sum(error_tensor * self.y_hat,
                                                                axis=1).reshape((error_tensor.shape[0], 1)))
        return error_tensor_next
