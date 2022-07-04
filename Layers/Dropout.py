import numpy as np

from Layers import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probablity):
        super(Dropout, self).__init__()
        self.dropout_matrix = None
        self.probablity = probablity

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.dropout_matrix = np.random.rand(*input_tensor.shape) < self.probablity
            input_tensor = input_tensor * self.dropout_matrix
            return input_tensor / self.probablity

    def backward(self, error_tensor):
        return error_tensor*self.dropout_matrix/self.probablity
