import numpy as np

from Layers import Base
from Optimization import Optimizers


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self._weights_gradient = None
        self.last_input = None
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self._optimizer: Optimizers = None

    def forward(self, input_tensor):
        input_tensor = np.column_stack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        self.last_input = input_tensor
        return np.dot(input_tensor, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_f):
        self._optimizer = optimizer_f

    @property
    def gradient_weights(self):
        return self._weights_gradient

    def backward(self, error_tensor):
        weights_without_bias = np.delete(self.weights, -1, axis=0)
        error_tensor_next = np.dot(error_tensor, np.transpose(weights_without_bias))
        self._weights_gradient = np.dot(np.transpose(self.last_input), error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._weights_gradient)
        return error_tensor_next

    def initialize(self, weights_initializer, bias_initializer):
        tmp_weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size,
                                                       self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.row_stack((tmp_weights, bias))
