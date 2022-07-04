import numpy as np
from copy import deepcopy
from Layers import Base
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.mean_input = None
        self.variance_input = None
        self.input_shape = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._bias_gradient = None
        self.input_tensor = None
        self.last_input = None
        self._optimizer = None
        self._weights_gradient = None
        self.moving_average_mean = 0
        self.moving_average_variance = 0
        self.trainable = True
        self.channels = channels
        self.weights = np.ones(shape=(1, self.channels))
        self.bias = np.zeros(shape=(1, self.channels))
        self.decay = 0
        self.initialize(None, None)

    def initialize(self, _, __):
        self.weights = np.ones_like(self.weights)
        self.bias = np.zeros_like(self.bias)

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 4:
            axis = (0, 2, 3)
        elif len(input_tensor.shape) == 3:
            axis = (0, 2)
        else:
            axis = 0

        if self.testing_phase:
            self.mean_input = self.moving_average_mean
            self.variance_input = self.moving_average_variance
        else:
            self.mean_input = np.mean(input_tensor, axis=axis)
            self.variance_input = np.var(input_tensor, axis=axis)

            self.moving_average_mean = self.decay * self.moving_average_mean + (1 - self.decay) * self.mean_input
            self.moving_average_variance = self.decay * self.moving_average_variance + (
                    1 - self.decay) * self.variance_input
            self.decay = 0.8

        scaled_input = (np.swapaxes(input_tensor, 1, len(input_tensor.shape) - 1) - self.mean_input) / np.sqrt(
            self.variance_input +
            1e-12)

        output = self.weights * scaled_input + self.bias
        output = np.swapaxes(output, 1, len(input_tensor.shape) - 1)
        self.last_input = np.swapaxes(scaled_input, 1, len(input_tensor.shape) - 1)
        return output

    def reformat(self, image_tensor):
        if len(image_tensor.shape) > 2:
            image_tensor2 = image_tensor.reshape(image_tensor.shape[0], self.channels, np.prod(image_tensor.shape[2:]))
            image_tensor3 = np.swapaxes(image_tensor2, 1, image_tensor2.ndim - 1)
            image_tensor4 = image_tensor3.reshape(-1, self.channels)
            return image_tensor4
        else:
            image_tensor2 = image_tensor.reshape(self.input_shape[0], -1, self.channels)
            image_tensor3 = np.swapaxes(image_tensor2, 1, image_tensor2.ndim - 1)
            image_tensor4 = image_tensor3.reshape(self.input_shape[0], self.channels, *self.input_shape[2:])
            return image_tensor4

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_f):
        self._optimizer = optimizer_f
        self._optimizer_weights = deepcopy(optimizer_f)
        self._optimizer_bias = deepcopy(optimizer_f)

    @property
    def gradient_weights(self):
        return self._weights_gradient

    @property
    def gradient_bias(self):
        return self._bias_gradient

    def backward(self, error_tensor):
        output = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor), self.weights,
                                      self.mean_input,
                                      self.variance_input)

        if len(error_tensor.shape) == 4:
            self._weights_gradient = np.sum(self.last_input * error_tensor,
                                            axis=(0, 2, 3)).reshape(1, -1)
        elif len(error_tensor.shape) == 3:
            self._weights_gradient = np.sum(self.last_input * error_tensor,
                                            axis=(0, 2)).reshape(1,
                                                                 -1)
        else:
            self._weights_gradient = np.sum(self.last_input * error_tensor,
                                            axis=0).reshape(1, -1)

        self._bias_gradient = np.sum(error_tensor, axis=(0, 2, 3)).reshape(1, -1) if len(error_tensor.shape) == 4 else \
            np.sum(error_tensor, axis=(0, 2)).reshape(1, -1) if len(error_tensor.shape) == 3 else \
            np.sum(error_tensor, axis=0).reshape(1, -1)
        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._weights_gradient)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._bias_gradient)

        return self.reformat(output)
