from copy import deepcopy
from math import floor, ceil

import numpy as np
from scipy import signal

from Layers import Base


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._optimizer = None
        self._bias_gradient = None
        self._weights_gradient = None
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        weights_size = (num_kernels,) + convolution_shape
        self.weights = np.random.uniform(size=weights_size)
        bias_size = (num_kernels, 1)
        self.bias = np.random.uniform(size=bias_size)
        self.y_stride = stride_shape[0]
        self.x_stride = stride_shape[0] if len(stride_shape) == 1 else stride_shape[1]

    @property
    def gradient_weights(self):
        return self._weights_gradient

    @property
    def gradient_bias(self):
        return self._bias_gradient

    def forward(self, input_tensor):
        self.last_input = input_tensor # saving input tensor to be used in backward
        # output shape will be b, h, y , channels will be collapsed
        output_shape = (input_tensor.shape[0], self.num_kernels) + (input_tensor.shape[2],)
        # if there is extra dimension output shape will be b, h, y, x so append x in the output shape
        if len(input_tensor.shape) == 4:
            output_shape = output_shape + (input_tensor.shape[3],)
        output = np.zeros(shape=output_shape)  # empty output ndarray
        for i in np.arange(input_tensor.shape[0]): # for all bactches
            for j in np.arange(self.num_kernels):  # apply all kernels
                for k in np.arange(self.convolution_shape[0]):  # for all channels
                    output[i][j] = output[i][j] + signal.convolve(input_tensor[i][k], self.weights[j][k], mode="same")
                output[i][j] = output[i][j] + self.bias[j] # add bias to the final result

        # subset using stride information
        if len(input_tensor.shape) == 4:
            output = output[..., ::self.y_stride, ::self.x_stride]
        else:
            output = output[..., ::self.x_stride]

        return output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_f):
        self._optimizer = optimizer_f
        self._optimizer_weights = deepcopy(optimizer_f)
        self._optimizer_bias = deepcopy(optimizer_f)

    def backward(self, error_tensor):
        output_shape = self.last_input.shape # the output shape of backward will be same as input shape
        # since error_tensor is strided we need to upscale it to b, h, y, x where y and x are original input shape
        error_tensor_shape_req = (error_tensor.shape[0], error_tensor.shape[1], *output_shape[2:])
        error_tensor_req = np.zeros(shape=error_tensor_shape_req)
        # upscaling the error tensor based on the required shape
        if len(output_shape) == 4:
            error_tensor_req[..., ::self.y_stride, ::self.x_stride] = error_tensor
        else:
            error_tensor_req[..., ::self.y_stride] = error_tensor
        error_tensor = error_tensor_req

        # calculate the padding required
        pad_y_r = ceil(abs(self.convolution_shape[1] - 1) / 2)
        pad_y_l = floor(abs(self.convolution_shape[1] - 1) / 2)
        if len(error_tensor.shape) == 4:
            pad_x_r = ceil(abs(self.convolution_shape[2] - 1) / 2)
            pad_x_l = floor(abs(self.convolution_shape[2] - 1) / 2)

        padding_width = np.array(((0, 0), (0, 0), (pad_y_r, pad_y_l), (pad_x_r, pad_x_l)), dtype="int") if len(
            error_tensor.shape) == 4 else np.array(((0, 0), (0, 0), (pad_y_r, pad_y_l)), dtype="int")
        xp = np.pad(self.last_input, padding_width, 'constant')
        # calculate the weights_gradient
        self._weights_gradient = np.zeros_like(self.weights)
        for n in np.arange(output_shape[0]): # for all batches
            for i in np.arange(self.weights.shape[0]): # for all kernels
                for j in np.arange(self.weights.shape[1]): # for all channels
                    self._weights_gradient[i][j] = self._weights_gradient[i][j] + signal.correlate(error_tensor[n][i],
                                                                                                  xp[n][j],
                                                                                                  mode="valid")
        # collapse error tensor along channels dimension
        self._bias_gradient = np.sum(error_tensor, axis=(0, 2, 3)).reshape(-1, 1) if len(error_tensor.shape) == 4 else \
            np.sum(error_tensor, axis=(0, 2)).reshape(-1, 1)

        output = np.zeros(shape=output_shape)
        # swap weights axis to be used in output calculation
        weights_rearranged = np.swapaxes(self.weights, 0, 1)
        for i in np.arange(output.shape[0]):
            for j in np.arange(output.shape[1]):
                for k in np.arange(self.num_kernels):
                    output[i][j] = output[i][j] + signal.correlate(error_tensor[i][k], weights_rearranged[j][k],
                                                                   mode="same")

        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._weights_gradient)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._bias_gradient)

        return output

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape) / self.convolution_shape[0] *
                                                      self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape),
                                                np.prod(self.convolution_shape) / self.convolution_shape[0] *
                                                self.num_kernels)
