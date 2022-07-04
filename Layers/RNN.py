from copy import deepcopy

import numpy as np

import Layers
from Layers import Base


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_state_list = []
        self.hidden_fc_layer_optimizer = None
        self.input_for_hidden_fc_layer = []
        self.input_for_hidden_tanh_layer = []
        self.input_for_output_sigmoid_layer = []
        self._optimizer = None
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(shape=(1, hidden_size))
        self._memorize = False
        self.hidden_fc_layer = Layers.FullyConnected.FullyConnected(self.input_size + self.hidden_size,
                                                                    self.hidden_size)
        self.hidden_tanh_layer = Layers.TanH.TanH()
        self.output_fc_layer = Layers.FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.output_sigmoid_layer = Layers.Sigmoid.Sigmoid()
        self.hidden_gradient = np.zeros(shape=(1, hidden_size))

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def forward(self, input_tensor):
        output = np.ndarray(shape=(input_tensor.shape[0], self.output_size))
        self.input_for_hidden_fc_layer = []
        self.input_for_output_fc_layer = []
        if not self.memorize:
            hidden_state = np.zeros(shape=(1, self.hidden_size))
        else:
            hidden_state = self.hidden_state
        # self.hidden_state_list.append(hidden_state)
        for t in np.arange(input_tensor.shape[0]):
            input_for_hidden_layer = (np.column_stack((input_tensor[t].reshape(1, -1), hidden_state)))
            output_of_hidden_fc_layer = self.hidden_fc_layer.forward(input_for_hidden_layer)
            self.input_for_hidden_fc_layer.append(self.hidden_fc_layer.last_input)
            self.input_for_hidden_tanh_layer.append(output_of_hidden_fc_layer)
            hidden_state = self.hidden_tanh_layer.forward(output_of_hidden_fc_layer)
            self.hidden_state_list.append(hidden_state)
            output[t] = self.output_fc_layer.forward(hidden_state)
            self.input_for_output_fc_layer.append(self.output_fc_layer.last_input)
            output[t] = self.output_sigmoid_layer.forward(output[t])
            self.input_for_output_sigmoid_layer.append(output[t])

        if self.memorize:
            self.hidden_state = hidden_state

        return output

    def backward(self, error_tensor):
        output = np.ndarray(shape=(error_tensor.shape[0], self.input_size))
        if not self.memorize:
            self.hidden_gradient = np.zeros(shape=(1, self.hidden_size))
        self.gradient_hidden_fc_layer = 0
        self.gradient_output_fc_layer = 0
        for t in list(reversed(np.arange(error_tensor.shape[0]))):
            self.output_sigmoid_layer.fx = self.input_for_output_sigmoid_layer[t]
            output_t = self.output_sigmoid_layer.backward(error_tensor[t].reshape(1, -1))

            self.output_fc_layer.last_input = self.input_for_output_fc_layer[t]
            output_t = self.output_fc_layer.backward(output_t)
            self.gradient_output_fc_layer += self.output_fc_layer.gradient_weights
            output_t = output_t + self.hidden_gradient

            self.hidden_tanh_layer.fx = self.hidden_state_list[t]
            output_t = self.hidden_tanh_layer.backward(output_t)

            self.hidden_fc_layer.last_input = self.input_for_hidden_fc_layer[t]
            tmp_out = self.hidden_fc_layer.backward(output_t).reshape(1, -1)
            self.gradient_hidden_fc_layer += self.hidden_fc_layer.gradient_weights
            output[t] = tmp_out[0, :self.input_size]
            self.hidden_gradient = tmp_out[0, self.input_size:]

        if self.optimizer is not None:
            self.output_fc_layer.weights = self.output_fc_layer_optimizer.calculate_update(self.output_fc_layer.weights,
                                                                                           self.gradient_output_fc_layer)
            self.hidden_fc_layer.weights = self.hidden_fc_layer_optimizer.calculate_update(self.hidden_fc_layer.weights,
                                                                                           self.gradient_hidden_fc_layer)

        return output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_f):
        self._optimizer = optimizer_f
        # self.hidden_fc_layer.optimizer = deepcopy(optimizer_f)
        # self.output_fc_layer.optimizer = deepcopy(optimizer_f)
        self.hidden_fc_layer_optimizer = deepcopy(optimizer_f)
        self.output_fc_layer_optimizer = deepcopy(optimizer_f)

    @property
    def weights(self):
        return self.hidden_fc_layer.weights

    @weights.setter
    def weights(self, weights):
        self.hidden_fc_layer.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_hidden_fc_layer

    def calculate_regularization_loss(self):
        regularization_loss = self.hidden_fc_layer.optimizer.regularizer.norm(self.hidden_fc_layer.weights)
        return regularization_loss

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc_layer.initialize(weights_initializer, bias_initializer)
        self.output_fc_layer.initialize(weights_initializer, bias_initializer)
