from copy import deepcopy

import Layers
from Optimization import Optimizers
from Layers import Initializers


class NeuralNetwork:
    def __init__(self, optmizer: Optimizers, weights_initializer: Initializers, bias_initializer: Initializers):
        self.label_tensor = None
        self.optimizer = optmizer
        self.weights_initilizer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss: list = []
        self.layers: list = []
        self.data_layer = None
        self.loss_layer = None

    def phase(self, testing_phase):
        for layer in self.layers:
            layer.testing_phase = testing_phase

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        regularization_loss = 0
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer is not None and layer.optimizer.regularizer is not None:
                regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
            input_tensor = output_tensor
        return self.loss_layer.forward(output_tensor, label_tensor) + regularization_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer: Layers):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initilizer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase(False)
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase(True)
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
        return output_tensor
