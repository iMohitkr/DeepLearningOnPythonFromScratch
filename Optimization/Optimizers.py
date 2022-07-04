from __future__ import division

import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super(Sgd, self).__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor - \
                        (
                            0 if self.regularizer is None else self.learning_rate * self.regularizer.calculate_gradient(
                                weight_tensor))
        return weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super(SgdWithMomentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v
        return weight_tensor + v - \
               (
                   0 if self.regularizer is None else self.learning_rate * self.regularizer.calculate_gradient(
                       weight_tensor))


class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.v = v
        r = self.rho * self.r + (1 - self.rho) * (gradient_tensor * gradient_tensor)
        self.r = r
        v_hat = v / (1 - np.power(self.mu, self.k))
        r_hat = r / (1 - np.power(self.rho, self.k))
        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + 1e-10)) - \
                        (
                            0 if self.regularizer is None else self.learning_rate * self.regularizer.calculate_gradient(
                                weight_tensor))
        self.k = self.k + 1
        return weight_tensor
