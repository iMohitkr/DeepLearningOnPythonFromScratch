from __future__ import division
import numpy as np

class Constant:
    def __init__(self, init_val=0.1):
        self._init_val = init_val

    def initialize(self, weights_shape, fan_in, fan_out):
        arr = np.ndarray(shape=weights_shape)
        arr[:] = self._init_val
        return arr


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(scale=sigma, size=weights_shape)


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(scale=sigma, size=weights_shape)
