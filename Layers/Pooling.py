import numpy as np

from Layers import Base


def get_strided_representation(arr, sub_shape, stride):
    try:
        n, c, y, x = arr.shape
        s_n, s_c, s_y, s_x = arr.strides
        y_p, x_p = sub_shape
        stride_y, stride_x = stride
        view_shape = (n, c, 1 + (y - y_p) // stride_y, 1 + (x - x_p) // stride_x, y_p, x_p)
        strides = (s_n, s_c, stride_y * s_y, stride_x * s_x, s_y, s_x)
    except ValueError:
        n, c, y = arr.shape
        s_n, s_c, s_y = arr.strides
        y_p = sub_shape[0]
        stride_y = stride[0]
        view_shape = (n, c, 1 + (y - y_p) // stride_y, y_p)
        strides = (s_n, s_c, stride_y * s_y, s_y)

    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super(Pooling, self).__init__()
        self.input_shape = None
        self.maxima_locations = None
        self.stride_shape = stride_shape
        self.stride_y = self.stride_shape[0]
        if len(self.stride_shape) == 2:
            self.stride_x = self.stride_shape[1]
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        view = get_strided_representation(input_tensor, self.pooling_shape, self.stride_shape)
        axis = (4, 5) if len(input_tensor.shape) == 4 else 4
        result = np.nanmax(view, axis=axis, keepdims=True)
        self.maxima_locations = np.where(result == view, 1, 0)
        return np.squeeze(result, axis=axis)

    def backward(self, error_tensor):
        result = np.zeros(shape=self.input_shape)
        n, c, y, x, p_y, p_x = np.where(self.maxima_locations == 1)
        values = error_tensor[n, c, y, x].flatten()
        for i_n, i_c, i_y, i_x, value in  zip(n, c, y*self.stride_y + p_y, x*self.stride_x + p_x, values):
            result[i_n, i_c, i_y, i_x] = result[i_n, i_c, i_y, i_x] + value
        return result
