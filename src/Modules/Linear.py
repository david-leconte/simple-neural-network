from .Module import Module
import numpy as np


class Linear(Module):
    def __init__(self, input, output):
        super(Linear, self).__init__()

        self._input = input
        self._output = output

        self._parameters = np.random.random((input, output))
        self._gradient = np.zeros((input, output))

    def forward(self, X):
        assert X.shape[1] == self._input
        return X @ self._parameters

    def backward_update_gradient(self, input, delta):
        assert input.shape[0] == delta.shape[0]
        self._gradient += input.T @ delta

    def backward_delta(self, input, delta):
        assert delta.shape[0] == self._output
        return delta @ self._parameters.T
