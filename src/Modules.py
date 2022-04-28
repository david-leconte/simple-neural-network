import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)

    def forward(self, X):
        pass

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        self.zero_grad()

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass


class Linear(Module):
    def __init__(self, input, output):
        super(Linear, self).__init__()

        self._input = input
        self._output = output

        self._parameters = np.random.uniform(-5, 5, (input, output))
        self._gradient = np.zeros((input, output))

    def forward(self, X):
        return X @ self._parameters

    def backward_update_gradient(self, input, delta):
        self._gradient += input.T @ delta

    def backward_delta(self, input, delta):
        return delta @ self._parameters.T


class Sigmoid(Module):
    def __init__(self, l=1):
        super(Sigmoid, self).__init__()

        self._lambda = l

    def forward(self, X):
        return 1 / (1 + np.exp(self._lambda * -X))

    def backward_delta(self, input, delta):
        return delta * self._lambda * (np.exp(-self._lambda * input) / ((1 + np.exp(-self._lambda * input)) ** 2))


class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)
