import numpy as np


class Module:
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        if(self._gradient is not None):
            self._gradient = np.zeros(self._gradient.shape)

    def forward(self, X):
        pass

    def update_parameters(self, gradient_step=1e-3):
        if(self._parameters is not None and self._gradient is not None):
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

        self._parameters = np.random.uniform(-1, 1, (input, output))
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
        forward = 1 / (1 + np.exp(self._lambda * -X))

        if(np.isnan(forward).any()):
            forward = np.exp(self._lambda * X) / (1 + np.exp(self._lambda * X))

        return forward

    def backward_delta(self, input, delta):
        new_delta = delta * \
            self._lambda * (np.exp(-self._lambda * input) /
                            ((1 + np.exp(-self._lambda * input)) ** 2))

        if(np.isnan(new_delta).any()):
            new_delta = delta * \
                self._lambda * (np.exp(self._lambda * input) /
                                ((1 + np.exp(self._lambda * input)) ** 2))

        nan_values = np.argwhere(np.isnan(new_delta))
        assert nan_values.size == 0

        return new_delta


class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def backward_delta(self, input, delta):
        return delta * np.where(input > 0, 1, 0)


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        return np.reshape(X, X.shape[0:-2] + (X.shape[-2] * X.shape[-1],))

    def backward_delta(self, input, delta):
        a = np.reshape(delta, input.shape)
        return np.reshape(delta, input.shape)


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1):
        super(Conv1D, self).__init__()

        self._k_size = k_size
        self._stride = stride

        self._chan_in = chan_in
        self._chan_out = chan_out

        self._parameters = np.random.uniform(-5,
                                             5, (k_size, chan_in, chan_out))
        self._gradient = np.zeros(self._parameters.shape)

    def forward(self, X):
        if self._chan_in == 1:
            X = X[:, :, np.newaxis]

        windows = np.lib.stride_tricks.sliding_window_view(
            X, self._k_size, axis=1)
        strided = windows[:, ::self._stride, :, :]

        return np.tensordot(strided, self._parameters, axes=([3, 2], [0, 1]))

    def backward_update_gradient(self, input, delta):
        if self._chan_in == 1:
            input = input[:, :, np.newaxis]

        for c_out in range(self._chan_out):
            for c_in in range(input.shape[-1]):
                for x in range(input.shape[0]):
                    for i in range(0, input.shape[1] - self._k_size, self._stride):
                        index_in_output = (
                            (i - self._k_size) // self._stride + 1) + np.arange(self._k_size)
                        window = input[x, i:i+self._k_size, c_in]

                        self._gradient[:, c_in, c_out] += delta[x,
                                                                index_in_output, c_out] * window

    def backward_delta(self, input, delta):
        if self._chan_in == 1:
            input = input[:, :, np.newaxis]

        new_delta = np.zeros(input.shape)

        for c_out in range(self._chan_out):
            for c_in in range(input.shape[-1]):
                for x in range(input.shape[0]):
                    for i in range(0, input.shape[1] - self._k_size, self._stride):
                        index_in_input = i + np.arange(self._k_size)
                        index_in_output = (
                            (i - self._k_size) // self._stride + 1) + np.arange(self._k_size)

                        new_delta[x, index_in_input, c_in] += np.sum(
                            delta[x, index_in_output, c_out] * self._parameters[:, c_in, c_out])

        if self._chan_in == 1:
            new_delta = np.reshape(new_delta, new_delta.shape[:-1])


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super(MaxPool1D, self).__init__()

        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]

        windows = np.lib.stride_tricks.sliding_window_view(
            X, self._k_size, axis=1)
        strided = windows[:, ::self._stride, :, :]

        return np.max(strided, axis=3)

    def backward_delta(self, input, delta):
        if len(input.shape) == 2:
            input = input[:, :, np.newaxis]

        new_delta = np.zeros(input.shape)

        for c in range(input.shape[-1]):
            for x in range(input.shape[0]):
                for i in range(0, input.shape[1] - self._k_size, self._stride):
                    window = input[x, i:i+self._k_size, c]
                    index_max_in_window = np.argmax(window)

                    index_max_in_input = i + index_max_in_window
                    index_max_in_output = (
                        i - self._k_size) // self._stride + 1 + index_max_in_window

                    new_delta[x, index_max_in_input,
                              c] += delta[x, index_max_in_output, c]

        if len(input.shape) == 2:
            new_delta = np.reshape(new_delta, new_delta.shape[:-1])

        return new_delta
