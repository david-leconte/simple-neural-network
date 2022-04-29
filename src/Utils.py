import numpy as np
import matplotlib.pyplot as plt

from Losses import MSELoss


class DataHandler:
    """ This class loads data then used by the network. """

    usps_train = "./data/USPS_train.txt"
    usps_test = "./data/USPS_test.txt"

    def load_usps_train():
        return DataHandler.load_usps(DataHandler.usps_train)

    def load_usps_test():
        return DataHandler.load_usps(DataHandler.usps_test)

    def load_usps(fn):
        with open(fn, "r") as f:
            f.readline()
            data = [[float(x) for x in l.split()]
                    for l in f if len(l.split()) > 2]

        tmp = np.array(data)
        return tmp[:, 1:], tmp[:, 0].astype(int)

    def get_usps(l, datax, datay):
        assert len(datax) == len(datay)
        if type(l) != list:
            resx = datax[datay == l, :]
            resy = datay[datay == l]
            return resx, resy

        tmp = list(zip(*[DataHandler.get_usps(i, datax, datay) for i in l]))
        tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])

        permutation = np.random.permutation(len(tmpx))
        return tmpx[permutation], tmpy[permutation]

    def show_usps(data):
        plt.imshow(data.reshape((16, 16)),
                   interpolation="nearest", cmap="gray")


class Sequential:
    def __init__(self):
        self._modules = []

    def append_modules(self, *modules):
        self._modules.extend(*modules)

    def forward(self, X):
        forwards = [self._modules[0].forward(X)]

        for i in range(1, len(self._modules)):
            forwards.append(self._modules[i].forward(forwards[-1]))

        return forwards

    def backward(self, x, yhat, loss, eps=1e-3):
        forwards = self.forward(x)
        deltas = [loss.backward(yhat, forwards[-1])]

        for i in range(len(self._modules) - 1, 0, -1):
            deltas = [self._modules[i].backward_delta(
                forwards[i - 1], deltas[0])] + deltas

        deltas = [self._modules[0].backward_delta(x, deltas[0])] + deltas
        self._modules[0].backward_update_gradient(x, deltas[1])

        for i in range(1, len(self._modules)):
            self._modules[i].backward_update_gradient(
                forwards[i - 1], deltas[i + 1])

        for module in self._modules:
            module.update_parameters(eps)
            module.zero_grad()


class Optim:
    def __init__(self, net, loss=MSELoss, eps=1e-3):
        self._net = net
        self._loss = loss()
        self._eps = eps

    def step(self, batch_x, batch_y):
        self._net.backward(batch_x, batch_y, self._loss, self._eps)

    def SGD(self, datax, datay, batch_size, nb_steps):
        for i in range(nb_steps):
            indexes = np.random.choice([i for i in range(len(datax))], size = batch_size)
            batch_x = datax[indexes]
            batch_y = datay[indexes]
            self.step(batch_x, batch_y)
            
        return self._net