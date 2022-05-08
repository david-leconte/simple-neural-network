import numpy as np


class Loss:
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return 2 * (yhat - y)


class BCELoss(Loss):
    def forward(self, y, yhat):
        return -(y * np.maximum(-100, np.log(yhat + 1e-10)) + (1 - y) * np.maximum(-100, np.log(1 - yhat + 1e-10)))

    def backward(self, y, yhat):
        return -((y / (yhat + 1e-10)) - ((1 - y) / (1 - yhat + 1e-10)))
