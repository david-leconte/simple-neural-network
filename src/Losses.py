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


class CELoss(Loss):
    def softmax(yhat):
        yhat_norm = yhat - np.max(yhat, axis=1, keepdims=True)
        return np.exp(yhat_norm) / np.reshape(np.repeat(np.sum(np.exp(yhat_norm), axis=1), yhat.shape[1]), yhat.shape)

    def forward(self, y, yhat):
        softmax = CELoss.softmax(yhat)
        return - np.log(softmax[np.where(y == 1)] + 1e-20)

    def backward(self, y, yhat):
        delta_softmax = CELoss.softmax(yhat)
        delta_softmax[np.where(y == 1)] -= 1

        return delta_softmax

class BCELoss(Loss):
    def forward(self, y, yhat):
        return -(y * np.maximum(-100, np.log(yhat + 1e-20)) + (1 - y) * np.maximum(-100, np.log(1 - yhat + 1e-20)))

    def backward(self, y, yhat):
        return -((y / (yhat + 1e-20)) - ((1 - y) / (1 - yhat + 1e-20)))
