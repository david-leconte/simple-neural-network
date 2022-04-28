from .Loss import Loss
import numpy as np


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return 2 * (yhat - y)
