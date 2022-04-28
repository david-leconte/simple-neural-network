from matplotlib import pyplot as plt
import numpy as np

import DataHandler
from Losses.MSELoss import MSELoss
from Modules.Linear import Linear

# Testing linear regression (part I)
d = 3
mse = MSELoss()
linear = Linear(d, 1)

size = 1000
X = np.random.randint(-3, 3, size=(size, d))
W = np.array([4, 0.22, -3])
Y = X @ W

Yhat = linear.forward(X)
original_loss = np.mean(mse.forward(Y, Yhat))

for i in range(len(X)):
    x = X[i:i+1]
    y = Y[i:i+1]
    yhat = linear.forward(x)

    delta = mse.backward(y, yhat)

    linear.backward_update_gradient(x, delta)
    linear.update_parameters()
    linear.zero_grad()

Yhat = linear.forward(X)
updated_loss = np.mean(mse.forward(Y, Yhat))

print(original_loss, updated_loss, linear._parameters)
