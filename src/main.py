from matplotlib import pyplot as plt
import numpy as np

import DataHandler
from Losses import MSELoss
from Modules import Linear, Sigmoid

# Testing linear regression (part I)
d = 3
mse = MSELoss()
linear = Linear(d, 1)

size = 1000
X = np.random.randint(-3, 3, size=(size, d))
W = np.array([4, 0.22, -3])
Y = np.expand_dims(X @ W, axis=1)

Yhat = linear.forward(X)
original_loss = np.mean(mse.forward(Y, Yhat))

for i in range(len(X)):
    x = X[i:i+1]
    y = Y[i]
    yhat = linear.forward(x)

    delta = mse.backward(y, yhat)

    linear.backward_update_gradient(x, delta)
    linear.update_parameters()
    linear.zero_grad()

Yhat = linear.forward(X)
updated_loss = np.mean(mse.forward(Y, Yhat))

print("Linear regression original and updated loss + parameters :", original_loss, updated_loss, linear._parameters)

# Testing perceptron
d = 2
mse = MSELoss()
linear = Linear(d, 1)
sigmoid = Sigmoid()

size = 1000

X = np.zeros((size, d))
X[:size // 2] = np.random.normal((-2, -2), 1, size=(size // 2, 2))
X[size // 2:] = np.random.normal((2, 2), 1, size=(size // 2, 2))

permuted_indexes = np.random.permutation(size)
X = X[permuted_indexes]

Y = np.concatenate((np.zeros(size // 2), np.ones(size // 2)))[permuted_indexes]
Y = np.expand_dims(Y, axis=1)

Yhat = sigmoid.forward(linear.forward(X))
original_loss = np.mean(mse.forward(Y, Yhat))

for i in range(len(X)):
    x = X[i:i+1]
    y = Y[i]
    yhat = sigmoid.forward(linear.forward(x))

    delta_loss = mse.backward(y, yhat)

    delta_sigmoid = sigmoid.backward_delta(linear.forward(x), delta_loss)

    linear.backward_update_gradient(x, delta_sigmoid)
    linear.update_parameters()
    linear.zero_grad()

Yhat = sigmoid.forward(linear.forward(X))
updated_loss = np.mean(mse.forward(Y, np.rint(Yhat)))

print("Perceptron original and updated loss + parameters :", original_loss, updated_loss, linear._parameters)

a, b = linear._parameters[:, 0]
V = np.linspace(-3, 3, 1000)

plt.plot(V, (-a / b) * V)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Perceptron approximation")
plt.show()