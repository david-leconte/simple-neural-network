import warnings

import time
from matplotlib import pyplot as plt
import numpy as np

from Utils import DataHandler, Sequential, Optim
from Losses import MSELoss, CELoss, BCELoss
from Modules import Linear, Sigmoid, TanH, Conv1D, MaxPool1D, ReLU, Flatten

warnings.filterwarnings("ignore", category=RuntimeWarning) 

# # Testing linear regression (part I)
# d = 3
# mse = MSELoss()
# linear = Linear(d, 1)

# size = 1000
# X = np.random.randint(-3, 3, size=(size, d))
# W = np.array([4, 0.22, -3])
# Y = np.expand_dims(X @ W, axis=1)

# Yhat = linear.forward(X)
# original_loss = np.mean(mse.forward(Y, Yhat))

# for i in range(len(X)):
#     x = X[i:i+1]
#     y = Y[i]
#     yhat = linear.forward(x)

#     delta = mse.backward(y, yhat)

#     linear.backward_update_gradient(x, delta)
#     linear.update_parameters()
#     linear.zero_grad()

# Yhat = linear.forward(X)
# updated_loss = np.mean(mse.forward(Y, Yhat))

# print("Linear regression original and updated loss + parameters :", original_loss, updated_loss, linear._parameters)

# # Testing perceptron
# d = 2
# mse = MSELoss()
# linear = Linear(d, 1)
# sigmoid = Sigmoid()

# size = 5000

# X = np.zeros((size, d))
# X[:size // 2] = np.random.normal((-2, -2), 1, size=(size // 2, 2))
# X[size // 2:] = np.random.normal((2, 2), 1, size=(size // 2, 2))

# permuted_indexes = np.random.permutation(size)
# X = X[permuted_indexes]

# Y = np.concatenate((np.zeros(size // 2), np.ones(size // 2)))[permuted_indexes]
# Y = np.expand_dims(Y, axis=1)

# Yhat = sigmoid.forward(linear.forward(X))
# original_loss = np.mean(mse.forward(Y, Yhat))

# for i in range(len(X)):
#     x = X[i:i+1]
#     y = Y[i]

#     res_lin = linear.forward(x)
#     yhat = sigmoid.forward(res_lin)

#     delta_loss = mse.backward(y, yhat)
#     delta_sigmoid = sigmoid.backward_delta(res_lin, delta_loss)

#     linear.backward_update_gradient(x, delta_sigmoid)
#     linear.update_parameters()
#     linear.zero_grad()

# res_lin = linear.forward(X)
# Yhat = sigmoid.forward(res_lin)
# updated_loss = np.mean(mse.forward(Y, np.rint(Yhat)))

# print("Perceptron original and updated loss + parameters :", original_loss, updated_loss, linear._parameters)

# a, b = linear._parameters[:, 0]
# V = np.linspace(-3, 3, 1000)

# plt.plot(V, (-a / b) * V)
# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Perceptron approximation")
# plt.show()

# # Testing digits classification
# alltrainx, alltrainy = DataHandler.load_usps_train()
# alltestX,alltestY = DataHandler.load_usps_test()

# neg = 6
# pos = 9

# dataX, dataY = DataHandler.get_usps([neg, pos], alltrainx, alltrainy)
# testX, testY = DataHandler.get_usps([neg, pos], alltestX, alltestY)

# dataY = np.where(dataY == pos, 1, 0)
# testY = np.where(testY == pos, 1, 0)

# dataY = np.expand_dims(dataY, axis=1)
# testY = np.expand_dims(testY, axis=1)

# # Without sequence utils
# linear1 = Linear(256, 64)
# linear2 = Linear(64, 1)
# mse = MSELoss()
# tanh = TanH()
# sigmoid = Sigmoid()

# res_lin = linear1.forward(testX)
# res_tanh = tanh.forward(res_lin)
# res_lin2 = linear2.forward(res_tanh)
# testYhat = sigmoid.forward(res_lin2)

# original_loss = np.mean(mse.forward(testY, testYhat))

# for i in range(1000):
#     x = dataX
#     y = dataY

#     res_lin = linear1.forward(x)
#     res_tanh = tanh.forward(res_lin)
#     res_lin2 = linear2.forward(res_tanh)
#     yhat = sigmoid.forward(res_lin2)

#     delta_mse = mse.backward(y, yhat)
#     delta_sig = sigmoid.backward_delta(res_lin2, delta_mse)
#     delta_lin2 = linear2.backward_delta(res_tanh, delta_sig)
#     delta_tanh = tanh.backward_delta(res_lin, delta_lin2)
#     delta_lin = linear1.backward_delta(x, delta_tanh)

#     linear2.backward_update_gradient(res_tanh, delta_sig)
#     linear2.update_parameters()
#     linear2.zero_grad()
#     linear1.backward_update_gradient(x, delta_tanh)
#     linear1.update_parameters()
#     linear1.zero_grad()

# res_lin = linear1.forward(testX)
# res_tanh = tanh.forward(res_lin)
# res_lin2 = linear2.forward(res_tanh)
# testYhat = sigmoid.forward(res_lin2)

# updated_loss = np.mean(mse.forward(testY, testYhat))

# print("Digits classification original and updated loss:", original_loss, updated_loss)

# # Same with the utils Sequential and Optim

# net = Sequential()
# net.append_modules([Linear(256, 64), 
#         TanH(), 
#         Linear(64, 1), 
#         Sigmoid()])

# optim = Optim(net)
# mse = MSELoss()

# testYhat = net.forward(testX)[-1]
# original_loss = np.mean(mse.forward(testY, testYhat))

# net = optim.SGD(dataX, dataY, len(dataX) // 2, 500)
# testYhat = net.forward(testX)[-1]
# updated_loss = np.mean(mse.forward(testY, testYhat))

# print("Digits classification original and updated loss:", original_loss, updated_loss)

# computed_pos = testX[np.where(np.rint(testYhat.ravel()) == 1)]

# index = np.random.randint(0, len(computed_pos))
# DataHandler.show_usps(computed_pos[index])
# plt.title("Digit classified as " + str(pos) + 
#     " by network picked randomly (" + str(index) + ")")
# plt.show()

# # Testing multi-class digits classification

# dataX, dataY = DataHandler.load_usps_train()
# testX, testY = DataHandler.load_usps_test()

# dataY1hot = np.zeros((dataY.size, 10))
# dataY1hot[np.arange(dataY.size), dataY] = 1

# testY1hot = np.zeros((testY.size, 10))
# testY1hot[np.arange(testY.size), testY] = 1

# batch_size = 5
# steps=len(dataX)

# net = Sequential()
# net.append_modules([Linear(256, 64),
#         TanH(),
#         Linear(64, 10),
#     ])

# optim = Optim(net, CELoss)
# ce = CELoss()

# testYhat = net.forward(testX)[-1]
# original_loss = np.mean(ce.forward(testY1hot, testYhat))

# net, losses = optim.SGD(dataX, dataY1hot, batch_size, steps, loss_length_modulo=10)
# testYhat = net.forward(testX)[-1]
# updated_loss = np.mean(ce.forward(testY1hot, testYhat))

# print("Digits classification original and updated loss:", original_loss, updated_loss)

# yhat_classes = np.argmax(testYhat, axis=1)
# diff = (1-np.mean(testY != yhat_classes))*100
# print(diff, "% accuracy")


# examples_shown = 6

# for i in range(1, examples_shown):
#     index = np.random.randint(0, len(testX) - examples_shown)
#     ax = plt.subplot(examples_shown, 1, i)
#     DataHandler.show_usps(testX[index])

#     title = "Real : " + str(testY[index]) + ", predicted : " + str(np.argmax(testYhat[index])) + " (index " + str(index) + ")" 
#     ax.set_title(title, { "fontsize": 10 })

# plt.tight_layout()
# plt.show()

# plt.plot(range(0, steps, 10), losses)
# plt.title("Multiclass loss evolution")
# plt.show()

# # Testing a compression network
# alltrainx, alltrainy = DataHandler.load_usps_train()
# alltestx, alltesty = DataHandler.load_usps_test()

# # neg = 6
# # pos = 9

# # dataX, dataY = DataHandler.get_usps([neg, pos], alltrainx, alltrainy)
# # testX, testY = DataHandler.get_usps([neg, pos], alltestx, alltesty)

# dataX, dataY = alltrainx, alltrainy
# testX, testY = alltestx, alltesty

# # dataX = np.where(dataX < 0.5, 0, 1)
# # testX = np.where(testX < 0.5, 0, 1)

# size = len(dataX)
# batch_size = 15
# steps = size * 20
# loss_array_length = steps // 20

# exec_start = time.time()

# net = Sequential()
# net.append_modules([
#     # Encoder
#     Linear(256, 200),
#     TanH(),
#     Linear(200, 64),
#     TanH(),
#     Linear(64, 16),
#     Sigmoid(),

#     # Decoder
#     Linear(16, 64), 
#     TanH(),
#     Linear(64, 200),
#     TanH(),
#     Linear(200, 256),
#     Sigmoid()
# ])

# loss = BCELoss
# optim = Optim(net, loss=loss)
# bce = loss()

# testXhat = net.forward(testX)[-1]
# original_loss = np.mean(bce.forward(testX, testXhat))

# net, losses = optim.SGD(dataX[:size], dataX[:size], batch_size, 
#     steps, loss_length_modulo=loss_array_length)

# testXhat = net.forward(testX)[-1]

# testXMiddle = net.forward(testX)[-7]
# updated_loss = np.mean(bce.forward(testX, testXhat))

# exec_time = time.time() - exec_start

# print("Digits classification original and updated loss:", original_loss, updated_loss)
# print("Execution in", str(exec_time), "s")

# # nb_original = len(np.unique(testX, axis=0))
# # nb_decompressed = len(np.unique(testXhat, axis=0))

# # print("How many different decompressed images ", nb_decompressed
# #     , "vs original number of images", nb_original)

# examples_shown = 6

# for i in range(1, examples_shown * 3, 3):
#     index = np.random.randint(0, len(testX) - examples_shown)
#     plt.subplot(examples_shown, 3, i)
#     DataHandler.show_usps(testX[index])
#     plt.subplot(examples_shown, 3, i+1)
#     DataHandler.show_usps(testXhat[index])
#     plt.subplot(examples_shown, 3, i+2)
#     plt.imshow(testXMiddle[index].reshape((4, 4)),
#                    interpolation="nearest", cmap="gray")

# plt.show()

# plt.plot(range(0, steps, loss_array_length), losses)
# plt.title("Loss evolution")
# plt.show()

# Testing convolution networks

dataX, dataY = DataHandler.load_usps_train()
testX, testY = DataHandler.load_usps_test()

dataY1hot = np.zeros((dataY.size, 10))
dataY1hot[np.arange(dataY.size), dataY] = 1

testY1hot = np.zeros((testY.size, 10))
testY1hot[np.arange(testY.size), testY] = 1

steps = 500

net = Sequential()
net.append_modules([Conv1D(3, 1, 32),
    MaxPool1D(2, 2),
    Flatten(),
    Linear(4064, 100),
    ReLU(),
    Linear(100, 10)
    ])

optim = Optim(net, CELoss)
ce = CELoss()

testYhat = net.forward(testX)[-1]
original_loss = np.mean(ce.forward(testY1hot, testYhat))

net, losses = optim.SGD(dataX, dataY1hot, 1, steps, loss_length_modulo=10)
testYhat = net.forward(testX)[-1]
updated_loss = np.mean(ce.forward(testY1hot, testYhat))

print("Digits classification original and updated loss:", original_loss, updated_loss)

yhat_classes = np.argmax(testYhat, axis=1)
diff = (1-np.mean(testY != yhat_classes))*100
print(diff, "% accuracy")

testYhat1hot = np.rint(CELoss.softmax(testYhat))

examples_shown = 6

for i in range(1, examples_shown):
    index = np.random.randint(0, len(testX) - examples_shown)
    ax = plt.subplot(examples_shown, 1, i)
    DataHandler.show_usps(testX[index])

    title = "Real : " + str(testY[index]) + ", predicted : " + str(np.argmax(testYhat[index])) + " (index " + str(index) + ")" 
    ax.set_title(title, { "fontsize": 10 })

plt.tight_layout()
plt.show()

plt.plot(range(0, steps, 10), losses)
plt.title("Multiclass loss evolution")
plt.show()