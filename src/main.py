import warnings
import sys
import cProfile
import time
import os

from matplotlib import pyplot as plt
import numpy as np

from Utils import DataHandler, Sequential, Optim
from Losses import MSELoss, CELoss, BCELoss
from Modules import Linear, Sigmoid, TanH, Conv1D, MaxPool1D, ReLU, Flatten

# warnings.filterwarnings("ignore", category=RuntimeWarning)


def linear_regression(plot=True):
    """Using the Linear module and MSELoss for linear regression"""

    d = 2 if plot else 4

    mse = MSELoss()
    linear = Linear(d, 1)

    size = 1000
    X = np.random.randint(-3, 3, size=(size, d))
    W = np.random.randn(d) * 3
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

    print("Linear regression original and updated loss + parameters :",
          original_loss, updated_loss)
    print("Original parameters:", W, ", computed parameters :",
          np.ravel(linear._parameters))

    if plot:
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], X @ W, c='green',
                     label="Original function", s=40)
        ax.scatter3D(X[:, 0], X[:, 1], X @ np.ravel(linear._parameters),
                     c='red', label="Computed function", s=2)

        plt.legend()

        title = "3D linear regression, W = " + str(W)
        plt.title(title)

        plt.show()


def perceptron(plot=True):
    """ Using the Linear and Sigmoid module with MSELoss to test a perceptron """

    d = 2
    mse = MSELoss()
    linear = Linear(d, 1)
    sigmoid = Sigmoid()

    size = 5000

    X = np.zeros((size, d))
    X[:size // 2] = np.random.normal((-2, -2), 1, size=(size // 2, 2))
    X[size // 2:] = np.random.normal((2, 2), 1, size=(size // 2, 2))

    permuted_indexes = np.random.permutation(size)
    X = X[permuted_indexes]

    Y = np.concatenate(
        (np.zeros(size // 2), np.ones(size // 2)))[permuted_indexes]
    Y = np.expand_dims(Y, axis=1)

    Yhat = sigmoid.forward(linear.forward(X))
    original_loss = np.mean(mse.forward(Y, Yhat))

    for i in range(len(X)):
        x = X[i:i+1]
        y = Y[i]

        res_lin = linear.forward(x)
        yhat = sigmoid.forward(res_lin)

        delta_loss = mse.backward(y, yhat)
        delta_sigmoid = sigmoid.backward_delta(res_lin, delta_loss)

        linear.backward_update_gradient(x, delta_sigmoid)
        linear.update_parameters()
        linear.zero_grad()

    res_lin = linear.forward(X)
    Yhat = sigmoid.forward(res_lin)
    updated_loss = np.mean(mse.forward(Y, np.rint(Yhat)))

    print("Perceptron original and updated loss + parameters :",
          original_loss, updated_loss, linear._parameters)

    accuracy = (1 - np.mean(Y != np.rint(Yhat))) * 100
    print(accuracy, "% accuracy")

    if plot:
        a, b = linear._parameters[:, 0]
        V = np.linspace(-4, 4, 1000)

        plt.plot(V, (-a / b) * V, c="black")
        plt.scatter(X[np.where(Y == 0), 0],
                    X[np.where(Y == 0), 1], c="red", s=1)
        plt.scatter(X[np.where(Y == 1), 0],
                    X[np.where(Y == 1), 1], c="green", s=1)
        plt.title("Perceptron approximation")
        plt.show()


def binary_digits_classif(SGD=True, plot=True):
    """ Using Linear, Sigmoid, TanH modules and MSELoss for binary digits classification (USPS) """

    alltrainx, alltrainy = DataHandler.load_usps_train()
    alltestX, alltestY = DataHandler.load_usps_test()

    neg = 6
    pos = 9

    dataX, dataY = DataHandler.get_usps([neg, pos], alltrainx, alltrainy)
    testX, testY = DataHandler.get_usps([neg, pos], alltestX, alltestY)

    dataY = np.where(dataY == pos, 1, 0)
    testY = np.where(testY == pos, 1, 0)

    dataY = np.expand_dims(dataY, axis=1)
    testY = np.expand_dims(testY, axis=1)

    # Without sequence utils
    if not SGD:
        linear1 = Linear(256, 64)
        linear2 = Linear(64, 1)
        mse = MSELoss()
        tanh = TanH()
        sigmoid = Sigmoid()

        res_lin = linear1.forward(testX)
        res_tanh = tanh.forward(res_lin)
        res_lin2 = linear2.forward(res_tanh)
        testYhat = sigmoid.forward(res_lin2)

        original_loss = np.mean(mse.forward(testY, testYhat))

        for i in range(1000):
            x = dataX
            y = dataY

            res_lin = linear1.forward(x)
            res_tanh = tanh.forward(res_lin)
            res_lin2 = linear2.forward(res_tanh)
            yhat = sigmoid.forward(res_lin2)

            delta_mse = mse.backward(y, yhat)
            delta_sig = sigmoid.backward_delta(res_lin2, delta_mse)
            delta_lin2 = linear2.backward_delta(res_tanh, delta_sig)
            delta_tanh = tanh.backward_delta(res_lin, delta_lin2)
            delta_lin = linear1.backward_delta(x, delta_tanh)

            linear2.backward_update_gradient(res_tanh, delta_sig)
            linear2.update_parameters()
            linear2.zero_grad()
            linear1.backward_update_gradient(x, delta_tanh)
            linear1.update_parameters()
            linear1.zero_grad()

        res_lin = linear1.forward(testX)
        res_tanh = tanh.forward(res_lin)
        res_lin2 = linear2.forward(res_tanh)
        testYhat = sigmoid.forward(res_lin2)

        updated_loss = np.mean(mse.forward(testY, testYhat))

    # Same with the utils Sequential and Optim
    else:
        net = Sequential()
        net.append_modules([Linear(256, 64),
                            TanH(),
                            Linear(64, 1),
                            Sigmoid()])

        optim = Optim(net)
        mse = MSELoss()

        testYhat = net.forward(testX)[-1]
        original_loss = np.mean(mse.forward(testY, testYhat))

        net = optim.SGD(dataX, dataY, len(dataX) // 2, 500)
        testYhat = net.forward(testX)[-1]
        updated_loss = np.mean(mse.forward(testY, testYhat))

    print("Binary digits classification original and updated loss:",
          original_loss, updated_loss)

    accuracy = (1 - np.mean(testY != np.rint(testYhat))) * 100
    print(accuracy, "% accuracy")

    if plot:
        for i in range(1, examples_shown + 1):
            index = np.random.randint(0, len(testX) - examples_shown)
            ax = plt.subplot(examples_shown, 1, i)
            DataHandler.show_usps(testX[index])

            real_class = pos if testY[index] == 1 else neg
            computed_class = pos if np.rint(testYhat[index]) == 1 else neg

            title = "Real : " + str(real_class) + ", predicted : " + \
                str(computed_class) + " (index " + str(index) + ")"
            ax.set_title(title, {"fontsize": 10})

        plt.tight_layout()
        plt.show()


def multiclass_digits_classif(plot=True):
    """ Sequence utils and modules quoted above + SoftMax and CELoss for multiclass digits classification """

    dataX, dataY = DataHandler.load_usps_train()
    testX, testY = DataHandler.load_usps_test()

    dataY1hot = np.zeros((dataY.size, 10))
    dataY1hot[np.arange(dataY.size), dataY] = 1

    testY1hot = np.zeros((testY.size, 10))
    testY1hot[np.arange(testY.size), testY] = 1

    batch_size = 5
    steps = len(dataX)

    net = Sequential()
    net.append_modules([Linear(256, 64),
                        TanH(),
                        Linear(64, 10),
                        ])

    optim = Optim(net, CELoss)
    ce = CELoss()

    testYhat = net.forward(testX)[-1]
    original_loss = np.mean(ce.forward(testY1hot, testYhat))

    net, losses = optim.SGD(dataX, dataY1hot, batch_size,
                            steps, loss_length_modulo=10)
    testYhat = net.forward(testX)[-1]
    updated_loss = np.mean(ce.forward(testY1hot, testYhat))

    print("Multiclass digits classification original and updated loss:",
          original_loss, updated_loss)

    yhat_classes = np.argmax(testYhat, axis=1)
    accuracy = (1 - np.mean(testY != yhat_classes)) * 100
    print(accuracy, "% accuracy")

    if plot:
        for i in range(1, examples_shown + 1):
            index = np.random.randint(0, len(testX) - examples_shown)
            ax = plt.subplot(examples_shown, 1, i)
            DataHandler.show_usps(testX[index])

            title = "Real : " + str(testY[index]) + ", predicted : " + str(
                np.argmax(testYhat[index])) + " (index " + str(index) + ")"
            ax.set_title(title, {"fontsize": 10})

        plt.tight_layout()
        plt.show()

        plt.plot(range(0, steps, 10), losses)
        plt.title("Multiclass loss evolution")
        plt.show()


def compression_net(plot=True):
    """ Sequence utils and modules Linear, TanH, Sigmoid + BCELos for testing a compression """

    alltrainx, alltrainy = DataHandler.load_usps_train()
    alltestx, alltesty = DataHandler.load_usps_test()

    # neg = 6
    # pos = 9

    # dataX, dataY = DataHandler.get_usps([neg, pos], alltrainx, alltrainy)
    # testX, testY = DataHandler.get_usps([neg, pos], alltestx, alltesty)

    dataX = alltrainx / 2
    testX = alltestx / 2

    # dataX = np.where(dataX < 0.5, 0, 1)
    # testX = np.where(testX < 0.5, 0, 1)

    batch_size = 100
    max_steps = 100000
    loss_array_length = 1000

    compression = 10

    net = Sequential()
    net.append_modules([
        # Encoder
        Linear(256, 64),
        TanH(),
        Linear(64, compression),
        TanH(),

        # Decoder
        Linear(compression, 64),
        TanH(),
        Linear(64, 256),
        Sigmoid()
    ])

    loss = BCELoss
    optim = Optim(net, loss=loss)
    bce = loss()

    testXhat = net.forward(testX)[-1]
    original_loss = np.mean(bce.forward(testX, testXhat))

    net, losses = optim.SGD(dataX, dataX, batch_size,
                            max_steps, losses_save_modulo=loss_array_length, early_stop=0.32)
    testXhat = net.forward(testX)[-1]

    sig = Sigmoid()
    testXMiddle = sig.forward(net.forward(testX)[-5])
    updated_loss = np.mean(bce.forward(testX, testXhat))

    print("Digits rebuild original and updated loss:",
          original_loss, updated_loss)

    if plot:
        for i in range(1, examples_shown * 3, 3):
            index = np.random.randint(0, len(testX) - examples_shown)
            plt.subplot(examples_shown, 3, i)
            DataHandler.show_usps(testX[index])
            plt.subplot(examples_shown, 3, i+1)
            DataHandler.show_usps(testXhat[index])
            plt.subplot(examples_shown, 3, i+2)
            plt.imshow(testXMiddle[index].reshape((2, 5)),
                       interpolation="nearest", cmap="gray")

        plt.show()

        plt.plot(np.arange(len(losses)) * loss_array_length, losses)
        plt.title("Loss evolution")
        plt.show()


def convolution_digits_classif(plot=True):
    """ Using 1 dimension convolution network for multiclass digits classification """

    dataX, dataY = DataHandler.load_usps_train()
    testX, testY = DataHandler.load_usps_test()

    dataY1hot = np.zeros((dataY.size, 10))
    dataY1hot[np.arange(dataY.size), dataY] = 1

    testY1hot = np.zeros((testY.size, 10))
    testY1hot[np.arange(testY.size), testY] = 1

    steps = 250

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

    print("Digits classification original and updated loss:",
          original_loss, updated_loss)

    yhat_classes = np.argmax(testYhat, axis=1)
    diff = (1-np.mean(testY != yhat_classes))*100
    print(diff, "% accuracy")

    if plot:
        for i in range(1, examples_shown):
            index = np.random.randint(0, len(testX) - examples_shown)
            ax = plt.subplot(examples_shown, 1, i)
            DataHandler.show_usps(testX[index])

            title = "Real : " + str(testY[index]) + ", predicted : " + str(
                np.argmax(testYhat[index])) + " (index " + str(index) + ")"
            ax.set_title(title, {"fontsize": 10})

        plt.tight_layout()
        plt.show()

        plt.plot(range(0, steps, 10), losses)
        plt.title("Multiclass loss evolution")
        plt.show()


plot = True if "--plot" in sys.argv else False
profiling = True if "--profile" in sys.argv else False
current_dir = os.path.dirname(__file__)

examples_shown = 6

if(profiling):
    profiler = cProfile.Profile()
    profiler.enable()

# linear_regression(plot=plot)
# perceptron(plot=plot)
# binary_digits_classif(plot=plot)
# multiclass_digits_classif(plot=plot)
compression_net(plot=plot)
# convolution_digits_classif(plot=plot)

if(profiling):
    profiler.disable()
    profiler.dump_stats("profiles/" + str(time.time()) + ".prof")
