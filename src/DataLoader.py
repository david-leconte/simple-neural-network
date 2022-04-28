import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    """ This class loads data then used by the network. """

    USPStrain = "../data/USPS_train.txt"
    USPStest = "../data/USPS_test.txt"

    def loadUSPStrain():
        DataLoader.loadUSPS(DataLoader.USPStrain)

    def loadUSPStest():
        DataLoader.loadUSPS(DataLoader.USPStest)

    def loadUSPS(fn):
        with open(fn, "r") as f:
            f.readline()
            data = [[float(x) for x in l.split()]
                    for l in f if len(l.split()) > 2]

        tmp = np.array(data)
        return tmp[:, 1:], tmp[:, 0].astype(int)

    def getUSPS(l, datax, datay):
        if type(l) != list:
            resx = datax[datay == l, :]
            resy = datay[datay == l]
            return resx, resy

        tmp = list(zip(*[DataLoader.getUSPS(i, datax, datay) for i in l]))
        tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])
        return tmpx, tmpy

    def show_usps(data):
        plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
