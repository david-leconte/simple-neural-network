import numpy as np
import matplotlib.pyplot as plt


class DataHandler(object):
    """ This class loads data then used by the network. """

    USPStrain = "./data/USPS_train.txt"
    USPStest = "./data/USPS_test.txt"

    def loadUSPStrain():
        return DataHandler.loadUSPS(DataHandler.USPStrain)

    def loadUSPStest():
        return DataHandler.loadUSPS(DataHandler.USPStest)

    def loadUSPS(fn):
        with open(fn, "r") as f:
            f.readline()
            data = [[float(x) for x in l.split()]
                    for l in f if len(l.split()) > 2]

        tmp = np.array(data)
        return tmp[:, 1:], tmp[:, 0].astype(int)

    def getUSPS(l, datax, datay):
        assert len(datax) == len(datay)
        if type(l) != list:
            resx = datax[datay == l, :]
            resy = datay[datay == l]
            return resx, resy

        tmp = list(zip(*[DataHandler.getUSPS(i, datax, datay) for i in l]))
        tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])

        permutation = np.random.permutation(len(tmpx))
        return tmpx[permutation], tmpy[permutation]

    def showUSPS(data):
        plt.imshow(data.reshape((16, 16)),
                   interpolation="nearest", cmap="gray")
