# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pdb
import numpy as np
from sampling import sampleData
from sklearn.metrics.pairwise import euclidean_distances as dist
from plotting import plotResults
# from tools import findPauses, findFlights, getPattern
# import scipy.stats as stats


class Increment(object):

    def __init__(self, dx, dy, dt):

        self.dx = dx
        self.dy = dy
        self.dt = dt


class Location(object):

    def __init__(self, x, y, t):

        self.x = x
        self.y = y
        self.t = t

    def __add__(self, increment):

        self.x += increment.dx
        self.y += increment.dy
        self.t += increment.dt


def gapFillGP(pings):

    x = np.matrix(pings[:, 0]).T
    y = np.matrix(pings[:, 1]).T

    t = np.arange(pings.shape[0])
    obsI = np.where(~np.isnan(x))[0]
    misI = np.where(np.isnan(x))[0]
    mis = t[misI].reshape(-1, 1)
    obs = t[obsI].reshape(-1, 1)
    
    # vMis = np.matrix(np.exp(-dist(mis)))
    covMisObs = np.matrix(np.exp(-dist(mis, obs)/30))
    vObs = np.matrix(np.exp(-dist(obs)/30))

    pObs = np.linalg.inv(vObs)
    
    xMis = covMisObs * pObs * x[obsI, :]
    yMis = covMisObs * pObs * y[obsI, :]

    newPings = np.hstack((xMis, yMis))
    newPings = np.vstack((pings[misI[0] - 1, :], newPings, pings[misI[-1] + 1, :]))
    
    return newPings


if __name__ == "__main__":

    np.random.seed(1996)
    N = 1000
    LAMBDA = 10
    FRAC_OBS = 0.99
    SD = 0.1
    C = 0.9
    PROB_P = 0.01
    PROB_F = 0.01

    truePings, pings = sampleData(N, LAMBDA, FRAC_OBS, SD, C, PROB_P, PROB_F)
    misInds = np.arange(850, 950)
    pings[misInds, :] = np.nan
    gapStart = 300
    gapStop = 600
    
    start = Location(pings[gapStart, 0], pings[gapStart], gapStart)
    stop = Location(pings[gapStop + 1, 0], pings[gapStop + 1, 1], gapStop)
    
    gapFill = gapFillGP(pings)

    pdb.set_trace()
    plt.plot(truePings[:, 1])
    misInds = np.append(849, np.append(misInds, 950))
    plt.plot(misInds, gapFill[:, 1])
    plt.show()
    
    # plotResults(truePings, [[gapFill]], np.isnan(pings[:, 0]))
