# This script checks if the new version of the MLE is
# valid, i.e. if it produces consistent estimates. If it is
# these functions should be copied to the main package and
# this script should be converted to a unit test.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhMob.sampling import sampleData, getMissingPattern
from pyhMob.tools import getIncrements


class Conf:
    N = 600000
    LAMBDA = 500
    SD = 1.0
    C = 0.95
    PROB_P = 0.2  # probability of a pause after a flight, theta1
    PROB_F = 0.2  # probability of a flight after a pause, theta2
    MISS_TYPE = "on-off"
    ON = 5
    OFF = 5
    SEED = np.random.randint(0, 1000)


def Bprime(t, on, off):

    K = (on + off)
    Bp = K * np.floor(t/K) + on - 1
    return(Bp)


def getP(n, pPP, pFF):
    """
    Transition probabilities after n steps
    """
    const1 = 2 - pFF - pPP
    const2 = (pPP + pFF - 1) ** n
    A1 = np.array([[1 - pFF, 1 - pPP], [1 - pFF, 1 - pPP]])
    A2 = np.array([[1 - pPP, -(1 - pPP)], [-(1 - pFF), 1 - pFF]])
    return np.array((A1 + const2 * A2) / const1)


def getDG(M, on, off):

    dInds = np.where(M[:-1, -1] < 0)[0]
    LjCeils = M[dInds - 1, :]
    Lj1Floors = M[dInds + 1, :]
    tLjC = LjCeils[:, 2]
    tLj1F = Lj1Floors[:, 2]
    d = Bprime(tLjC, on, off) - tLjC - LjCeils[:, 5]
    g = tLj1F - (on + off) * np.floor(tLj1F/(on + off))
    delta = (d > 0).astype('int')
    gamma = (g > 0).astype('int')
    return(d, delta, g, gamma)


def logLik(stats, theta):
    """
    stats are passed as 
    (Nfp, Nff, Np, sumOfDeltas, d, delta, g, gamma)
    theta are (theta1, theta2)
    """
    Nfp, Nff, Np, sumOfDeltas, gLenghts = stats
    Ngaps = gLenghts.shape[0]
    logLik = Nfp * np.log(theta[0])
    logLik += Nff * np.log(1 - theta[0])
    logLik += Np * np.log(theta[1])
    logLik += (sumOfDeltas - Np) * np.log(1 - theta[1])

    for j in range(Ngaps):
        P = getP(gLenghts[j] + 1, theta[0], theta[1])
        logLik += np.log(P[0, 0])

    return(logLik)


def getSuffStats(pings):

    M = getIncrements(pings)

    pInds = np.where(M[:, -1] == 1)[0]
    fInds = np.where(M[:, -1] == 0)[0]
    dInds = np.where(M[:, -1] < 0)[0]
    # number of pauses
    Np = pInds.shape[0]
    # is the last increment a flight?
    IsF = M[-1, -1] == 0

    # Recall that the trajectory starts wiht a flight. Therefore,
    # every flight, except for one that ends the trajectory,
    # is either followed by a flight or by a pause. Thus the
    # number of flights followed by a pause is just the number of
    # pauses.
    if IsF:
        Nff = np.where(M[fInds[:-1] + 1, -1] == 0)[0].shape[0]
        Nfp = np.where(M[fInds[:-1] + 1, -1] == 1)[0].shape[0]
    else:
        Nff = np.where(M[fInds + 1, -1] == 0)[0].shape[0]
        Nfp = np.where(M[fInds + 1, -1] == 1)[0].shape[0]

    sumOfDeltas = M[pInds, -2].sum()
    return(Nfp, Nff, Np, sumOfDeltas, M[dInds, 5])


if __name__ == "__main__":

    pd.options.display.precision = 4
    np.set_printoptions(precision=4)
    np.random.seed(Conf.SEED)
    print(f"seed = {Conf.SEED}")
    truePings, pings = sampleData(Conf.N, Conf.LAMBDA, 1.0, Conf.SD,
                                  Conf.C, Conf.PROB_P, Conf.PROB_F,
                                  inclTimes=True)

    print("Done simulating data!")
    misParms = {"on": Conf.ON, "off": Conf.OFF}
    misInds = getMissingPattern(Conf.N, missType=Conf.MISS_TYPE,
                                params=misParms)
    
    stats = getSuffStats(pings)

    def ll(params):
        return(logLik(stats, params))

    # find the MLE by grid search
    Npoints = 30
    M = min(5 * max(Conf.PROB_F, Conf.PROB_P), 1)
    thetas = np.linspace(1e-3, M-1e-3, Npoints)
    logliks = np.zeros((Npoints, Npoints))

    for idx1 in range(Npoints):
        # print(f"{idx1} out of {Npoints}")
        for idx2 in range(Npoints):
            params = np.array([thetas[idx1], thetas[idx2]])
            # coordinates for plotting are different than matrix
            # indexing which is  why we have to do '(1-y, x)'
            # instead of y
            logliks[Npoints - 1 - idx2, idx1] = ll(params)

    argmax = np.unravel_index(np.nanargmax(logliks), logliks.shape)
    MLE = (thetas[argmax[1]], thetas[Npoints - 1 - argmax[0]])
    print(f"theta MLE = {MLE}")
    
    im = plt.imshow(logliks, extent=(0.001, M - 0.001, 0.001, M - 0.001))
    plt.axhline(y=Conf.PROB_F, linestyle="dashed", color="black", linewidth=0.5)
    plt.axvline(x=Conf.PROB_P, linestyle="dashed", color="black", linewidth=0.5)
    mle = plt.scatter(MLE[0], MLE[1], marker="x", color="red")
    plt.colorbar(im)
    plt.xlabel(f"theta 1")
    plt.ylabel(f"theta 2")
    plt.show()
    
