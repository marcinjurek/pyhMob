# import pdb
import numpy as np
# from src.tools import getIncrements
# import matplotlib.pyplot as plt


def getPartLogLik(chain, pPP, pFF):

    pInds = np.where(chain[:-1] == 1)[0]
    fInds = np.where(chain[:-1] == 0)[0]

    Npp = np.where(chain[pInds + 1] == 1)[0].shape[0]
    Npf = np.where(chain[pInds + 1] == 0)[0].shape[0]
    Nff = np.where(chain[fInds + 1] == 0)[0].shape[0]
    Nfp = np.where(chain[fInds + 1] == 1)[0].shape[0]
    
    Ns = np.array([Npp, Npf, Nff, Nfp])
    logPs = np.log(np.array([1 - pPP, pPP, 1 - pFF, pFF]))

    return np.dot(Ns, logPs)


def P(n, pPP, pFF):
    """
    Transition probabilities after n steps
    """
    const1 = 2 - pFF - pPP
    const2 = (pPP + pFF - 1) ** n
    A1 = np.array([[1 - pFF, 1 - pPP], [1 - pFF, 1 - pPP]])
    A2 = np.array([[1 - pPP, -(1 - pPP)], [-(1 - pFF), 1 - pFF]])
    return np.array((A1 + const2 * A2) / const1)


def newLikelihood(pings, Io, Iu, pPP, pFF):
    """
    This method evaluates the likelihood of the chain of
    pauses and flights.
    1 = pause, 0 = flight
    """
    incs = pings[:-1, :2] - pings[1:, :2]
    incType = np.sum(np.abs(incs), axis=1) == 0
    incType = incType.astype('int')

    breakInds = np.flatnonzero(np.diff(pings[:, -1]) > 1)
    gapStarts = pings[breakInds, -1]
    gapEnds = pings[np.flatnonzero(np.diff(pings[:, -1]) > 1) + 1, -1]
    gapLengths = gapEnds - gapStarts

    parts = np.array_split(incType, breakInds)
    loglik = getPartLogLik(parts[-1], pPP, pFF)

    for idx in range(len(parts)-1):
        loglik += getPartLogLik(parts[idx], pPP, pFF)
        gapLength = gapLengths[idx]
        transMat = P(gapLength, pPP, pFF)
        lastState = parts[idx][-1]
        firstState = parts[idx][0]
        transProb = transMat[lastState, firstState]
        loglik += np.log(transProb)

    return loglik


def getMLEMarkovOnOff(pings, Io, Iu, Ngrid, pPPmax, pFFmax):

    Npp = Ngrid
    Nff = Ngrid
    pPPgrid = np.linspace(1e-3, pPPmax, Npp)
    pFFgrid = np.linspace(1e-3, pFFmax, Nff)

    MLEs = np.zeros((Npp, Nff))
    for pPPidx in range(Npp):
        for pFFidx in range(Nff):

            pPP = pPPgrid[pPPidx]
            pFF = pFFgrid[pFFidx]
            MLEs[pPPidx, pFFidx] = newLikelihood(pings, Io, Iu, pPP, pFF)

    maxIdx = np.where(MLEs == np.amax(MLEs))
    pPPest = pPPgrid[maxIdx[0]][0]
    pFFest = pFFgrid[maxIdx[1]][0]

    # extent = np.min(pPPgrid), np.max(pPPgrid), np.min(pFFgrid), np.max(pFFgrid)
    # plt.imshow(MLEs, extent=extent, origin="lower")
    # plt.colorbar()
    # plt.scatter(pPPest, pFFest, marker="x", color="black")
    # plt.show()
    assert np.amax(MLEs) == newLikelihood(pings, Io, Iu, pPPest, pFFest)
    return pPPest, pFFest
