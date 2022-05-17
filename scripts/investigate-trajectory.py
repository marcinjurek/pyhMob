# imports
import numpy as np
import numpy.linalg as ln
import pdb
from loadtraces import load_data
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import mobilityMetrics as mm
from imputation import fillGapsRheeData
from likelihood import getMLEs
from plotting import getLimits, TrajGraph
from sampling import sampleMissing

# settings
np.random.seed(1996)
N_SAMP_GAP = 1
NGRID = 50
LAMBDA = 50
SET = "Statefair"  # KAIST, Orlando, NewYork, NCSU, Statefair
NREP = 50
LWD = 0.8
SMALL_SIZE = 8
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


def flat(doubleList):

    return([item for sublist in doubleList for item in sublist])


def evalMobMetrics(pingsList, trueMetrics, xEdges, yEdges):

    listLength = len(pingsList)
    for pings in pingsList:
        H = np.histogram2d(pings[:, 0], pings[:, 1], bins=(xEdges, yEdges))
        metrics = {"hist": 0, "dTrav": 0, "rOfG": 0}
        metrics["hist"] += H[0] / np.sum(H[0])
        metrics["dTrav"] += mm.distanceTravelled(pings)
        metrics["rOfG"] += mm.radiusOfGyration(pings)
    for k, v in metrics.items():
        metrics[k] = v/listLength
    return metrics


# main script
if __name__ == "__main__":

    fig = plt.figure()

    data = load_data(SET)
    ind = np.random.choice(np.arange(len(data)), size=1, replace=False)[0]
    truePings = data[ind]/1000

    n = truePings.shape[0]
    zeros = np.zeros(NGRID)
    zerosD = {"MJ": dc(zeros), "LI": dc(zeros)}
    dmetrics = {"dTrav": dc(zerosD), "rOfG": dc(zerosD), "hist": dc(zerosD)}
    fracObs = np.linspace(0.1, 0.9, num=NGRID)

    for idx, perc in enumerate(fracObs[::-1]):

        print(idx)
        imputed = np.copy(truePings)
        LIed = np.copy(truePings)

        LIs = [None] * NREP
        imputes = [None] * NREP
        for repNo in range(NREP):

            # hide observations
            #misPat = sampleMissing(n, LAMBDA, perc / (1 - perc) * LAMBDA)
            misPat = np.ones(n)
            gapStart = round(n * (0.5 - perc / 2))
            gapStop = round(n * (0.5 + perc / 2))
            misPat[gapStart:gapStop] = 0
            print(f"percent observed {sum(misPat) * 100 / n}")
            misInds = np.where(misPat == 0)[0]

            imputes[repNo] = np.copy(truePings)
            LIs[repNo] = np.copy(truePings)
            if misInds.shape[0] > 0:
                pings = np.copy(truePings)
                pings[misInds, :] = np.nan
                params = getMLEs(pings, verbose=True)
                imputes[repNo][misInds, :] = fillGapsRheeData(pings, "MJ", params)
                LIs[repNo][misInds, :] = fillGapsRheeData(pings, "LI", params)
            
        xlim = getLimits([truePings] + LIs + imputes, "x")
        ylim = getLimits([truePings] + LIs + imputes, "y")
        xEdges = np.linspace(xlim[0], xlim[1], 11)
        yEdges = np.linspace(ylim[0], ylim[1], 11)

        trueDTrav = mm.distanceTravelled(truePings)
        trueROfG = mm.radiusOfGyration(truePings)
        trueHist = np.histogram2d(truePings[:, 0], truePings[:, 1],
                                  bins=(xEdges, yEdges))[0]
        trueHist /= np.sum(trueHist)
        dTravI = 0
        dTravL = 0
        rOfGI = 0
        rOfGL = 0
        dHistI = 0
        dHistL = 0
        for repNo in range(NREP):
            dTravI += abs(mm.distanceTravelled(imputes[repNo]) - trueDTrav)
            dTravL += abs(mm.distanceTravelled(LIs[repNo]) - trueDTrav)
            rOfGI += abs(mm.radiusOfGyration(imputes[repNo]) - trueROfG)
            rOfGL += abs(mm.radiusOfGyration(LIs[repNo]) - trueROfG)
            histI = np.histogram2d(imputes[repNo][:, 0], imputes[repNo][:, 1],
                                   bins=(xEdges, yEdges))[0]
            histI /= np.sum(histI)
            dHistI += np.linalg.norm(histI - trueHist, ord=2)
            histL = np.histogram2d(LIs[repNo][:, 0], LIs[repNo][:, 1],
                                   bins=(xEdges, yEdges))[0]
            histL /= np.sum(histL)
            dHistL += np.linalg.norm(histL - trueHist, ord=2)

        dmetrics["dTrav"]["LI"][idx] = dTravL / (trueDTrav * NREP)
        dmetrics["dTrav"]["MJ"][idx] = dTravI / (trueDTrav * NREP)
        dmetrics["rOfG"]["LI"][idx]  = rOfGI  / (trueROfG * NREP)
        dmetrics["rOfG"]["MJ"][idx]  = rOfGL  / (trueROfG * NREP)
        dmetrics["hist"]["LI"][idx]  = dHistL / NREP
        dmetrics["hist"]["MJ"][idx]  = dHistI / NREP

    axDTrav = fig.add_subplot(1, 3, 1)
    axDTrav.plot(fracObs, dmetrics["dTrav"]["LI"], color="black")
    axDTrav.plot(fracObs, dmetrics["dTrav"]["MJ"], color="blue")
    axDTrav.set_title("dist trav difference")

    axROfG = fig.add_subplot(1, 3, 2)
    axROfG.plot(fracObs, dmetrics["rOfG"]["LI"], color="black")
    axROfG.plot(fracObs, dmetrics["rOfG"]["MJ"], color="blue")
    axROfG.set_title("r of g difference")

    axHist = fig.add_subplot(1, 3, 3)
    axHist.plot(fracObs, dmetrics["hist"]["LI"], color="black")
    axHist.plot(fracObs, dmetrics["hist"]["MJ"], color="blue")
    axHist.set_title("histogram difference")

    plt.show()
