import pandas as pd
import numpy as np
import dill
import mobilityMetrics as mm
from plotting import getLimits
from imputation import fillGaps
from data.loadtraces import load_data


# simulation settings
class Conf:
    N = 1000
    NGRID = 40
    SET = "Statefair"
    NREP = 50
    NSIMS = 50
    SEED = None
    MISS_TYPE = "contiguous"
    NBINS = 40


def getMissingPattern(n, perc):

    if Conf.MISS_TYPE == "contiguous":
        misPat = np.ones(n)
        gapStart = round(n * (0.5 - perc / 2))
        gapStop = round(n * (0.5 + perc / 2))
        misPat[gapStart:gapStop] = 0
        misInds = np.where(misPat == 0)[0]
    return(misInds)


np.random.seed(Conf.SEED)
scores = ["JenSha", "dTrav", "rOfG", "dHist"]
colNames = ["simNo", "percMis", "imputNo", "imputMet"] + scores
methods = ["LI", "MJ", "B"]
fracMis = np.linspace(0.01, 0.9, num=Conf.NGRID)


results = pd.DataFrame(columns=colNames)
trajectories = dict()

data = load_data(Conf.SET)
alldata = np.vstack(data)
trueHist, xedg, yedg = np.histogram2d(alldata[:, 0]/1000, alldata[:, 1]/1000,
                                      bins = Conf.NBINS, normed = True)
trueDTrav = mm.distanceTravelled(alldata)
trueROfG = mm.radiusOfGyration(alldata)


for simNo in range(len(data)):

    print(f"working on data set {simNo}")
    truePings = np.array(data[simNo]/1000)
    nPings = truePings.shape[0]

    trajectories[simNo] = {"truth": truePings}
    for perc in fracMis:

        print(f"percentage missing: {perc}")
        # hide observations
        misInds = getMissingPattern(nPings, perc)
        pings = np.copy(truePings)
        pings[misInds, :] = np.nan
        trajectories[simNo][perc] = {"obs": pings}

        for repNo in range(Conf.NREP):
            
            trajectories[simNo][perc][repNo] = dict()
            for method in methods:

                filled = fillGaps(pings, method, consolidate=True)
                hist = np.histogram2d(filled[:, 0], filled[:, 1],
                                      bins=(xedg, yedg), normed=True)[0]
                JenSha = mm.JensenShannon(hist, trueHist)
                dTrav = abs(mm.distanceTravelled(filled) - trueDTrav)/trueDTrav
                rOfG = abs(mm.radiusOfGyration(filled) - trueROfG)/trueROfG
                dHist = mm.histRMSE(hist, trueHist)
                
                trajectories[simNo][perc][repNo][method] = filled

                vals = [simNo, perc, repNo, method, JenSha, dTrav, rOfG, dHist]
                res = pd.DataFrame(dict(zip(colNames, vals)), index=[0])
                results = results.append(res)

    toStore = {"scores": results,
               "trajectories": trajectories,
               "settings": Conf}

    with open(f"results-{Conf.SET}-{simNo}.pickle", 'wb') as opFile:
        dill.dump(toStore, opFile)
