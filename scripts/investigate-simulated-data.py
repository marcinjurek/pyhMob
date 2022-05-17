# imports
import pandas as pd
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt

import mobilityMetrics as mm
from imputation import fillGaps
from sampling import sampleMissing, sampleData

# simulation settings
# np.random.seed(1996)
N = 1000
NGRID = 3
LAMBDA = 500
NREP = 2
NSIMS = 2
SD = 1.0
C = 0.9
PROB_P = 0.01
PROB_F = 0.1

# plot settings
LWD = 0.8
SMALL_SIZE = 8
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


def flat(doubleList):

    return([item for sublist in doubleList for item in sublist])


# main script
if __name__ == "__main__":

    scores = ["dTrav", "rOfG", "JenSha"]
    colNames = ["simNo", "percObs", "imputNo", "imputMet"] + scores
                
    results = pd.DataFrame(columns = colNames)

    bounds = (-SD * N, SD * N)
    nBins = 2 * N * SD / 10 * SD
    xEdges = np.linspace(min(bounds), max(bounds), round(2 * N / 10))
    yEdges = np.linspace(min(bounds), max(bounds), round(2 * N / 10))
    bins = (xEdges, yEdges)
    
    for simNo in range(NSIMS):
        truePings, pings = sampleData(N, LAMBDA, FRAC_OBS, SD, C, PROB_P, PROB_F)

        assert np.max(truePings) < np.max(bounds) and np.min(truePings) > np.min(bounds)

        # n = truePings.shape[0]
        # assert n == N
        
        fracObs = np.linspace(0.2, 0.5, num=NGRID)
        
        for idx, perc in enumerate(fracObs):

            print(f"percentage observed: {perc}")
            imputed = np.copy(truePings)
            LIed = np.copy(truePings)
            for repNo in range(NREP):
            
                # hide observations
                misPat = np.ones(N)
                gapStart = round(N * (0.5 - perc / 2))
                gapStop = round(N * (0.5 + perc / 2))
                misPat[gapStart:gapStop] = 0
                
                misInds = np.where(misPat == 0)[0]
                pings = np.copy(truePings)                
                pings[misInds, :] = np.nan

                trueDTrav = mm.distanceTravelled(truePings)
                trueROfG = mm.radiusOfGyration(truePings)
                trueHist = np.histogram2d(truePings[:, 0], truePings[:, 1],
                                          bins=bins, normed=True)[0]

                for method in methods:

                    fill = fillGaps(pings, method, consolidate=True)
                    
                dTravI = abs(mm.distanceTravelled(imputes) - trueDTrav)
                dTravL = abs(mm.distanceTravelled(LIs) - trueDTrav)
                dTravB = abs(mm.distanceTravelled(barnett) - trueDTrav)
                
                rOfGI = abs(mm.radiusOfGyration(imputes) - trueROfG)
                rOfGL = abs(mm.radiusOfGyration(LIs) - trueROfG)
                rOfGB = abs(mm.radiusOfGyration(barnett) - trueROfG)

                histI = np.histogram2d(imputes[:, 0], imputes[:, 1],
                                       bins=bins, normed=True)[0]
                dHistI = mm.JensenShannon(histI, trueHist)

                histL = np.histogram2d(LIs[:, 0], LIs[:, 1],
                                       bins=bins, normed=True)[0]
                dHistL = mm.JensenShannon(histL, trueHist)
                
                histB = np.histogram2d(barnett[:, 0], barnett[:, 1],
                                       bins=bins, normed=True)[0]
                dHistB = mm.JensenShannon(histB, trueHist)

                valsI = [simNo, perc, repNo, "MJ", dTravI, rOfGI, dHistI]
                resI = pd.DataFrame(dict(zip(colNames, valsI)), index=[0])
                results = results.append(resI)

                valsL = [simNo, perc, repNo, "LI", dTravL, rOfGL, dHistL]
                resL = pd.DataFrame(dict(zip(colNames, valsL)), index=[0])
                results = results.append(resL)

                valsB = [simNo, perc, repNo, "BA", dTravB, rOfGB, dHistB]
                resB = pd.DataFrame(dict(zip(colNames, valsB)), index=[0])
                results = results.append(resB)
                
    # fig = plt.figure()
    
    # axDTrav = fig.add_subplot(1, 3, 1)
    # lLI = axDTrav.plot(fracObs, dmetrics["dTrav"]["LI"], color="royalblue")
    # lBA = axDTrav.plot(fracObs, dmetrics["dTrav"]["BA"], color="deepskyblue")
    # lMJ = axDTrav.plot(fracObs, dmetrics["dTrav"]["MJ"], color="red")
    # axDTrav.set_title("dist trav difference")

    # axROfG = fig.add_subplot(1, 3, 2)
    # axROfG.plot(fracObs, dmetrics["rOfG"]["LI"], color="royalblue")
    # axROfG.plot(fracObs, dmetrics["rOfG"]["BA"], color="deepskyblue")
    # axROfG.plot(fracObs, dmetrics["rOfG"]["MJ"], color="red")
    # axROfG.set_title("r of g difference")

    # axHist = fig.add_subplot(1, 3, 3)
    # axHist.plot(fracObs, dmetrics["hist"]["LI"], color="royalblue") 
    # axHist.plot(fracObs, dmetrics["hist"]["BA"], color="deepskyblue") 
    # axHist.plot(fracObs, dmetrics["hist"]["MJ"], color="red")
    # axHist.set_title("histogram difference")

    # lineLabels = ["lin. int.", "Barnett", "ours"]
    # fig.legend([lLI, lBA, lMJ], labels = lineLabels, loc="upper center", ncol=3)
    
    # plt.show()
    # plt.tight_layout()
    pdb.set_trace()
