# imports
import dill
import numpy as np
import pdb
from dataExploration.loadtraces import load_data
import matplotlib
import matplotlib.pyplot as plt

from tools import getMissingPattern, findPauses
from imputation import fillGaps
from likelihood import getMLEs
#from plotting import TrajGraph, findPauses, getLimits

# settings
np.random.seed(1996)
N_SAMP_GAP = 1
NSAMPLE = 1000
MIS_PERC = 0.15  # half of the missing interval
SET = "Statefair"  # KAIST, Orlando, NewYork, NCSU, Statefair


class Conf:
    NGRID = 21
    NREP = 50
    SEED = 1962
    MISS_TYPE = ["contiguous", "on-off"]
    ON = 100
    OFF = 100
    LWD = 0.8
    SMALL_SIZE = 8
    FILENAME = "real-data.pickle"

    
matplotlib.rc('font', size=Conf.SMALL_SIZE)
matplotlib.rc('axes', titlesize=Conf.SMALL_SIZE)
methods = ["LI", "MJ", "B"]

# main script
if __name__ == "__main__":

    np.random.seed(Conf.SEED)
    fracMis = np.linspace(0.1, 0.9, num=Conf.NGRID)
    
    data = load_data(SET)
    #inds = np.random.choice(np.arange(len(data)), size=NSAMPLE, replace=False)
    #data = [data[i]/1000 for i in inds]

    trajectories = []
    for idx, truePings in enumerate(data):

        trajectories += [dict()]
        print(f"===== set {idx + 1} =====")
        nPings = truePings.shape[0]
        print(f"Total number of pings is {nPings}")
        truePings = np.copy(truePings)
        pT = getMLEs(truePings, verbose=False)
        
        pauses = findPauses(truePings, inclTimes = True)
        print(f"Param. estimates using full data")
        print(f"C: {pT[0]:0.4}, SD: {pT[1]:0.4}, Pp: {pT[2]:0.4}, Pf: {pT[3]:0.4}")
        
        trajectories[idx] = {"truth": truePings}

        for perc in fracMis:

            print(f"\tMissing percentage: {100*perc:0.4}")
            # hide observations
            misParms = {"perc" : perc, "on": Conf.ON, "off" : Conf.OFF}
            misInds = getMissingPattern(nPings, missType=Conf.MISS_TYPE,
                                        params=misParms)
            pings = np.copy(truePings)
            pings[misInds, :] = np.nan
            obsPings = np.copy(pings)
            trajectories[idx][perc] = {"obs": obsPings}
            p = getMLEs(pings, verbose=False)

            print(f"\tC: {p[0]:0.4}, SD: {p[1]:0.4}, Pp: {p[2]:0.4}, Pf: {p[3]:0.4}")
            for repNo in range(Conf.NREP):

                trajectories[idx][perc][repNo] = dict()

                for method in methods:
                    if method == "MJ":
                        filled = fillGaps(pings, method, params=p, consolidate=True)
                    else:
                        filled = fillGaps(pings, method, consolidate=True)
                    trajectories[idx][perc][repNo][method] = filled

        toStore = {"trajectories": trajectories, "settings": Conf}

        with open(Conf.FILENAME, 'wb') as opFile:
            dill.dump(toStore, opFile)
