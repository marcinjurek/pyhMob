# This script simulates the data, simulates missingness,
# estimates parameters and imputes missing parts of the
# trajectory. It can do so several times and at the end
# it saves the results of all simulations to a file.
from datetime import datetime
import os
import numpy as np
import dill
from pyhMob.tools import getIncrements
from pyhMob.likelihood import getMLEsUnbiased
from pyhMob.imputation import fillGaps
from pyhMob.sampling import sampleData


# simulation settings
class Conf:
    N = 1000
    NGRID = 21
    LAMBDA = 500
    NREP = 22
    NSIMS = 36
    SD = 1.0
    C = 0.95
    PROB_P = 0.1
    PROB_F = 0.1
    SEED = 1996
    MISS_TYPE = ["contiguous", "on-off"]
    ON = 25
    OFF = 25
    RECOMPUTE = False
    METHODS = ["LI", "MJ", "MJu", "B"]
    FRAC_MIS = np.linspace(0.2, 0.8, num=NGRID)
    SIM_NAME = "test"
    RESULTS_DIR = "~/mobility/simulation-results"

    
def getMissingPattern(N, missType, params=None):

    misInds = np.array([])
    if "contiguous" in missType and "perc" in params.keys():
        perc = params["perc"]
        misPat = np.ones(N)
        gapStart = round(N * (0.5 - perc / 2))
        gapStop = round(N * (0.5 + perc / 2))
        misPat[gapStart:gapStop] = 0
        misIndsC = np.where(misPat == 0)[0]
        misInds = np.union1d(misIndsC, misInds)

    if "on-off" in missType and "on" in params.keys() and "off" in params.keys():
        misPat = np.ones(N)
        K = params["on"] + params["off"]
        misPat[np.arange(N) % K >= params["on"]] = 0
        misPat[K * int(N / K):] = 1
        misPat[-2:] = 1
        misIndsOO = np.where(misPat == 0)[0]
        misInds = np.union1d(misIndsOO, misInds)

    misInds = misInds.astype("int")
        
    return(misInds)


def getMaxSeqLength(misParms):

    if "on-off" in Conf.MISS_TYPE:
        return misParms["on"]
    elif "contiguous" in Conf.MISS_TYPE:
        return round(Conf.N * (1-misParms["perc"])/2)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    resultsPath = f"{Conf.RESULTS_DIR}/results-{Conf.SIM_NAME}"
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)
        simNoOffset = 0
    elif os.listdir(resultsPath):
        simNoOffset = max([int(fname.split(".")[0]) for fname in os.listdir(resultsPath)])

    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')} Started simulation {Conf.SIM_NAME}")
    for simNo in range(simNoOffset, Conf.NSIMS + simNoOffset):

        trajectories = dict()
        try:
            np.random.seed(Conf.SEED + simNo)
            now = datetime.now()
            print(f"{now.strftime('%d/%m/%Y %H:%M:%S')} Simulation no: {simNo}")
            truePings, pings = sampleData(Conf.N, Conf.LAMBDA, 1.0, Conf.SD,
                                          Conf.C, Conf.PROB_P, Conf.PROB_F, inclTimes=True)
            means = np.mean(truePings, axis=0)
            truePings -= means
            pings -= means

            p = getMLEsUnbiased(pings, Conf.N - 2)
            print(f"Parameter estimates using all pings")
            print(f"C: {p[0]:0.4}, SD: {p[1]:0.4}, Pp: {p[2]:0.4}, Pf: {p[3]:0.4}")
            trajectories["truth"] = truePings[:, :2]

            for perc in Conf.FRAC_MIS:

                print(f"\tMissing percentage: {100*perc:0.4}")
                # hide observations
                misParms = {"perc" : perc, "on": Conf.ON, "off" : Conf.OFF}
                misInds = getMissingPattern(Conf.N, missType=Conf.MISS_TYPE,
                                            params=misParms)
                obsInds = np.setdiff1d(np.arange(Conf.N), misInds)

                pings[misInds, :] = np.nan
                obsPings = np.copy(pings)
                trajectories[perc] = {"obs": obsPings[:, :2]}

                p = getMLEsUnbiased(pings[obsInds, :], Conf.N - 2)
                pu = getMLEsUnbiased(pings[obsInds, :], getMaxSeqLength(misParms))
                print(f"\tParameter estimates:")
                print(f"\tC: {p[0]:0.4}, SD: {p[1]:0.4}, Pp: {p[2]:0.4}, Pf: {p[3]:0.4}")
                print(f"\tC: {pu[0]:0.4}, SD: {pu[1]:0.4}, Pp: {pu[2]:0.4}, Pf: {pu[3]:0.4}")
                for repNo in range(Conf.NREP):

                    np.random.seed(repNo+1)
                    trajectories[perc][repNo] = dict()

                    for method in Conf.METHODS:
                        param = p if method == "MJ" else pu
                        filled = fillGaps(pings[:, :2], method, params=p, consolidate=True)
                        trajectories[perc][repNo][method] = filled

            toStore = {"trajectories": trajectories,
                       "settings": Conf}

            filename = os.path.join(resultsPath, f"{simNo}.pickle")
            with open(filename, 'wb') as opFile:
                dill.dump(toStore, opFile)

        except Exception as err:
            print(err)
            pass

    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')} Simulations finished successfully!")
