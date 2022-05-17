import os
import pdb
import numpy as np
import dill


# scoring parameters
class Params:
    DIR = "simulation-results"
    SEEDS = ['1988', '1996']
    FILES = ["results-seed"]


methods = ["LI", "MJ", "B"]

maxM = 0
for seed in Params.SEEDS:

    # load the data
    tData = []
    fname = 'new-results-seed-' + seed + '.pickle'
    fpath = os.path.join(Params.DIR, fname)
    with open(fpath, 'rb') as ipFile:
        results = dill.load(ipFile)
        tData += list(results["trajectories"].values())
        Conf = results["settings"]

    nSims = len(tData)
    simKeys = list(tData[0].keys())
    pMis = list(filter(lambda _k: isinstance(_k, float) and _k >= 0.1, simKeys))

    for sim in range(nSims):
        m = np.mean(tData[sim]["truth"], axis=0)
        tData[sim]["truth"] -= m
        maxM = max(max(abs(m)), maxM)

    # calculate scores
    for sim in range(nSims):

        truePings = tData[sim]["truth"]
        nPings = truePings.shape[0]

        for perc in pMis:

            for rep in range(Conf.NREP):

                for method in methods:

                    filled = tData[sim][perc][rep][method]
                    m = np.mean(filled, axis=0)
                    tData[sim][perc][rep][method] -= m
                    maxM = max(max(abs(m)), maxM)

    fName = 'new-' + Params.FILES[0] + "-" + seed + '.pickle'
    newFpath = os.path.join(Params.DIR, fName)

    trajectories = dict(zip(np.arange(nSims), tData))
    newData = {"trajectories": trajectories, "settings": results['settings']}
    
    #with open(newFpath, "wb") as opFile:
    #    dill.dump(newData, opFile)

print(maxM)
