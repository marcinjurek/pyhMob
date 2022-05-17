import sys
import os
import pdb
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import mobilityMetrics as mm
import pandas as pd
from plotting import getAxesWithTrajectory, getLimits


# scoring parameters
class Params:
    MISS_LWD = 0.3
    NORMAL_LWD = 1
    SMALL_SIZE = 10
    mpl.rc('font', size=SMALL_SIZE)
    mpl.rc('axes', titlesize=SMALL_SIZE)
    CMAP = plt.get_cmap("Set1")
    DIR = os.path.join(os.getcwd(), "simulation-results")
    RADII = [10, 25, 50, 100, 200]
    HOTSPOT_COL = (1, 0, 0, 0.5)
    PLOT_PMIS = 0.8
    SCORE_NAMES = ["hotspotProb"]
    SPOT_SD = 300
    SIM_PLOT = 1
    REP_PLOT = 0


np.random.seed(1988)    
simName = sys.argv[1]
colNames = ["simNo", "percMis", "imputNo", "imputMet"] + Params.SCORE_NAMES


# load the data
#tData = []
#for filename in :
#    fname = os.path.join(Params.DIR, f"results-{simName}", filename)
#    print(f"Reading file {filename}")
#    with open(fname, 'rb') as ipFile:
#        results = dill.load(ipFile)
#    if isinstance(results["trajectories"], list):
#        tData += results["trajectories"]
#    elif isinstance(results["trajectories"], dict):
#        tData += list(results["trajectories"].values())
#    else:
#        raise TypeError("data saved in an unknown format")
#    Conf = results["settings"]

simDir = os.path.join(Params.DIR, f"results-{simName}")
sims = os.listdir(simDir)
nSims = len(sims)
simKeys = []
pMis = []

# generate hotspots:
M = [0, 0]
m = [0, 0]
for fname in sims:
    with open(os.path.join(simDir, fname), 'rb') as ipFile:
        results = dill.load(ipFile)

    truePings = results["trajectories"]["truth"]
    if not simKeys:
        simKeys = list(results["trajectories"].keys())
    if not pMis:
        pMis = list(filter(lambda _k: isinstance(_k, float) and _k >= 0.1, simKeys))
    for d in (0, 1):
        M[d] = max(M[d], np.nanmax(truePings[:, d]))
        m[d] = min(m[d], np.nanmin(truePings[:, d]))

spot_x = np.random.normal(loc = 0, scale = min(abs(M[0]), abs(m[0])) / 3, size = nSims)
spot_y = np.random.normal(loc = 0, scale = min(abs(M[1]), abs(m[1])) / 3, size = nSims)
spots = np.vstack((spot_x, spot_y)).T

for radius in Params.RADII:
    print(f"Scoring radius {radius}")
    scoresL = []
    indexList = []
    
    # calculate scores
    for simNo, fname in enumerate(sims):

        print(f"Working on simulation {simNo}")
        with open(os.path.join(simDir, fname), 'rb') as ipFile:
            results = dill.load(ipFile)
        tData = results["trajectories"]
        Conf = results["settings"]
        methods = list(tData[simKeys[1]][0].keys())
        #if sum([abs(81 - len(list(tData[p].keys()))) for p in pMis]):
        #    print([len(list(tData[p].keys())) for p in pMis])
        

        truePings = tData["truth"]
        if truePings.shape[1] == 3:
            truePings = truePings[:, :2]
        nPings = truePings.shape[0]

        spot = spots[simNo]
        passedSpotTrue = mm.hotspotProb(truePings, spot, radius)

        
        for perc in pMis:
            for rep in range(Conf.NREP):
                for method in methods:
                    filled = tData[perc][rep][method]
                    passedSpot = mm.hotspotProb(filled, spot, radius)
                    scoresL.append((passedSpotTrue, passedSpot))
                    indexList.append((simNo, perc, rep, method))

        # plot sample tData
        if simNo == Params.SIM_PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(m[0], M[0])
            ax.set_ylim(m[1], M[1])
            percMisToPlot = pMis[(np.abs(np.array(pMis) - Params.PLOT_PMIS)).argmin()]
            obs = tData[percMisToPlot]["obs"]
            
            for idx, method in enumerate(methods):

                pings = tData[percMisToPlot][Params.REP_PLOT][method]
                ax, t = getAxesWithTrajectory(ax, pings, "grey",
                                              Params.CMAP(idx), Params.NORMAL_LWD)
            
            ax, t = getAxesWithTrajectory(ax, truePings, "grey", "black", Params.MISS_LWD)
            ax, t = getAxesWithTrajectory(ax, obs, "grey", "black", Params.NORMAL_LWD)
            spotPatch = plt.Circle(spot, radius, color=Params.HOTSPOT_COL)
            dummyPatch = plt.Circle(spot, 200, color=Params.HOTSPOT_COL, alpha=0)
            ax.add_patch(spotPatch)
            ax.add_patch(dummyPatch)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"radius = {radius}")
            plt.savefig(os.path.join(Params.DIR, f"sample-trajectory-radius-{radius:03d}.pdf"))

    levels = [range(nSims), pMis, range(Conf.NREP), methods]
    levelNames = ["sim", "percMis", "rep", "method"]
    #indx = pd.MultiIndex.from_product(levels, names=levelNames)
    indx = pd.MultiIndex.from_tuples(indexList, names=levelNames)
    scores = pd.DataFrame(scoresL, index=indx, columns=["trueTraj", "impTraj"])

    scoreFileName = os.path.join(Params.DIR, f"simulation-scores-radius-{radius}-{simName}.csv")
    scores.to_csv(scoreFileName)

curDir = os.getcwd()
os.chdir(Params.DIR)
os.system(f"pdftk sample-trajectory-radius* cat output sample-trajectories-{simName}.pdf")
os.system("rm sample-trajectory-radius*")
os.chdir(curDir)
