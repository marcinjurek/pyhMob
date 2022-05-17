import sys
import os
import pdb
import numpy as np
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import mobilityMetrics as mm
import pandas as pd
from plotting import getAxesWithTrajectory

    
# scoring parameters
class Params:
    MISS_LWD = 0.3
    NORMAL_LWD = 1
    SMALL_SIZE = 10
    mpl.rc('font', size=SMALL_SIZE)
    mpl.rc('axes', titlesize=SMALL_SIZE)
    CMAP = plt.get_cmap("Set1")
    DIR = "simulation-results"
    FILES = ["results-single-curve.pickle"]
    PLOT_PMIS = 0.7
    SCORE_NAMES = ["hotspotProb"]
    SPOT_SD = 300


simName = sys.argv[1]
simDir = os.path.join(Params.DIR, f"results-{simName}")
sims = os.listdir(simDir)
nSims = len(sims)
simKeys = []
pMis = []


simNo = 0
print(f"Working on simulation {simNo}")
fname = f"{simNo}.pickle"
with open(os.path.join(simDir, fname), 'rb') as ipFile:
    results = dill.load(ipFile)
tData = results["trajectories"]
Conf = results["settings"]
simKeys = list(results["trajectories"].keys())
methods = list(tData[simKeys[1]][0].keys())
pMis = list(filter(lambda _k: isinstance(_k, float) and _k >= 0.1, simKeys))    
percMis = pMis[(np.abs(np.array(pMis) - Params.PLOT_PMIS)).argmin()]

# generate hotspots:
truePings = tData["truth"]
obs = tData[percMis]["obs"]    


# plot sample tData
fig = plt.figure(figsize = (20, 15))
ax = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)

for idx, method in enumerate(methods):
    pings = tData[percMis][0][method]
    ax, t = getAxesWithTrajectory(ax, pings, "grey",
                                  Params.CMAP(idx), Params.NORMAL_LWD)

ax, t = getAxesWithTrajectory(ax, tData["truth"], "grey", "black", Params.MISS_LWD)
ax, t = getAxesWithTrajectory(ax, obs, "grey", "black", Params.NORMAL_LWD)
ax.set_aspect('equal', adjustable='box')

ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
pattern = np.array(["observed" for i in range(Conf.N)])
indsNan = np.where(np.isnan(obs[:, 0]))[0]
pattern[indsNan] = "missing"
ax2.plot(np.arange(Conf.N), pattern, 'o', color="black", markersize=0.3)
ax2.set_yticks(["missing", "observed"])


plt.show()
plt.tight_layout()
#plt.savefig(f"sample-trajectory-radius-{radius:03d}.pdf")
