# imports
import sys
import os
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# plot settings
LWD = 0.8
SMALL_SIZE = 10
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)
cmap = plt.get_cmap("Set1")
DIR = "simulation-results"
#DIR = "simulation-results/rhee"
#RADII = [10, 25, 50, 100, 200]
RADII = [50, 100, 200]
simName = sys.argv[1]
# UQ = 0.9
# LQ = 0.1

for radius in RADII:
    
    print(f"Radius {radius}")
    fName = os.path.join(DIR, f"simulation-scores-radius-{radius}-{simName}.csv")
    scores = pd.read_csv(fName, index_col=[0, 1, 2, 3])
    methods = scores.index.get_level_values("method").unique().to_list()
    nMeth = len(methods)
    scoresOff = scores.query("trueTraj == 0").drop(columns="trueTraj")
    scoresOn = scores.query("trueTraj == 1").drop(columns="trueTraj")

    noTruePos = scoresOn.index.get_level_values("sim").unique().shape[0]
    noTrueNeg = scoresOff.index.get_level_values("sim").unique().shape[0]
    print(f"\tTrue in: {noTruePos}")
    print(f"\tTrue out: {noTrueNeg}")

    def names2Codes(names):
        codes = np.zeros(nMeth)
        for methIdx, method in enumerate(methods):
            if method in names:
                codes[methIdx] = 1
        return codes
    
    #find the winner(s) for each trajectory and missing percentage
    winnerOn = scoresOn.groupby(['sim', 'method', 'percMis']).mean() 
    indexDF = winnerOn.index.to_frame(index=False)
    pMis = np.unique(indexDF.drop(columns=['sim', 'method']).to_numpy())
    sims = np.unique(winnerOn.index.get_level_values("sim").to_numpy())
    winnersOn = [np.zeros(nMeth) for p in pMis]

    for pIdx, p in enumerate(pMis):
        for sim in sims:
            locRes = winnerOn.loc[sim, :, p].droplevel(["sim", "percMis"])
            locWinners = locRes.loc[locRes["impTraj"]==locRes["impTraj"].max()].index.tolist()
            if pIdx==11:
                print(locRes["impTraj"])
            codes = names2Codes(locWinners)
            winnersOn[pIdx] += codes
        winnersOn[pIdx] /= len(sims)
    winnersOn = np.array(winnersOn)

    pdb.set_trace()
    
    #find the winner(s) for each trajectory and missing percentage
    winnerOff = scoresOff.groupby(['sim', 'method', 'percMis']).mean() 
    indexDF = winnerOff.index.to_frame(index=False)
    pMis = np.unique(indexDF.drop(columns=['sim', 'method']).to_numpy())
    sims = np.unique(winnerOff.index.get_level_values("sim").to_numpy())
    winnersOff = [np.zeros(nMeth) for p in pMis]

    for pIdx, p in enumerate(pMis):
        for sim in sims:
            locRes = winnerOff.loc[sim, :, p].droplevel(["sim", "percMis"])
            locWinners = locRes.loc[locRes["impTraj"]==locRes["impTraj"].min()].index.tolist()
            codes = names2Codes(locWinners)
            winnersOff[pIdx] += codes
        winnersOff[pIdx] /= len(sims)

    winnersOff = np.array(winnersOff)
    fig = plt.figure()

    axTP = fig.add_subplot(1, 2, 1)
    axTN = fig.add_subplot(1, 2, 2)
    for methIdx, method in enumerate(methods):
    
        axTP.plot(pMis, winnersOn[:, methIdx], color = cmap(methIdx))
        axTN.plot(pMis, winnersOff[:, methIdx], color = cmap(methIdx))
                
    plt.tight_layout()
    plt.show()
    
#     # "off-the-curve" scores
#     grOffCounts = scoresOff.groupby(['method', 'percMis', 'sim'])["impTraj"]
#     prFP = grOffCounts.mean().to_frame()
#     grPrFP = prFP.groupby(['method', 'percMis'])["impTraj"]

#     negP = dict()
#     for method in methods:
#         queryStr = f"method == '{method}'"
#         negP[method] = 1 - grPrFP.mean().to_frame().query(queryStr).to_numpy().flatten()
    
#     # "on-the-curve" scores
#     grOnCounts = scoresOn.groupby(['method', 'percMis', 'sim'])["impTraj"]
#     prTP = grOnCounts.mean().to_frame()
#     grPrTP = prTP.groupby(['method', 'percMis'])["impTraj"]

#     posP = dict()
#     for method in methods:
#         queryStr = f"method == '{method}'"
#         posP[method] = grPrTP.mean().to_frame().query(queryStr).to_numpy().flatten()

#     # plotting
#     percentages = prTP.index.droplevel(['method', 'sim']).unique().to_numpy()

#     fig = plt.figure(figsize=(10, 6))
#     axes = [None] * (2 * (nMeth + 1))
#     axes[2 * nMeth] = fig.add_subplot(2, nMeth + 1, nMeth + 1, sharey=axes[0])
#     axes[2 * nMeth].set_title("means")
#     axes[2 * nMeth + 1] = fig.add_subplot(2, nMeth + 1, 2 * (nMeth + 1), sharey=axes[1])

#     lines = []
#     for idx, method in enumerate(methods):
#         axes[2*idx] = fig.add_subplot(2, nMeth + 1, idx + 1)

#         axes[2*idx].set_title(method)
#         axes[2*idx + 1] = fig.add_subplot(2, nMeth + 1, idx + 2 + nMeth)

#         for p in percentages:
#             queryStr = f"percMis == {p} & method == '{method}'"
#             score = prTP.query(queryStr).to_numpy().flatten()
#             (score, counts) = np.unique(score, return_counts=True)
#             counts = counts / sum(counts)
#             ps = p * np.ones(score.shape[0])
#             axes[2 * idx].scatter(ps, score, color=cmap(idx), s=counts * 15)

#             score = 1 - prFP.query(queryStr).to_numpy().flatten()
#             (score, counts) = np.unique(score, return_counts=True)
#             counts = counts / sum(counts)
#             ps = p * np.ones(score.shape[0])
#             axes[2 * idx + 1].scatter(ps, score, color=cmap(idx), s=counts * 15)

#         axes[2 * nMeth].plot(percentages, posP[method], color = cmap(idx))
#         axes[2 * nMeth + 1].plot(percentages, negP[method], color = cmap(idx))

#     for idx, ax in enumerate(axes):
#         ax.set_ylim(-0.05, 1.05)
#         if idx % 2 == 1:
#             ax.set_xlabel("% missing")
#         if idx == 0:
#             ax.set_ylabel("True positives")
#         if idx == 1:
#             ax.set_ylabel("True negatives")

#     plt.suptitle(f"radius = {radius}, true trajectories in/out: {noTruePos}/{noTrueNeg}")
#     plt.tight_layout()
#     plt.savefig(f"scores-radius-{radius:03d}.pdf")

# os.system(f"pdftk scores-radius* cat output scores-{simName}.pdf")
# os.system(f"mv scores-{simName}.pdf simulation-results")
# os.system("rm scores-radius*")
