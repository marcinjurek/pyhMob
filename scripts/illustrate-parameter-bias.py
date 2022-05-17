# This script is supposed to show how parameter estimates might be biased
# if incorrect data collection scheme is used.
from likelihood import getMLEsUnbiased
from sampling import sampleData
import numpy as np
import matplotlib.pyplot as plt
from tools import getIncrements
import pdb

np.random.seed(1988)
N = 2000000
LAMBDA = 0.05  # avg. length of sequence of observations
# is 1/LAMBDA when using a poisson or geometric distribution
U = 50 # the length of the sequence of observations
# if this length is not random
# FRAC_OBS = 1.0
SDEV = 1.0
C = 0.95
# probability of a pause after a flight
PROB_P = 0.0001 # avg length of seuqence of flights is 1/PROB_P
# probability of a flight after a 1 second of a pause
PROB_F = 0.05 # avg. pause duration is 1/PROB_F


if __name__ == "__main__":

    maxIter = 1
    obs_fracs = [0.5] # [0.3, 0.7, 1.0] 
    fig = plt.figure(figsize = (8, 10), tight_layout = True)

    for idx, frac in enumerate(obs_fracs):

        MLEs = np.zeros((maxIter, 4))
        for iterNo in range(maxIter):

            truePings, pings = sampleData(N, U, frac, SDEV, C, PROB_P,
                                          PROB_F, missingPatDist = "none",
                                          inclTimes = True)
            # print(f"Perc. obs.: {100*pings.shape[0]/N}")
            M = getIncrements(pings)
            Mt = getIncrements(truePings)

            if pings.shape[0] == N:
                U = N
            p = getMLEsUnbiased(pings, U)
            MLEs[iterNo, :] = getMLEsUnbiased(pings, U)
            print(f"C: {p[0]:0.4}, SD: {p[1]:0.4}, Pp: {p[2]:0.4}, Pf: {p[3]:0.4}")

        bins = np.linspace(0.05, 0.15, 31)
        ax1 = plt.subplot2grid((2*len(obs_fracs), 2), (2*idx, 0))
        ax1.hist(MLEs[:, 2], color="orange", bins=bins)
        ax1.axvline(x=PROB_P, color="black", linestyle="dashed")
        ax1.set_xlim([0, 0.1])
        ax1.set_ylim([0, maxIter/2])
        ax1.set_title("P(flight -> pause)")

        ax2 = plt.subplot2grid((2*len(obs_fracs), 2), (2*idx, 1))
        ax2.hist(MLEs[:, 3], color="blue", bins=bins)
        ax2.axvline(x=PROB_F, color="black", linestyle="dashed")
        ax2.set_xlim([0, 0.1])
        ax2.set_ylim([0, maxIter/2])
        ax2.set_title("P(pause -> flight)")
        
        ax3 = plt.subplot2grid((2*len(obs_fracs), 2), (2*idx+1, 0), colspan=2)
        vals = np.array(["missing" for i in range(N)])
        obs_inds = pings[:, -1].astype("int")
        vals[obs_inds] = "obsrvd."

        ax3.plot(np.arange(N), vals, 'o', color="black", markersize=0.3)
        ax3.set_yticks(["missing", "obsrvd."])
        # ax3.set_ylim([-1.1, 1.1])

    plt.show()
