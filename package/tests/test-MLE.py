# This script is supposed to show how parameter estimates might be biased
# if incorrect data collection scheme is used.
import sys
sys.path.append("..")
from pyMobility.likelihood import getMLEsUnbiased
from pyMobility.sampling import sampleData
import numpy as np
import matplotlib.pyplot as plt
from pyMobility.tools import getIncrements
import pdb

np.random.seed(1996)
N = 2000000
U = 100 # the length of the sequence of observations
SDEV = 0.6
C = 0.95
# probability of a pause after a flight
PROB_P = 0.01 # avg length of seuqence of flights is 1/PROB_P
# probability of a flight after a 1 second of a pause
PROB_F = 0.01 # avg. pause duration is 1/PROB_F
OBS_FRAC = 0.5
NITER = 10

if __name__ == "__main__":

    fig = plt.figure(figsize = (16, 6), tight_layout = True)

    MLEs = np.zeros((NITER, 4))
    for iterNo in range(NITER):

        truePings, pings = sampleData(N, U, OBS_FRAC, SDEV, C, PROB_P,
                                      PROB_F, missingPatDist = "none",
                                      inclTimes = True)

        #print(f"Perc. obs.: {100*pings.shape[0]/N}")
        M = getIncrements(pings)
        if pings.shape[0] == N:
            U = N
        p = getMLEsUnbiased(pings, U)
        MLEs[iterNo, :] = p
        print(f"C: {p[0]:0.4}, SD: {p[1]:0.4}, Pp: {p[2]:0.4}, Pf: {p[3]:0.4}")

    print("means")
    print(MLEs.mean(axis=0))
    print("std. dev")
    print(MLEs.std(axis=0))
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax1.hist(MLEs[:, 0], color="orange", bins=31)
    ax1.axvline(x=C, color="black", linestyle="dashed")
    ax1.set_xlim([0.5, 1])
    ax1.set_ylim([0, NITER/2])
    ax1.set_title("Corr Coef.")

    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax2.hist(MLEs[:, 1], color="orange", bins=31)
    ax2.axvline(x=SDEV, color="black", linestyle="dashed")
    ax2.set_ylim([0, NITER/2])
    ax2.set_title("Std. dev.")

    bins = np.linspace(0, 0.5, 31)
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.hist(MLEs[:, 2], color="orange", bins=bins)
    ax3.axvline(x=PROB_P, color="black", linestyle="dashed")
    ax3.set_xlim([0, 0.5])
    ax3.set_ylim([0, NITER/2])
    ax3.set_title("P(flight -> pause)")

    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax4.hist(MLEs[:, 3], color="blue", bins=bins)
    ax4.axvline(x=PROB_F, color="black", linestyle="dashed")
    ax4.set_xlim([0, 0.5])
    ax4.set_ylim([0, NITER/2])
    ax4.set_title("P(pause -> flight)")

    ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    vals = np.array(["missing" for i in range(N)])
    obs_inds = pings[:, -1].astype("int")
    vals[obs_inds] = "obsrvd."
    ax5.plot(np.arange(N), vals, 'o', color="black", markersize=0.3)
    ax5.set_yticks(["missing", "obsrvd."])

    plt.show()
