import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.stats as stats
# import scipy as sp
from plotting import TrajGraph, getLimits
from prototypes.arima110imputation import smooth
from sampling import samplePauses, sampleData
from likelihood import getMLEs
from barnett import fillGapsBarnett
from imputation import fillGaps


def fillGap(preStart, start, stop, postStop, nSteps, sdev, c, pPause, pFlight):

    # start is the start of the gap, i.e. the last available observation.
    # stop is the end of the gap, i.e. the first available observation
    # after the gap ends. preStart is an observation preceding start and
    # postStop is the one following stop. nSteps is the number of steps
    # for which observations are missing between start and stop.

    pre_increment = start - preStart
    # post_increment = postStop - stop

    pauses = samplePauses(nSteps + 2, pre_increment, pPause, pFlight)

    if pauses[-2] == pauses[-1]:
        pauses[-1] += 1

    n = int(pauses.max()) + 1
    ndim = 2

    A = np.matrix([[1, 1], [0, c]])
    H = np.matrix([[1, 0]])
    Q = (sdev ** 2) * np.matrix([[1, c], [c, c ** 2]])
    R = 0

    trajectory = np.zeros((nSteps + 2, ndim))
    # we can do it like that because increments are uncorrelated
    for d in range(ndim):

        mu0 = np.matrix([[start[d]], [start[d] - preStart[d]]])
        Sig0 = np.zeros((2, 2))
        obs = np.nan * np.zeros(n)
        obs[-2:] = np.array([stop[d], postStop[d]])

        nu_hat = stats.norm.rvs(0, scale=sdev, size=n)
        eps_hat = np.zeros(n)
        for t in range(1, n):
            eps_hat[t] = c * eps_hat[t - 1] + nu_hat[t]

        x_hat = np.cumsum(eps_hat)
        y_hat = np.nan * eps_hat
        y_hat[-2:] = x_hat[-2:]

        y_hat_star = obs - y_hat

        muS, sigS = smooth(A, H, Q, R, mu0, Sig0, y_hat_star)
        mu = np.array([m[0, 0] for m in muS])

        x_samp = x_hat + mu
        x_samp = x_samp[pauses[:-1]]
        trajectory[:, d] = np.hstack((start[d], x_samp))
        # trajectory[:, d] = np.hstack((start[d], x_samp[:-1]))

    if sum(abs(trajectory[0, :] - start)) > 1e-10:
        raise ValueError("Imputed start does not match the true start")
    if sum(abs(trajectory[-1, :] - stop)) > 1e-10:
        pdb.set_trace()
        raise ValueError("Imputed end does not match the true end")

    return(trajectory)


# def fillGaps(pings, method, theta, stack=False, consolidate=False):

#     N = pings.shape[0]
#     pattern = np.ones(N)
#     pattern[np.isnan(pings[:, 0])] = 0
#     d = pattern[:-1] - pattern[1:]
#     gapStarts = np.flatnonzero(d == 1) + 1
#     nGaps = len(gapStarts)
#     gapEnds = np.flatnonzero(d == -1) + 1

#     fills = [None] * nGaps
#     for gapNo in range(nGaps):
#         gapStart = gapStarts[gapNo]
#         gapEnd = gapEnds[gapNo]
#         prev = pings[gapStart - 2, :]
#         last = pings[gapStart - 1, :]
#         first = pings[gapEnd, :]
#         second = pings[gapEnd + 1, :]
#         if method == "MJ":
#             fills[gapNo] = fillGap(prev, last, first, second,
#                                    gapEnd - gapStart, theta[1],
#                                    theta[0], theta[2], theta[3])# [1:-1, :]
#         if method == "LI":
#             fills[gapNo] = linInt(first, last, gapEnd - gapStart)

#     if stack or consolidate:
#         if method == "MJ":
#             fills = [fill[1:-1, :] for fill in fills]
#         fills = np.vstack(fills)
#     if consolidate:
#         filledPings = np.copy(pings)
#         filledPings[np.where(pattern == 0), :] = fills
#         fills = filledPings

#     return(fills)


def linInt(first, last, nSteps):

    LIx = np.linspace(last[0], first[0], nSteps)
    LIy = np.linspace(last[1], first[1], nSteps)
    LI = np.vstack((LIx, LIy)).T

    return(LI)


# def linInt(pings):

#     gapStart = np.where(np.isnan(pings[:, 0]))[0][0]
#     gapEnd = np.where(np.isnan(pings[:, 0]))[0][-1] + 1
#     # prev = pings[gapStart - 2, :]
#     last = pings[gapStart - 1, :]
#     first = pings[gapEnd, :]
#     # second = pings[gapEnd + 1, :]

#     LIx = np.linspace(last[0], first[0], gapEnd - gapStart)
#     LIy = np.linspace(last[1], first[1], gapEnd - gapStart)
#     LI = np.vstack((LIx, LIy)).T

#     return(LI)


if __name__ == "__main__":

    np.random.seed(741996)
    N = 1000
    LAMBDA = 200
    FRAC_OBS = 0.5
    SD = 1
    C = 0.99
    PROB_P = 0.01
    PROB_F = 0.1

    truePings, pings = sampleData(N, LAMBDA, FRAC_OBS, SD, C, PROB_P, PROB_F)
    gapFillB = fillGapsBarnett(pings, consolidate=True)
    params = getMLEs(pings, verbose=True)
    gapFillJ = fillGaps(pings, "MJ", params, consolidate=True)
    gapFillLI = fillGaps(pings, "LI", params, consolidate=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    pingList = [truePings, gapFillB, gapFillJ, gapFillLI]

    xlim = getLimits(pingList, "x")
    ylim = getLimits(pingList, "y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    t = TrajGraph(truePings, linewidth=1.0, linestyle="dotted", color = "black")
    ax.add_collection(t)

    t = TrajGraph(gapFillB, colors="royalblue", linewidth=1.0)
    ax.add_collection(t)

    t = TrajGraph(gapFillJ, colors="darkred", linewidth=1.0)
    ax.add_collection(t)

    t = TrajGraph(gapFillLI, colors = "deepskyblue", linewidth=1.0)
    ax.add_collection(t)

    plt.show()
