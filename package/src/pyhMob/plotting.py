import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tools import findPauses

# ------ settings ---------
SAMP_LINE_WIDTH = 0.1
TRUE_LINE_WIDTH = 1.0
SAMP_LINE_COLOR = (1, 0, 0, 1)
TRUE_LINE_COLOR = (0, 0, 0, 1)


class TrajGraph(LineCollection):

    def __init__(self, pings, **kwargs):

        if len(pings):
            x = pings[:, 0]
            y = pings[:, 1]
            segments = self._makeSegments(x, y)
        else:
            segments = []

        if "cmap" in kwargs:
            norm = plt.Normalize(0.0, 1.0)
            z = np.linspace(0.0, 1.0, len(x))
            z = np.asarray(z)
            super().__init__(segments, array=z, norm=norm, **kwargs)
        else:
            super().__init__(segments, **kwargs)

    def _makeSegments(self, x, y):

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments


# data is list with np.arrays
# coord is "x" or "y"
# typ is "global" or "local"
def getLimits(data, coord="", subset=np.empty(0)):

    if coord == "":
        M = max([np.nanmax(t) for t in data])
        m = min([np.nanmin(t) for t in data])

    d = int(coord == "y")
    if not len(subset):
        try:
            M = max([np.nanmax(t[:, d]) for t in data])
            m = min([np.nanmin(t[:, d]) for t in data])
        except TypeError:
            pdb.set_trace()
    else:
        try:
            M = max([np.nanmax(t[subset, d]) for t in data])
            m = min([np.nanmin(t[subset, d]) for t in data])
        except ValueError:
            pdb.set_trace()
    return m, M


def getAxesWithTrajectory(ax, pings, pColor, lColor, lwd):

    # tFull = TrajGraph(pings, colors=TRUE_LINE_COLOR,
    #                   linewidth=TRUE_LINE_WIDTH)
    tFull = TrajGraph(pings, colors=lColor, linewidth=lwd)
    ax.add_collection(tFull)
    pauses = findPauses(pings)
    for i in range(pauses.shape[0]):
        ax.scatter(pauses[i, 0], pauses[i, 1], color=pColor,
                   s=4 * pauses[i, 2])
    return ax, tFull


def plotResults(truePings, sampTraj, missing=np.empty(0)):

    # set-up
    nGaps = len(sampTraj)
    if nGaps > 0:
        nSims = len(sampTraj[0])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)

    # set axes limits
    flatSampTraj = [item for sublist in sampTraj for item in sublist]
    xlim = getLimits([truePings] + flatSampTraj, "x")
    ylim = getLimits([truePings] + flatSampTraj, "y")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # plot results
    for gapNo in range(nGaps):
        for simNo in range(nSims):
            pings = sampTraj[gapNo][simNo]
            pdb.set_trace()
            ax = getAxesWithTrajectory(ax, pings, "grey", "red", 1)
            # t = TrajGraph(pings, colors=SAMP_LINE_COLOR,
            #               linewidth=SAMP_LINE_WIDTH)
            # ax.add_collection(t)
            # pauses = findPauses(pings)
            # for i in range(pauses.shape[0]):
            #     ax.scatter(pauses[i, 0], pauses[i, 1], color=SAMP_LINE_COLOR,
            #                s=4 * pauses[i, 2])

    ax = getAxesWithTrajectory(ax, truePings, "grey", TRUE_LINE_COLOR,
                               TRUE_LINE_WIDTH)

    # ax.scatter(truePings[:, 0], truePings[:, 1], color=TRUE_LINE_COLOR, s=1)

    for gapNo in range(nGaps):
        for simNo in range(nSims):
            pings = sampTraj[gapNo][simNo]
            if simNo == (nSims - 1):
                ax.scatter(pings[0, 0], pings[0, 1], c="green", zorder=2,
                           marker="X", s=100)
                ax.scatter(pings[-1, 0], pings[-1, 1], c="blue", zorder=2,
                           marker="X", s=100)

    plt.show()
