import matplotlib.pyplot as plt
import pdb
import numpy as np
from sampling import sampleData
# from imputation import fillGaps
# from likelihood import getMLEs
from plotting import TrajGraph, getLimits
from tools import findPauses, findFlights, getPattern, findChunks
import scipy.stats as stats


class Increment(object):

    def __init__(self, dx, dy, dt):

        self.dx = dx
        self.dy = dy
        self.dt = dt


class Location(object):

    def __init__(self, x, y, t):

        self.x = x
        self.y = y
        self.t = t

    def __add__(self, increment):

        x = self.x + increment.dx
        y = self.y + increment.dy
        t = self.t + increment.dt
        return(Location(x, y, t))

    def spatial(self):

        return(np.hstack([self.x, self.y]))

    def __repr__(self):

        return f"Location(x: {self.x}, y: {self.y}, t: {self.t})"

    

def getPProb(loc, pauses, pings):

    chunks = findChunks(pings, times=True)

    prevChunksLengths = 0
    indFPs = []
    weights = []
    for chunk in chunks:

        pattern = getPattern(chunk)
        indF = np.where(pattern[:-1] == 0)[0]
        # we skip the last element because even if it's
        # a flight we can't know if it's followed by a
        # pause or a flight
        indFP = np.where(pattern[indF + 1] == 1)[0]

        times = chunk[:, -1]
        weights += [stats.t.pdf(loc.t - times[indF], df=2)]
        indFPs += [indFP + prevChunksLengths]
        prevChunksLengths += indF.shape[0]

    weights = np.hstack(weights)
    indFPs = np.hstack(indFPs)
    weights /= np.sum(weights)
    pauseProb = np.sum(weights[indFP])

    return(pauseProb)


def getPauseLength(loc, pauses):
    
    #pStarts = np.where(np.diff(pattern) == 1)[0] + 1
    pStarts = pauses[:, 2].astype("int")
    pLengths = pauses[:, 3].astype("int")

    weights = stats.t.pdf(loc.t - pStarts, df=2)
    weights /= np.sum(weights)

    pauseL = np.random.choice(pLengths, 1, p=weights)[0]

    return(pauseL)


def getFlight(loc, flights):

    flTimes = flights[:, -1]

    diff = np.tile(loc.t, (flTimes.shape[0], 1)) - flights[:, -1]
    dist = np.linalg.norm(diff, ord=2, axis=1)

    weights = stats.t.pdf(dist, df=2)
    weights /= np.sum(weights)

    assert sum(weights) > 0
    
    NFlights = flights.shape[0]
    flightInd = np.random.choice(np.arange(NFlights), 1, p=weights)[0]
    flight = Increment(flights[flightInd, 2], flights[flightInd, 3], 1)
    return(flight)


# (x0, y0, t0) are the coordinates of the last observed location
# (x1, y1, t1) are the coordinates of the first location
# after the gap
def gapFillBarnett(start, stop, pings):

    pauses = findPauses(pings, inclTimes = True)
    flights = findFlights(pings)
    pattern = getPattern(pings)
    
    canPause = True
    trajectory = start.spatial()
    loc = start
    while loc.t < stop.t:
        pauseP = getPProb(loc, pauses, pings)
        if canPause and (np.random.random_sample(1) < pauseP):
            pauseL = getPauseLength(loc, pauses)
            #pauseL = getPauseLength(loc, pauses, pattern)
            for i in range(pauseL):
                trajectory = np.vstack((trajectory, loc.spatial()))
            canPause = False
            loc.t = loc.t + pauseL
        else:
            inc = getFlight(loc, flights)
            loc = loc + inc
            locArr = loc.spatial()
            trajectory = np.vstack((trajectory, locArr))
            canPause = True
        
            
    trajectory = np.vstack((trajectory, stop.spatial()))
    sampPat = getPattern(trajectory)
    sampPat = np.append(0, sampPat)

    Nflights = int(len(sampPat) - np.sum(sampPat))
    weights = np.linspace(0, 1, num=Nflights)

    inds = np.where(sampPat == 0)[0]
    flights = trajectory[inds, :]

    end = stop.spatial()

    term1 = flights * (1 - weights[:, np.newaxis])
    term2 = np.tile(end, (len(weights), 1)) * weights[:, np.newaxis]

    trajectory[inds, :] = term1 + term2
    for ind in np.where(sampPat == 1)[0]:
        trajectory[ind, :] = trajectory[ind - 1, :]

    return trajectory


def fillGapsBarnett(pings, consolidate=False, *args):

    nPings = pings.shape[0]
    indsMis = np.where(np.isnan(pings[:, 0]))[0]
    if indsMis.shape[0] == 0:
        return[]

    dpat = np.diff(indsMis)
    dinds = np.where(dpat > 1)[0]
    gapStarts = np.append(indsMis[0], indsMis[dinds + 1])
    gapEnds = np.append(indsMis[dinds], indsMis[-1])

    nGaps = len(gapStarts)
    imputed = []  # np.zeros((np.sum(gapEnds - gapStarts), 2))

    for gapNo in range(nGaps):
        gStart = gapStarts[gapNo]
        gStop = gapEnds[gapNo]
        start = Location(pings[gStart - 1, 0], pings[gStart - 1, 1], gStart)
        stop = Location(pings[gStop + 1, 0], pings[gStop + 1, 1], gStop)
        imputed += [gapFillBarnett(start, stop, pings)]

    if consolidate:
        pingList = []  # pings[:gapStarts[0], :]]
        for gapNo in range(nGaps):
            gStart = gapEnds[gapNo - 1] + 1 if gapNo > 0 else 0
            gStop = gapStarts[gapNo]
            insert = pings[gStart:gStop, :]
            assert np.all(~np.isnan(insert))
            pingList.append(insert)
            pingList.append(imputed[gapNo])
        if gapEnds[-1] < nPings:
            gStart = (gapEnds[-1] + 1)
            pingList.append(pings[gStart:, :])
        imputed = np.vstack(pingList)
    return(imputed)


# if __name__ == "__main__":

#     np.random.seed(741996)
#     N = 1000
#     LAMBDA = 500
#     FRAC_OBS = 0.15
#     SD = 1
#     C = 0.9
#     PROB_P = 0.01
#     PROB_F = 0.1

#     truePings, pings = sampleData(N, LAMBDA, FRAC_OBS, SD, C, PROB_P, PROB_F)
#     # gapStart = 20
#     # gapStop = 30

#     # start = Location(pings[gapStart, 0], pings[gapStart, 1], gapStart)
#     # stop = Location(pings[gapStop + 1, 0], pings[gapStop + 1, 1], gapStop)
#     # pings[gapStart:gapStop, :] = np.nan

#     # gapFill = gapFillBarnett(start, stop, pings)
#     gapFillB = fillGapsBarnett(pings)
#     params = getMLEs(pings, verbose=True)
#     gapFillJ = fillGapsRheeData(pings, "MJ", params)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.set_xlim(getLimits([truePings] + gapFillJ + gapFillB, "x"))
#     ax.set_ylim(getLimits([truePings] + gapFillB, "y"))

#     t = TrajGraph(truePings, linewidth=0.5)
#     ax.add_collection(t)

#     for gap in gapFillB:
#         t = TrajGraph(gap, colors="blue", linewidth=0.5)
#         ax.add_collection(t)

#     for gap in gapFillJ:
#         t = TrajGraph(gap, colors="red", linewidth=0.5)
#         ax.add_collection(t)

#     plt.show()
