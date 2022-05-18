import numpy as np
import pdb
import scipy.stats as stats
from .tools import cart2polar, polar2cart
# from pyMobility.tools import cart2polar, polar2cart
# from ar1imputation import imputeAR1


def patternIsIncorrect(pattern):

    if np.any(np.isnan(pattern)):
        return 1
    if not pattern[1] == pattern[0]:
        return 2
    if not pattern[0] == 1:
        return 3

    n = len(pattern)

    if pattern[n - 1] == 0 or pattern[n - 2] == 0:
        return 4

    # each gap/non gap has have length at least 2
    d = pattern[:(n - 1)] - pattern[1:]
    if np.sum(np.abs(d[np.flatnonzero(d != 0) + 1])) > 0:
        return 5

    return False


def sampleMissing(n, paramMiss, paramObs, dist="geometric"):
    """
    generates the pattern of missing data by sampling
    the length of observed an unobserved intervals
    """
    pattern = np.nan*np.zeros(n)
    pattern[:2] = 1

    if dist == "geometric":
        rvs = lambda theta: stats.geom.rvs(theta)
    if dist == "poisson":
        rvs = lambda theta: stats.geom.poisson(theta)
    if dist == "none":
        rvs = lambda theta: round(theta)
        
    params = [paramMiss, paramObs]
    while patternIsIncorrect(pattern):
        i = 2
        flag = 1 # 1 if observed, 0 otherwise
        while i < (n - 2):
            p = params[flag]            
            nextn = min(n, i + rvs(p))
            pattern[i:nextn] = flag
            i = nextn
            flag = 1 - flag
        pattern[-2:] = 1
    return(pattern)


def sampleAnglesLengths(n, sdev, c, mu0=0):

    draws = np.empty([n, 2])
    draws[0, :] = sdev * stats.norm.rvs(size=2) + c * mu0
    for i in range(1, n):
        draws[i, :] = stats.norm.rvs(size=2) * sdev + c * draws[i - 1, :]

    drawsPolar = cart2polar(draws)
    return(drawsPolar, draws)


def samplePauses(nSteps, pre_increment, pPause, pFlight):
    """
    Samples the sequence of states using a two state markov chain.
    The transition matrix in the model is:

    [[1 - pPause, pPause], 
     [pFlight, 1 - pFlight]]

    Parameters
    -------
    nSteps : int
         the number of steps of the chain
    pPause : float 
         the probability of a pause following a flight
    pFlight : float
         the probability of a flight following a pause

    Returns
    -------
    sequence : np.ndarray
        an array of size 1 x nSteps with indices of the states
        that are flights.
    """
    # 0 = flight, 1 = pause
    sequence = np.zeros(nSteps, dtype="int")
    isPause = bool(sum(abs(pre_increment)))
    for n in range(1, nSteps):
        u = np.random.rand(1)[0]
        if isPause:
            if u <= pFlight:
                sequence[n] = sequence[n - 1] + 1
                isPause = False
            else:
                sequence[n] = sequence[n - 1]
        else:
            if u <= pPause:
                sequence[n] = sequence[n - 1]
                isPause = True
            else:
                sequence[n] = sequence[n - 1] + 1

    return(sequence)


def sampleTrajectory(preStart, start, nSteps, sdev, c, pPause, pFlight):

    mu0 = start - preStart
    pauses = samplePauses(nSteps, mu0, pPause, pFlight)

    n = int(pauses.max()) + 1
    drawsPolar, drawsRaw = sampleAnglesLengths(n, sdev, c, mu0)
    locs = np.cumsum(polar2cart(drawsPolar), axis=0)
    locs = locs[pauses, :]

    if sum(abs(mu0)) > 0:
        pdb.set_trace()

    trajectory = locs + start
    trajectory = np.vstack([start, trajectory])

    return(trajectory, tuple([drawsPolar, drawsRaw]))


# simulate angles and directions
def sampleData(N, lamb, fracObs, sdev, c, pP, pF, missingPatDist = "none", inclTimes = False):

    if fracObs == 1:
        pattern = np.ones(N)
    else:
        # paramMiss = 1 / ((1 / lamb - 1) / fracObs + 1)
        paramMiss = (1 / fracObs - 1) * lamb
        pattern = sampleMissing(N, paramMiss, lamb, missingPatDist)
    start = np.array([0, 0])
    truePings, draws = sampleTrajectory(start, start, N - 1, sdev, c, pP, pF)

    pings = np.copy(truePings)
    if inclTimes:
        times = np.arange(N).reshape((N, 1))
        truePings = np.hstack((truePings, times))
        pings = np.hstack((pings, times))
        pings = pings[pattern == 1, :]
    else:
        pings[pattern == 0, :] = np.nan
        #pings = np.hstack((pings, times))
    
    return (truePings, pings)
