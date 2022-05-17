import pdb
import matplotlib.pyplot as plt
from collections import Counter
# import scipy.stats as stats
import numpy as np

np.random.seed(1996)
Ndays = 90
NMinOn = 1
NMinOff = 1

# 0 - flight, 1 - pause
p_00 = 0.99
p_11 = 0.99
p_01 = 1 - p_00
p_10 = 1 - p_11
probs = np.array([[p_00, p_01], [p_10, p_11]])


def truncate(segment):

    ind0 = np.where(segment == 0)[0]
    if len(ind0) < 2:
        return np.array([])
    else:
        return segment[ind0[0]:(ind0[-1]+1)]


def P(n):
    """
    Transition probabilities after n steps
    """
    const1 = 2 - p_00 - p_11
    const2 = (p_00 + p_11 - 1) ** n
    A1 = np.array([[1 - p_11, 1 - p_00], [1 - p_11, 1 - p_00]])
    A2 = np.array([[1 - p_00, -(1 - p_00)], [-(1 - p_11), 1 - p_11]])
    return np.matrix((A1 + const2 * A2) / const1)


def pi(K, l):
    """
    Calculates the probability of a break of length l
    """
    stationary_0 = p_10 / (1 + p_10 - p_00)
    total = np.zeros(K - l - 1)
    for k in range(1, K - l - 1):
        total[k] = stationary_0 * p_01 * (p_11**(l-1)) * p_10
    return total.sum()


def simulateChain(probs, K):
    X = np.zeros(K, dtype="int")
    for k in range(1, K - 1):
        u = np.random.uniform()
        p = probs[X[k - 1], X[k - 1]]
        if u <= p:
            X[k] = X[k - 1]
        else:
            X[k] = 1 - X[k - 1]
    return X


def calculateCounts(X, maxLen):

    counts = np.zeros(maxLen)
    splitPoints = np.where(np.abs(np.diff(X)))[0]+1
    seqsOfOnes = np.array_split(X, splitPoints)
    lengths = [sum(s) for s in seqsOfOnes]
    nZLengths = filter(lambda _s: _s > 0, lengths)
    counter = Counter(nZLengths)
    counts[list(counter.keys())] = list(counter.values())
    # if sum(counts) == 0:
    #    counts[0] = 1
    return(counts)


if __name__ == "__main__":

    Nchain = int(3600 * 24 * Ndays)  # seconds seconds over several days
    maxSeqLen = int(NMinOn * 60)

    X = simulateChain(probs, Nchain)
    countsT = calculateCounts(X, 10000)
    trueEmpDist = countsT / countsT.sum()

    # which indices to mask
    NMinCycle = NMinOn + NMinOff
    NSecCycle = NMinCycle * 60
    inds = np.arange(Nchain)
    indMiss = np.where(inds % NSecCycle >= maxSeqLen)[0]
    Xobs = np.copy(X)
    Xobs[indMiss] = -10

    splitPoints = np.where(np.abs(np.diff(Xobs)) > 5)[0] + 1
    segmentList = np.array_split(Xobs, splitPoints)
    obsList = segmentList[::2]

    truncSegs = [truncate(segment) for segment in obsList]
    truncSegs = filter(lambda _s: len(_s) > 2, truncSegs)
    truncSegsList = list(truncSegs)

    # calculate the empirical distirbution using the
    # truncated data
    empDist = np.zeros(maxSeqLen)
    counts = [calculateCounts(seg, maxSeqLen) for seg in truncSegsList]
    empDist = sum(counts)
    Nsegments = int(empDist.sum())
    empDist /= Nsegments
    p_11_hat = 1 - sum(counts).sum() / np.dot(sum(counts), np.arange(maxSeqLen))

    theoretical = np.zeros(maxSeqLen)
    for l in range(1, maxSeqLen):
        theoretical[l] = (maxSeqLen - l - 1) * (p_11 ** l)
    #    theoretical[l] += pi(maxSeqLen, l)
    theoretical = theoretical/theoretical.sum()
    
    
    # output the results
    print(f"MLE for p_11 = {p_11_hat:.6f}; true p_11 = {p_11}")
    print(f"N segments = {Nsegments}")
    print(f"true N segments = {int(countsT.sum())}")


    lengths = np.arange(1, maxSeqLen)
    ax = plt.axes()
    l1, = ax.plot(lengths, theoretical[1:maxSeqLen])
    l2, = ax.plot(lengths, empDist[1:maxSeqLen])
    l3, = ax.plot(lengths, trueEmpDist[1:maxSeqLen])
    title = f"p_00 = {p_00}, p_11 = {p_11}, Nchain = {Ndays} days, " \
            f"NMinOn = {NMinOn}, NMinOff = {NMinOff}, " \
            f"p_11_hat = {p_11_hat:.6}, # obs pauses  = {Nsegments}/{int(countsT.sum())}"
    ax.set_title(title)
    ax.legend([l1, l2, l3],
              ['theoretical when observed with breaks',
               'empirical when observed with breaks',
               'empirical if the entire chain observed'])
    plt.show()
    #plt.savefig("picture.png")
