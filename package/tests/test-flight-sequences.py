import matplotlib.pyplot as plt
import pdb
import numpy as np
import sys
sys.path.append("..")
from pyMobility.tools import getLinkFlightSeqs, getIncrements
from pyMobility.sampling import sampleData


np.random.seed(1995)
N = 2000000
LAMBDA = 200
FO = 0.5
SDEV = 0.6
C = 0.95
pP = 0.01
pF = 0.01

np.set_printoptions(suppress=True)


if __name__ == "__main__":

    truePings, pings = sampleData(N, LAMBDA, FO, SDEV, C, pP, pF, inclTimes=1)

    increments = getIncrements(pings)
    linkFlightSeqsInds = getLinkFlightSeqs(increments)

    fSeqLengths = np.array([len(seq) for seq in linkFlightSeqsInds])
    # lengths, counts = np.unique(fSeqLengths, return_counts=True)
    # counts = counts/counts.sum()
    # order = np.argsort(lengths)
    # plt.plot(lengths[order], counts[order])
    # theoretical = np.zeros(LAMBDA)
    # for l in range(1, LAMBDA):
    #     theoretical[l] = (LAMBDA - l - 1) * ((1-pP) ** l)
    # theoretical = theoretical/theoretical.sum()
    # plt.plot(np.arange(1, LAMBDA), theoretical[1:])
    # plt.show()
    
    def logLikFactory(seqs, maxSeqLength):
        
        def logLik(p):
            Ls = np.arange(1, maxSeqLength)
            theoretical = (maxSeqLength - Ls - 1) * ((1 - p) ** Ls)
            const = sum(theoretical)
            likelihood = np.sum(np.log((maxSeqLength - seqs - 1) * ((1 - p) ** seqs) / const))
            return(likelihood)

        return(logLik)

    logLik = logLikFactory(fSeqLengths, LAMBDA)
    Ngrid = 1000
    thetas = np.linspace(1e-3, 1-1e-3, 1000)
    vals = [logLik(theta) for theta in thetas]

    order = np.argsort(np.abs(vals))
    print(thetas[order[0]])

    pdb.set_trace()
    plt.plot(thetas, vals)
    plt.axvline(x=pP)
    plt.show()
