import pdb
import numpy as np
import matplotlib.pyplot as plt
from plotting import TrajGraph, findPauses
from scipy.spatial import distance as dist
# in all functions below pings is assumed to be an
# N by 2 numpy array


def distanceTravelled(pings):

    def norm2(_u):
        return np.sqrt(_u[0]**2 + _u[1]**2)

    increments = np.diff(pings, axis = 0)
    norms = np.apply_along_axis(norm2, 1, increments)
    dTrav = np.sum(norms)
    return(dTrav)


def radiusOfGyration(pings):

    centralLocation = np.nansum(pings, axis=0) / pings.shape[0]
    centeredPings = pings - centralLocation
    norms = np.linalg.norm(centeredPings, ord=2, axis=1)
    normsSq = norms ** 2
    RoG = np.sqrt(np.nanmean(normsSq))
    return RoG


def hotspotProb(pings, spot, radius):

    distFromTheSpot = np.linalg.norm(pings - spot, axis = 1)
    wasSpotVisited = min(distFromTheSpot) < radius
    return(int(wasSpotVisited))


def chi2(hist1, hist2):

    return (hist1 - hist2)**2 / hist2


def histRMSE(hist1, hist2):

    diff = hist1 - hist2
    MSD = np.mean(diff ** 2)
    return(np.sqrt(MSD))


def KLDiv(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * (np.log(p) - np.log(q)), 0))


# calculates the KL divergence of hist1
# with respect to hist2, i.e.
def JensenShannon(hist1, hist2):

    v1 = hist1.flatten()
    v2 = hist2.flatten()
    JS = dist.jensenshannon(v1, v2, base = 2)
    return(JS)
   

def maxDiam(pings):

    pauses = findPauses(pings)
    if len(pauses) > 1:
        pauseXY = pauses[:, :2]
        diams = dist.pdist(pauseXY)
        return(np.max(diams))
    else:
        return 0


# if __name__ == "__main__":

#     pings = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0.5],
#                       [0, 1], [0, 1], [0.5, 0.5], [1, 0],
#                       [1, 0], [1, 0], [1, 1]])

#     t = TrajGraph(pings, color="black")
#     fig = plt.figure()

#     ax = fig.add_subplot(111)
#     ax.add_collection(t)

#     pauses = findPauses(pings)
#     nPauses = pauses.shape[0]
#     for pauseId in range(nPauses):
#         ax.scatter(pauses[pauseId, 0], pauses[pauseId, 1],
#                    s=10 * pauses[pauseId, -1], color="black")

#     ax.set_xlim(-0.2, 1.2)
#     ax.set_ylim(-0.2, 1.2)
#     plt.hist2d(pings)
#     plt.show()
#     # print(maxDiam(pings))
