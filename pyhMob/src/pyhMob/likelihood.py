import numpy as np
from .tools import cart2polar, getIncrements, getLinkFlightSeqs

np.seterr(all='raise')


# Likelihood of a trajectory given the preceding and following observations
def trajLogLik(trajectory, prevObs, secondObs, sdev, c, pFlight, pPause):

    if c == 1:
        raise ValueError("Cannot evaluate likelihood unless c > 0 and c < 1.")

    incremCart = np.diff(np.vstack((prevObs, trajectory, secondObs)), axis=0)
    incremPolar = cart2polar(incremCart)
    nInc = incremPolar.shape[0]

    fullLogLik = loglik(incremPolar, c, sdev**2, pFlight, pPause)
    incremPolar[2:(nInc-2), :] = np.nan

    integLogLik = loglik(incremPolar, c, sdev**2, pFlight, pPause)
    ll = fullLogLik - integLogLik

    return ll


def getMLEsUnbiased(pings, K):

    increments = getIncrements(pings)
    rhoHat = getCorrCoef(increments)
    sig_hat = getSigHat(increments, rhoHat)
    pPause_hat = getFlightSeqLengthUnbiasedGeom(increments, K)
    pFlight_hat = getPauseLengthUnbiasedGeom(increments, K)
        
    return((rhoHat, sig_hat, pPause_hat, pFlight_hat))


def getSigHat(increments, rho):

    linkFlightSeqInds = getLinkFlightSeqs(increments)
    sse = 0
    nInc = 0
    for chunk in linkFlightSeqInds:
        flights = increments[chunk]
        if len(flights) < 3:
            continue
        dx = flights[1:, 0] - flights[:-1, 0]
        dy = flights[1:, 1] - flights[:-1, 1]
        sse += sum((dx[1:] - rho * dx[:-1]) ** 2)
        sse += sum((dy[1:] - rho * dy[:-1]) ** 2)
        nInc += len(dx) - 1
    sig2_hat = sse / (2*nInc)
    return(np.sqrt(sig2_hat))
    

def getCorrCoef(increments):

    linkFlightSeqInds = getLinkFlightSeqs(increments)
    num = 0
    denom = 0
    for chunk in linkFlightSeqInds:
        flights = increments[chunk]
        dx = flights[1:, 0] - flights[:-1, 0]
        dy = flights[1:, 1] - flights[:-1, 1]
        num += sum(dx[1:] * dx[:-1])
        num += sum(dy[1:] * dy[:-1])
        denom += sum(dx[1:]**2)
        denom += sum(dy[1:]**2)
    rhoHat = num/denom
    return(rhoHat)


def getFlightSeqLengthUnbiasedGeom(increments, K):
    '''
    Coding principles:
    we end with 10 so that if we have two "leading" zero it 
    looks different than an actual 0

    gap start: 1 - pause, 0 flight
    gap end: 1 - pause, 0 flight

    -111: pause, pause
    -101: pause, flight
    - 11: flight, pause
    -  1: flight, flight
    '''
    linkFlightSeqInds = getLinkFlightSeqs(increments)
    fSeqLengths = np.array([len(f) for f in linkFlightSeqInds])
    if len(fSeqLengths) == 0:
        return -1.0

    logLik = logLikFactory(fSeqLengths, K)
    Npoints = 1000
    thetas = np.linspace(1e-3, 1-1e-3, Npoints)
    vals = np.array([logLik(thetas[i]) for i in range(Npoints)])

    inds = np.argsort(np.abs(vals))
    estimate = thetas[inds[0]]
    return(estimate)


def getPauseLengthUnbiasedGeom(increments, K):

    types = increments[:, -1]
    pInds = np.where(types == 1)[0]
    pLengths = increments[pInds, 5]    
    if len(pLengths) == 0:
        return -1.0

    logLik = logLikFactory(pLengths, K)
    Npoints = 1000
    thetas = np.linspace(1e-3, 1-1e-3, Npoints)
    vals = np.array([logLik(thetas[i]) for i in range(Npoints)])
    
    inds = np.argsort(np.abs(vals))
    estimate = thetas[inds[0]]
    return(estimate)


def smallPowers(base, exponents):

    logPowers = exponents * np.log(base)
    indsL = np.where(logPowers < -500)[0]
    indsH = np.where(logPowers > 100)[0]
    indsN = np.where((logPowers <= 100) & (logPowers >= -500))[0]
    powers = np.zeros(len(logPowers))
    powers[indsL] = 0
    powers[indsH] = np.inf
    powers[indsN] = np.exp(logPowers[indsN])
    return(powers)
    

def logLikFactory(seqs, maxSeqLength):
        
    def logLik(p):
        Ls = np.arange(1, maxSeqLength)    
        theoretical = (maxSeqLength - Ls - 1) * smallPowers(1 - p, Ls)
        const = sum(theoretical)
        likelihood = np.sum(np.log(maxSeqLength - seqs - 1)  + seqs * np.log(1 - p)- np.log(const))
        return(likelihood)

    return(logLik)


# def derivativeFactory(meanDelta, maxSeqLength):

#     def derivative(theta):

#         if theta < 0:
#             return np.inf
#         elif theta > 1:
#             return -np.inf

#         cs = maxSeqLength - 2 - np.arange(1, maxSeqLength - 1)
#         LogPowOfTheta = np.arange(1, maxSeqLength - 1) * np.log(theta)

#         indsL = np.where(LogPowOfTheta < -500)[0]
#         indsH = np.where(LogPowOfTheta > 100)[0]
#         indsN = np.where((LogPowOfTheta <= 100) & (LogPowOfTheta >= -500))
#         powersOfTheta = np.zeros(maxSeqLength - 2)
#         powersOfTheta[indsL] = 0
#         powersOfTheta[indsH] = np.inf
#         powersOfTheta[indsN] = np.exp(LogPowOfTheta[indsN])

#         sumNum = np.dot(np.arange(1, maxSeqLength - 1), cs * powersOfTheta)
#         sumDen = np.dot(cs, powersOfTheta)
#         derValue = meanDelta - sumNum/sumDen
#         return derValue

#     return(derivative)

# # loglikehood of samples given in a
# # samples are given in the polar format


# # pPause = prob of a pause after a flight
# # pFlight = prob of a flight after a pause
# def getMarkovLogLik(states, pFlight, pPause):

#     indF = np.where(states[:-1] == 0)[0]
#     FF = len(np.where(states[indF + 1] == 0)[0])
#     FP = len(indF) - FF

#     indP = np.where(states[:-1] == 1)[0]
#     PF = len(np.where(states[indP + 1] == 0)[0])
#     PP = len(indP) - PF

#     try:
#         loglik = FF * np.log(1 - pPause)
#     except RuntimeWarning:
#         pdb.set_trace()
#     if pPause > 0:
#         loglik += FP * np.log(pPause)
#     if pFlight > 0:
#         loglik += PF * np.log(pFlight)
#     if pFlight < 1:
#         loglik += PP * np.log(1 - pFlight)

#     return loglik


# # TO DO: make sure you can calculate likelihood
# # when there are pauses
# def loglik(samples, rho, sig2, pFlight, pPause):

#     # state = 1 if it's a pause and state = 0 if its a flight
#     states = np.array(samples[:, 0] == 0, dtype="int")
#     stateLl = getMarkovLogLik(states, pFlight, pPause)

#     # indices of increments other than pauses
#     # (not observed or flights)
#     fInds = np.where(states == 0)[0]
#     samples = samples[fInds, :]

#     obsPat = np.where(np.isfinite(samples[:, 0]))[0]
#     nObs = len(obsPat)
#     samples = samples[obsPat, :]

#     rhoExps = obsPat[1:] - obsPat[:(nObs - 1)]
#     nExps = len(rhoExps)
#     mins = -500 * np.ones(nExps)
#     rhoPowExps = np.exp(np.max([2 * rhoExps * np.log(rho), mins], axis=0))
#     neg2sqrtTerm = np.sum(np.log((rhoPowExps - 1) / (rho ** 2 - 1)))
#     sqrtTerm = -0.5 * neg2sqrtTerm

#     vAll = polar2cart(samples)
#     v = vAll[:(nObs - 1), :]
#     vLag = vAll[1:, :]
#     rhos = np.exp(np.max([rhoExps * np.log(rho), mins], axis=0))
#     diff = v - rhos[:, None] * vLag
#     norms = np.linalg.norm(diff, 2, axis=1) ** 2
#     quadTerm = -np.sum(norms) / (2 * sig2)
#     prodLTerm = np.sum(np.log(samples[:, 0]))

#     const = -nObs * np.log(2 * np.pi)
#     logdet = -nObs * np.log(sig2)
#     loglik = const + prodLTerm + logdet + sqrtTerm + quadTerm + stateLl

#     return(loglik)


# def getMLEs(pings, verbose=False):
#     incremCart = np.diff(pings, axis=0)
#     incremPolar = cart2polar(incremCart)

#     dx = incremCart[:, 0]
#     dy = incremCart[:, 1]

#     rhoHat = np.nansum(dx[1:] * dx[:-1]) / np.nansum(dx[1:]**2)
#     xTerm = (dx[1:] - rhoHat * dx[:-1]) ** 2
#     yTerm = (dy[1:] - rhoHat * dy[:-1]) ** 2
#     S = np.nansum(xTerm + yTerm)
#     sig2_hat = S / dx[1:].shape[0]

#     state = (incremPolar[:, 0] == 0).astype("int")

#     indF = np.where(state[:-1] == 0)[0]
#     FF = len(np.where(state[indF + 1] == 0)[0])
#     FP = len(indF) - FF

#     indP = np.where(state[:-1] == 1)[0]
#     PF = len(np.where(state[indP + 1] == 1)[0])
#     PP = len(indP) - PF

#     if (PP + PF) > 0:
#         pFlight_hat = PF/(PP + PF)
#     else:
#         pFlight_hat = 0.0
#     pPause_hat = FP/(FF + FP)

#     return((rhoHat, np.sqrt(sig2_hat), pPause_hat, pFlight_hat))
