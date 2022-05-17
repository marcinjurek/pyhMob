import pdb
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from likelihood import loglik
#from tools import sampleMissing, sampleAnglesLengths
from tools import sampleData, cart2polar
np.random.seed(1996)

### This script verifies that the likelihood function is implemented properly
# In order to do this we first simulate a trajectory with missing data.
# Next we evaluate the likliehood on a grid and expect that the values of the
# parameters at which the likelihood has the highest values
# are close to the true parameters. This verification is done visually by
# plotting a heatmap of the likelihood and the true parameter values.


## Trajectory parameters
N = 1000
SD = 0.1
C = 0.2
LAMBDA = 50
FRAC_OBS = 0.15
PROB_PAUSE = 0.05 # prob of a pause after a flight
PROB_FLIGHT = 0.3 # prob of a flight after a pause


## simulate trajectory
truePings, pings = sampleData(N, LAMBDA, FRAC_OBS, SD, C, PROB_PAUSE, PROB_FLIGHT)

incremCart = np.diff(pings, axis = 0)
incremPolar = cart2polar(incremCart)
states = np.array(incremPolar[:, 0]==0, dtype = "int")


## evaluate likelihood on a grid
#nSig2 = 20
#nC = 20
#sig2s = np.linspace(0.005, 0.02, num = nSig2)
#cs = np.linspace(0.001, 0.4, num = nC)
#params = np.array(list(it.product(sig2s, cs)))
#nParams = np.size(params, 0)

numProbs = 100
pfs = np.linspace(0.001, 0.999, num = numProbs)
pps = np.linspace(0.001, 0.999, num = numProbs)
params = np.array(list(it.product(pps, pfs)))
nParams = np.size(params, 0)
logliks = np.zeros(nParams)

for i in range(nParams):
    
    pp = params[i, 0]
    pf = params[i, 1]
    incremCart = np.diff(pings, axis = 0)
    incremPolar = cart2polar(incremCart)
    nInc = incremPolar.shape[0]
    
    ll = loglik(incremPolar, C, SD**2, pf, pp)
    logliks[i] = ll


    
## Plot likelihood and the true values
# color palette
levels = mpl.ticker.MaxNLocator(nbins = 15).tick_values(logliks.min(), logliks.max())
cmap = plt.get_cmap("Spectral")
norm = mpl.colors.BoundaryNorm(levels, ncolors = cmap.N, clip = True)

# plot a heatmap
fig, ax = plt.subplots()
im = ax.pcolormesh(pfs, pps, logliks.reshape(numProbs, numProbs), cmap = cmap, norm = norm, shading = 'auto')
fig.colorbar(im, ax = ax)
ax.set_title("likelihood")
#ax.set_ylabel("sig2")
#ax.set_xlabel("cs")
#ax.axhline(y = SD ** 2, linestyle = "dashed", color = "k" )
#ax.axvline(x = C, linestyle = "dashed", color = "k")
ax.set_ylabel("prob pauses")
ax.set_xlabel("prob flights")
ax.axhline(y = PROB_PAUSE, linestyle = "dashed", color = "k" )
ax.axvline(x = PROB_FLIGHT, linestyle = "dashed", color = "k")
plt.show()
