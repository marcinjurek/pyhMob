import pdb
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from likelihood import loglik
from tools import sampleMissing, sampleAnglesLengths
np.random.seed(1996)

### This script verifies that the likelihood function is implemented properly
# In order to do this we first simulate a trajectory with missing data.
# Next we evaluate the likliehood on a grid and expect that the values of the
# parameters at which the likelihood has the highest values
# are close to the true parameters. This verification is done visually by
# plotting a heatmap of the likelihood and the true parameter values.


## Trajectory parameters
N = 200
SD = 0.1
C = 0.2
LAMBDA = 10
FRAC_OBS = 0.99
PROB_FLIGHT = 0
PROB_PAUSE = 0


## simulate trajectory
pattern = sampleMissing(N, LAMBDA, FRAC_OBS/(1-FRAC_OBS) * LAMBDA)
draws, drawsRaw = sampleAnglesLengths(N, SD, C) 
allDraws = draws
draws[pattern==0, :] = np.nan
print(f"We observe {int(100 * sum(pattern) / N)}% of pings")



## evaluate likelihood on a grid
nSig2 = 20
nC = 20
sig2s = np.linspace(0.005, 0.1, num = nSig2)
cs = np.linspace(0.001, 0.999, num = nC)
params = np.array(list(it.product(sig2s, cs)))
nParams = np.size(params, 0)

logliks = np.zeros(nParams)

for i in range(nParams):
    sig2 = params[i, 0]
    c = params[i, 1]
    ll = loglik(draws, c, sig2, PROB_FLIGHT, PROB_PAUSE)
    logliks[i] = ll


## Plot likelihood and the true values
# color palette
levels = mpl.ticker.MaxNLocator(nbins = 15).tick_values(logliks.min(), logliks.max())
cmap = plt.get_cmap("Spectral")
norm = mpl.colors.BoundaryNorm(levels, ncolors = cmap.N, clip = True)

# plot a heatmap
fig, ax = plt.subplots()
im = ax.pcolormesh(cs, sig2s, logliks.reshape(nSig2, nC), cmap = cmap, norm = norm, shading = 'auto')
fig.colorbar(im, ax = ax)
ax.set_title("likelihood")
ax.set_ylabel("sig2")
ax.set_xlabel("cs")
ax.axhline(y = SD ** 2, linestyle = "dashed", color = "k" )
ax.axvline(x = C, linestyle = "dashed", color = "k")
plt.show()
