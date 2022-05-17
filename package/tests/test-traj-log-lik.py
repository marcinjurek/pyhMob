import pdb
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from likelihood import trajLogLik, loglik
from plotting import TrajGraph
from tools import sampleMissing, sampleTrajectory
np.random.seed(1996)

### This script verifies that the trajLogLik function is implemented properly
# In order to do this we first simulate a trajectory with missing data.
# Next we evaluate the likliehood on a grid and expect that the values of the
# parameters at which the likelihood has the highest values
# are close to the true parameters. This verification is done visually by
# plotting a heatmap of the likelihood and the true parameter values.


## Trajectory parameters
N = 2000
SD = 0.1
C = 0.99
LAMBDA = 50
FRAC_OBS = 0.8


## simulate trajectory
pattern = sampleMissing(N, LAMBDA, FRAC_OBS/(1-FRAC_OBS) * LAMBDA)
start = np.array([0, 0])
truePings, draws = sampleTrajectory(start, start, N-1, SD, C)
pings = np.copy(truePings)
pings[pattern==0, :] = np.nan
print(f"We observe {int(100 * sum(pattern) / N)}% of pings")


## plot simulated trajectory
m = np.min(truePings, axis = 0)
M = np.max(truePings, axis = 0)
xlim = np.array([m[0], M[0]])
ylim = np.array([m[1], M[1]])
            
fig1 = plt.figure()
ax = fig1.add_subplot(121)
colors = (0, 0, 0, 1)
trajFull = TrajGraph(pings, colors = colors, linewidth = 1.0)
ax.add_collection(trajFull)

ax.set_xlim(xlim)
ax.set_ylim(ylim)



## evaluate likelihood on a grid
nSig2 = 20
nC = 20
sig2s = np.linspace(0.005, 0.02, num = nSig2)
cs = np.linspace(0.8, 0.999, num = nC)
params = np.array(list(it.product(sig2s, cs)))
nParams = np.size(params, 0)

logliks = np.zeros(nParams)

for i in range(nParams):

    sig2 = params[i, 0]
    c = params[i, 1]
    ll = trajLogLik(pings[1:(N-1), :], pings[0, :], pings[N-1, :], sig2, c)
    logliks[i] = ll

    
## Plot likelihood and the true values
# color palette
levels = mpl.ticker.MaxNLocator(nbins = 15).tick_values(logliks.min(), logliks.max())
cmap = plt.get_cmap("Spectral")
norm = mpl.colors.BoundaryNorm(levels, ncolors = cmap.N, clip = True)

# plot a heatmap
ax2 = fig1.add_subplot(122)
im = ax2.pcolormesh(cs, sig2s, logliks.reshape(nSig2, nC), cmap = cmap, norm = norm, shading = 'auto')
fig1.colorbar(im, ax = ax)
ax2.set_title("likelihood")
ax2.set_ylabel("sig2")
ax2.set_xlabel("cs")
ax2.axhline(y = SD ** 2, linestyle = "dashed", color = "k" )
ax2.axvline(x = C, linestyle = "dashed", color = "k")
plt.show()
