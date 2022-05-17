# ## imports
import numpy as np
# import pdb
# from loadtraces import load_data
# import matplotlib
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

from imputation import fillGap
# from likelihood import loglik, getMLEs
# from tools import cart2polar
# from sampling import sampleData
# from plotting import findPauses, TrajGraph, getLimits

# ## settings
# np.random.seed(1996)
# N_SAMP_GAP = 1
# NSAMPLE = 1
# LAMBDA = 10
# MIS_PERC = 0.3
# SET = "Statefair" # KAIST, Orlando, NewYork, NCSU, Statefair


# SMALL_SIZE = 8
# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)


def fillGapsRheeData(pings, theta):
                    
    gapStart = np.where(np.isnan(pings[:, 0]))[0][0]
    gapEnd = np.where(np.isnan(pings[:, 0]))[0][-1] + 1
    prev = pings[gapStart - 2, :]
    last = pings[gapStart - 1, :]
    first = pings[gapEnd, :]
    second = pings[gapEnd + 1, :]

    filled = fillGap(prev, last, first, second, gapEnd - gapStart, theta[1], theta[0], theta[2], theta[3])
    return(filled)


# ## main script
# if __name__ == "__main__":

#     fig = plt.figure()
#     sets = ["KAIST"]#, "NCSU", "Statefair", "NewYork", "Orlando"]):
    
#     for sIdx, setName in enumerate(sets):
#         data = load_data(setName)
#         inds = np.random.choice(np.arange(len(data)), size = NSAMPLE, replace = False)
#         data = [data[i]/1000 for i in inds]
        
#         for idx, pings in enumerate(data):
            
#             print(f"===== set {idx + 1} =====")
#             n = pings.shape[0]
#             print(f"Total number of pings is {n}")
#             truePings = np.copy(pings)

#             ## hide observations
#             misStart = max(round(np.quantile(range(n), 0.5 - MIS_PERC)), 2)
#             misStop = min(round(np.quantile(range(n), 0.5 + MIS_PERC)), n - 2)
#             obsInds = np.hstack([np.arange(misStart), np.arange(misStop, n)])
#             misInds = np.setdiff1d(np.arange(n), obsInds)    
            
#             print(f"No missing: {len(misInds)}")
#             pings[misInds, :] = np.nan
#             pauses = findPauses(truePings)
#             Npauses = pauses.shape[0]
        
#             ## plot observations
#             ax = fig.add_subplot(len(sets), NSAMPLE, sIdx * NSAMPLE + idx + 1)
#             t = TrajGraph(truePings[obsInds, :], colors = "black", linewidth = 1)
#             ax.add_collection(t)
#             for i in range(Npauses):
#                 ax.scatter(pauses[i, 0], pauses[i, 1], s = pauses[i, 2], color = "grey")
            

#             tm = TrajGraph(truePings[misInds, :], colors = "red", linewidth = 1)
#             ax.add_collection(tm)
        
        
#             ## fill in gaps
#             params = getMLEs(pings, verbose = True)
#             fills = [None] * N_SAMP_GAP

#             for iter in range(N_SAMP_GAP):
#                 fills[iter] = fillGapsRheeData(pings, params)

#             sub = np.empty(0)#np.arange(max(np.quantile(n, 0.5 - 2 * MIS_PERC), 2),
#             #      min(np.quantile(n, 0.5 + 2 * MIS_PERC), n - 2), dtype = "int")
#             trueXlim = getLimits([truePings], "x", subset = sub)
#             trueYlim = getLimits([truePings], "y", subset = sub)
#             fillXlim = getLimits(fills, "x")
#             fillYlim = getLimits(fills, "y")
#             ax.set_xlim(min(trueXlim + fillXlim), max(trueXlim + fillXlim))
#             ax.set_ylim(min(trueYlim + fillYlim), max(trueYlim + fillYlim))
            
    
#             for iter in range(N_SAMP_GAP):
#                 tmf = TrajGraph(fills[iter], colors = "blue", linewidth = 0.3)
#                 ax.add_collection(tmf)
#                 pauses = findPauses(fills[iter])
#                 Npauses = pauses.shape[0]
#                 for i in range(Npauses):
#                     ax.scatter(pauses[i, 0], pauses[i, 1], s = pauses[i, 2], color = "grey")
                
#             ax.set_title(f"C = {params[0]:.2f}, SD = {np.sqrt(params[1]):.2f}, len = {len(misInds)}")
                

        
#     #plt.tight_layout()
#     plt.show()
