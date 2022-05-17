import scipy as sp
import pdb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tools import sampleMissing
from plotting import TrajGraph


def imputeAR1(NMiss, obs_1n, rho, var):

    Var_1n = var * np.matrix([[1, rho ** NMiss], [rho ** NMiss, 1]])
    Prec_1n = np.linalg.inv(Var_1n)
    
    part_cov = var * (rho ** np.arange(1, NMiss + 1))
    Cov_2nm1 = np.matrix(np.vstack((part_cov, part_cov[::-1])).T)
    mu_cond = Cov_2nm1 * Prec_1n * obs_1n
    
    cond_Sig = sp.linalg.toeplitz(part_cov) - Cov_2nm1 * Prec_1n * (Cov_2nm1.T)
    C = np.linalg.cholesky(cond_Sig)
    sim = C * (np.matrix(np.random.normal(0, np.sqrt(var), NMiss)).T)
    
    return(sim + mu_cond)


if __name__ == "__main__":

    ## settings
    RHO = 0.99
    T = 100
    LAMBDA = 30
    FRAC_OBS = 0.6
    MU_0 = 0
    SIG_2_0 = 1.0
    sig_2 = (1 - RHO ** 2) * SIG_2_0
    NREP = 9

    fig = plt.figure()
    for rep in range(NREP):
        ## generate data
        eps = np.random.normal(0, np.sqrt(sig_2), T)
        
        x = np.zeros(T)
        x[0] = np.random.normal(MU_0, np.sqrt(SIG_2_0), 1)
        for t in range(1, T):
            x[t] = RHO * x[t - 1] + eps[t]
        
        pattern = sampleMissing(T, LAMBDA, FRAC_OBS/(1 - FRAC_OBS) * LAMBDA)
        indsM = np.where(pattern==0)[0]
        NMiss = indsM.shape[0]
        obs = np.copy(x)
        obs[pattern == 0] = np.nan

        ## plot data
        ax = fig.add_subplot(3, 3, rep + 1)
        data = np.vstack((np.arange(T), obs)).T
        trajObs = TrajGraph(data, color = "black")
        ax.add_collection(trajObs)
    
        dataMiss = np.vstack((np.arange(T), x)).T
        dataMiss = dataMiss[indsM,:]
        trajMiss = TrajGraph(dataMiss, color = "blue")
        ax.add_collection(trajMiss)
        
        lastObs = np.min(indsM) - 1
        firstObs = np.max(indsM) + 1
        obs_1n = np.matrix([[obs[lastObs]], [obs[firstObs]]]) 
        imputed = imputeAR1(NMiss, obs_1n, RHO, SIG_2_0)
        ax.plot(np.arange(lastObs + 1, firstObs), imputed, color = "red", linestyle = "dashed")

        ax.set_xlim(-1, T + 1)
        m = min(np.min(imputed), np.min(x))
        M = max(np.max(imputed), np.max(x))
        ax.set_ylim(m, M)

    plt.show()


