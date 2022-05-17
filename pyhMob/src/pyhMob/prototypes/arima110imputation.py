import pdb
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def smooth(E, H, Q, R, mu0, Sig0, obs):

    ndim = mu0.shape[0]
    T = len(obs)

    Zm = np.matrix(np.zeros((ndim, ndim)))
    Zv = np.matrix(np.zeros((ndim, 1)))
    mu_pred = [Zv] * T
    sig_pred = [Zm] * T
    mu_filt = [Zv] * T
    sig_filt = [Zm] * T
    mu_smooth = [Zv] * T
    sig_smooth = [Zm] * T

    mu_filt[-1] = mu0
    sig_filt[-1] = Sig0
    for t in range(0, T):
        sig_pred[t] = E * sig_filt[t - 1] * E.T + Q
        mu_pred[t] = E * mu_filt[t - 1]
        if np.any(np.isnan(obs[t])):
            mu_filt[t] = mu_pred[t]
            sig_filt[t] = sig_pred[t]
        else:
            K = sig_pred[t] * H.T * np.linalg.inv(H * sig_pred[t] * H.T + R)
            mu_filt[t] = mu_pred[t] + K * (obs[t] - H * mu_pred[t])
            sig_filt[t] = (np.eye(2) - K * H) * sig_pred[t]

    mu_smooth[-1] = mu_filt[-1]
    sig_smooth[-1] = sig_filt[-1]
    for t in range(T - 2, -1, -1):
        C = sig_filt[t] * E.T * np.linalg.pinv(sig_pred[t + 1])
        mu_smooth[t] = mu_filt[t] + C * (mu_smooth[t + 1] - mu_pred[t + 1])
        dif = sig_smooth[t + 1] - sig_pred[t + 1]
        sig_smooth[t] = sig_filt[t] + C * dif * C.T

    return(mu_smooth, sig_smooth)


if __name__ == "__main__":

    T = 1000
    RHO = 0.9
    SIG = 1
    gapStart = 30  # index of the first unavailable obs
    gapEnd = 500  # index of the first available obs after the gap

    # # simulate data
    nu = norm.rvs(0, scale=SIG, size=T)
    delta = np.zeros(T)
    for t in range(1, T):
        delta[t] = RHO * delta[t - 1] + nu[t]
    x = np.cumsum(delta)  

    gapLength = gapEnd - gapStart

    A = np.matrix([[1, RHO], [0, RHO]])
    H = np.matrix([[1, 0]])
    Q = (SIG ** 2) * np.matrix([[1, 1], [1, 1]])
    R = 0
    mu0 = np.matrix([[x[gapStart - 1]], [x[gapStart - 1] - x[gapStart - 2]]])
    Sig0 = np.zeros((2, 2))
    obs = np.nan * np.zeros(gapLength + 2)
    obs[-2:] = x[gapEnd:(gapEnd + 1)]

    # xA = np.zeros((2, T))
    # xA[:, 1] = np.array([delta[1], delta[1]])
    # for t in range(2, T):
    #     xm = np.matrix(xA[:, t - 1]).T
    #     num = np.matrix([[nu[t], nu[t]]]).T
    #     next_xm = A * xm + num
    #     xA[:, t] = np.array(next_xm).ravel()

    # xA = xA[0, :]
    
    # Generate several examples to check if things look good
    fig = plt.figure()
    for i in range(9):
        nu_hat = norm.rvs(0, scale=SIG, size=gapLength + 2)
        eps_hat = np.zeros(gapLength + 2)
        for t in range(1, gapLength + 2):
            eps_hat[t] = RHO * eps_hat[t - 1] + nu_hat[t]
        x_hat = np.cumsum(eps_hat)
        y_hat = np.nan * eps_hat
        y_hat[-2:] = x_hat[-2:]

        y_hat_star = obs - y_hat

        muS, sigS = smooth(A, H, Q, R, mu0, Sig0, y_hat_star)
        mu = np.array([m[0, 0] for m in muS])
        x_samp = x_hat + mu
        x_samp = np.hstack((x[gapStart - 1], x_samp))

        fig.add_subplot(3, 3, i + 1)
        plt.plot(x, color="black")
        missingInds = np.arange(gapStart - 1, gapEnd + 2)
        plt.plot(missingInds, x_samp, color="blue", linestyle="dashed")

    plt.show()
