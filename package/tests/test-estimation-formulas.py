import pdb
import numpy as np
import matplotlib.pyplot as plt

K = 20

def derivative(theta, K):
    cs = K - 1 - np.arange(1, K - 1)
    LogPowOfTheta = np.arange(1, K - 1) * np.log(theta)
    LogPowOfTheta[LogPowOfTheta < -500] = 0
    powersOfTheta = np.exp(LogPowOfTheta)
    sumNum = np.dot(np.arange(1, K - 1), cs * powersOfTheta)
    sumDen = np.dot(cs, powersOfTheta)
    derValue = -sumNum/sumDen
    return derValue

#theta1_0 = len(pFollowsF) / len(fInds) # the normal MLE
Npoints = 10000
thetas = np.linspace(1e-16, 1 - 1e-16, Npoints)
vals = np.array([derivative(thetas[i], 10) for i in range(Npoints)])
vals2 = np.array([derivative(thetas[i], 11) for i in range(Npoints)])
vals3 = np.array([derivative(thetas[i], 10) for i in range(Npoints)])
plt.plot(thetas, vals, color = "blue")
plt.plot(thetas, vals2, color = "green")
plt.plot(thetas, vals3, color = "orange")
plt.axhline(y=0, color="black")
plt.show()

pdb.set_trace()

