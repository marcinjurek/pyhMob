import unittest
from src.uniMLE import getMLEMarkovOnOff
from src.sampling import sampleData
# from src.tools import getPattern
import numpy as np
# import matplotlib.pyplot as plt
# import pdb

N = 10000
U = 100  # the length of the sequence of observations
SDEV = 0.6
C = 0.95
# probability of a pause after a flight
PROB_P = 0.01  # avg length of seuqence of flights is 1/PROB_P
# probability of a flight after a 1 second of a pause
PROB_F = 0.01  # avg. pause duration is 1/PROB_F
OBS_FRAC = 0.5
NGRID = 50
pPPmax = 0.999
pFFmax = 0.999


class TestEstimation(unittest.TestCase):

    def test_MLE_OnOff(self):

        np.random.seed(1996)
        truePings, pings = sampleData(N, U, OBS_FRAC, SDEV, C, PROB_P,
                                      PROB_F, missingPatDist="none",
                                      inclTimes=True)

        dif = np.diff(pings[:, -1])
        bLens = np.unique(dif)[1:]
        incidence = np.array([np.where(dif == bL)[0].shape[0] for bL in bLens])
        if OBS_FRAC == 1.0:
            Io = N
            Iu = 0
        else:
            Iu = bLens[np.argmax(incidence)]
            Io = U
        pPP, pFF = getMLEMarkovOnOff(pings, Io, Iu, NGRID, pPPmax, pFFmax)
        
        
        isClosePP = abs(pPP - PROB_P) < 1/NGRID
        message = f"estimator for PP is {pPP} vs. true value of {PROB_P}"
        self.assertTrue(isClosePP, message)

        isCloseFF = abs(pFF - PROB_F) < 1/NGRID
        message = f"estimator for PP is {pPP} vs. true value of {PROB_P}"
        self.assertTrue(isCloseFF, message)

        
if __name__ == "__main__":
    unittest.main()
