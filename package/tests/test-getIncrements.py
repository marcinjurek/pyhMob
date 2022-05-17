import sys
sys.path.append("..")
from pyMobility.tools import getIncrements
from pyMobility.sampling import sampleTrajectory, sampleData
import numpy as np


# test for a standard set of pings
def test_getIncrements():

    pings = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 1, 3],
                      [1, 1, 4],
                      # [np.nan, np.nan. 5],
                      # [np.nan, np.nan, 6],
                      [2, 2, 7],
                      [2, 2, 8],
                      [2, 3, 9],
                      [2, 3, 10],
                      [2, 3, 11],
                      [4, 1, 12],
                      # [np.nan, np.nan, 13],
                      [1, 2, 14],
                      [1, 1, 15]])
    
    trueIncs = np.array([[0, 0, 0, 0, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1],
                         [0, 1, 2, 1, 0, 1, 0],
                         [1, 1, 4, 1, 1, 3, -111],
                         [2, 2, 8, 0, 1, 1, 0],
                         [2, 3, 9, 0, 0, 2, 1],
                         [2, 3, 11, 2, -2, 1, 0],
                         [4, 1, 12, -3, 1, 2, -1],
                         [1, 2, 14, 0, -1, 1, 0]]) 
    
    incs = getIncrements(pings)
    #print(incs)
    assert np.sum(np.abs(incs - trueIncs)) == 0, "test 1 failed"


# test for pings 
def test_getIncrements2():
    
    pings2 = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 2],
                       [1, 1, 3],
                       [1, 1, 4],
                       # [np.nan, np.nan. 5],
                       # [np.nan, np.nan, 6],
                       [2, 2, 7],
                       [2, 2, 8],
                       [2, 3, 9],
                       [2, 3, 10],
                       [2, 3, 11],
                       [4, 1, 12],
                       # [np.nan, np.nan, 13],
                       [1, 2, 14],
                       [1, 2, 15]])

    trueIncs2 = np.array([[0, 0, 0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0, 1, 1],
                          [0, 1, 2, 1, 0, 1, 0],
                          [1, 1, 4, 1, 1, 3, -111],
                          [2, 2, 8, 0, 1, 1, 0],
                          [2, 3, 9, 0, 0, 2, 1],
                          [2, 3, 11, 2, -2, 1, 0],
                          [4, 1, 12, -3, 1, 2, -101]])

    incs2 = getIncrements(pings2)
    assert np.sum(np.abs(incs2 - trueIncs2)) == 0, "test 2 failed"



    # test for pings 
def test_getIncrements3():
    
    pings2 = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 2],
                       [1, 1, 3],
                       [1, 1, 4],
                       # [np.nan, np.nan. 5]
                       [1, 2, 6],
                       [2, 2, 7],
                       [2, 2, 8],
                       [2, 3, 9],
                       [2, 3, 10],
                       [2, 3, 11],
                       [4, 1, 12],
                       # [np.nan, np.nan, 13],
                       [1, 2, 14],
                       [1, 2, 15]])

    trueIncs2 = np.array([[0, 0, 0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0, 1, 1],
                          [0, 1, 2, 1, 0, 1, 0],
                          [1, 1, 4, 0, 1, 2, -11],
                          [1, 2, 6, 1, 0, 1, 1],
                          [2, 2, 8, 0, 1, 1, 0],
                          [2, 3, 9, 0, 0, 2, 1],
                          [2, 3, 11, 2, -2, 1, 0],
                          [4, 1, 12, -3, 1, 2, -101]])

    incs2 = getIncrements(pings2)
    print(incs2)
    print(trueIncs2)
    assert np.sum(np.abs(incs2 - trueIncs2)) == 0, "test 3 failed"

    
if __name__ == "__main__":
    
    test_getIncrements()
    test_getIncrements2()
    test_getIncrements3()

    start = np.array([0, 0])
    Nsteps = 1000
    trajectory = sampleTrajectory(start, start, 1000, 1.0, 0.95, 0.1, 0.1)
