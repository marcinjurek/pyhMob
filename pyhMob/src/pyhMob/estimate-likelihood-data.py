import numpy as np
from imputation import fillGap


def fillGapsRheeData(pings, theta):

    gapStart = np.where(np.isnan(pings[:, 0]))[0][0]
    gapEnd = np.where(np.isnan(pings[:, 0]))[0][-1] + 1
    prev = pings[gapStart - 2, :]
    last = pings[gapStart - 1, :]
    first = pings[gapEnd, :]
    second = pings[gapEnd + 1, :]

    filled = fillGap(prev, last, first, second, gapEnd - gapStart, theta[1], theta[0], theta[2], theta[3])
    return(filled)
