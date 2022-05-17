import numpy as np
import pdb


def getLinkFlightSeqs(increments):

    nIncs = len(increments)
    types = increments[:, -1]
    fInds = np.where(types == 0)[0]
    if fInds[-1] == (types.shape[0] - 1):
        fInds = fInds[:-1]

    mask = np.where(types != 0)[0]
    splits = np.split(np.arange(nIncs), mask)
    
    fSeqs = []
    for noSeq in range(len(splits)-1):
        seq = splits[noSeq]
        
        if len(seq) == 1:
            continue
        if not len(seq):
            continue
        # has the sequence ended with a flight?
        follIncInd = splits[noSeq + 1][0] 
        if types[follIncInd] in [-1, -11]:
            continue
        precIncInd = splits[noSeq][0]
        if types[precIncInd] in [-1, -101]:
            continue
        if noSeq > 0:
            seq = seq[1:]  # we subtract the first element
        fSeqs += [seq]

    return(fSeqs)



# def getFlightSequencesInds(increments):

#     types = np.copy(increments[:, -1])
#     types[types != 0] = 1
#     allInds = np.arange(len(types))
#     rawChunks = np.split(allInds, np.where(types == 1)[0])
#     chunks = [rawChunks[0]] + [chunk[1:] for chunk in rawChunks[1:]]

#     return(chunks)


def getIncrements(data, tempRes=1):
    """
    Generates an array with linkable increments from the array of
    pings

    Parameters
    -------
    data : np.ndarray
        an 3-column array where the first two columns represent the 
        spatial location and the third is the timestamp, expressed
        as the number of seconds since the first ping.

    tempRes : int
        the temporal resultion of the problem: the number of seconds
        that a flight lasts; pauses can have durations that are 
        multiples of this number

    Returns
    -------
    allIncs : np.ndarray
        a 7-column array where each row represents an increment encoded
        using the notation from the paper:
        startX, startY, startTime, DeltaX, DeltaY, DeltaT, type
        where type is 1 for pauses and 0 for flight, 2 for unobsered 
        periods.

    avgData : np.ndarray
        returned only if if tempRes > 1, denotes the locations averaged over
        tempRes seconds.
    """

    # remove rows that contain a nan
    data = data[~np.isnan(data).any(axis=1)]

    pauses = findPauses(data, inclTimes = True)
    pauses[:, -1] *= tempRes # this corrects the length of pauses
    flights = findFlights(data)
    Nflights = flights.shape[0]
    Npauses = pauses.shape[0]
    dummies = findDummies(data, inclTimes = True)
    
    fColOrder = np.array([0, 1, 4, 2, 3])
    flights = flights[:, fColOrder]
    flights = np.hstack((flights, np.ones((Nflights, 1)))) # add inc length
    flights = np.hstack((flights, np.zeros((Nflights, 1)))) # add inc type

    pauses = np.hstack((pauses, np.zeros((Npauses, 2))))
    pColOrder = np.array([0, 1, 2, 4, 5, 3])
    pauses = pauses[:, pColOrder]
    pauses = np.hstack((pauses, np.ones((Npauses, 1)))) # add inc type
    
    allIncs = np.vstack((flights, pauses, dummies))
    times = allIncs[:, 2]
    incOrder = np.argsort(times)
    allIncs = allIncs[incOrder, :]
    
    return(allIncs)


def getEffSampSize(pings, tempRes = 1):

    incs, avgPings = getIncrements(pings, tempRes = tempRes)
    return incs.shape[0], int(incs[:, -1].sum())


def cart2polar(x):
    rho = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    phi = np.arctan2(x[:, 1], x[:, 0])
    return(np.column_stack([rho, phi]))


def polar2cart(p):
    x = p[:, 0] * np.cos(p[:, 1])
    y = p[:, 0] * np.sin(p[:, 1])
    return(np.column_stack([x, y]))


# 0 = flight, 1 = pause
def getPattern(pings):

    Npings = pings.shape[0]
    difx = np.diff(pings[:, 0])
    dify = np.diff(pings[:, 1])
    inds = np.intersect1d(np.where(difx == 0)[0], np.where(dify == 0)[0])
    pattern = np.zeros(Npings - 1)
    pattern[inds] = 1
    return(pattern)


# returns an N x p array, where p = 3, 4. The first two
# columns are always the start location of the dummy. The last
# column is always the duration of the dummy. If inclTimes = True
# then there are four columns and the third column is the time
# at which the dummy starts.
def findDummies(pings, inclTimes = False):

    if pings.shape[1] == 3:
        chunks = findChunks2(pings)
    elif pings.shape[1] == 2:
        chunks = findChunks(pings, times = inclTimes)

    dummies = []
    nChunks = len(chunks)
    if nChunks == 1:
        return(np.empty((0, 7)))
    
    for chunkNo in range(nChunks - 1):
        chunk = chunks[chunkNo]
        nextChunk = chunks[chunkNo + 1]
        
        dummyStart = chunk[-1, :]
        dummyEnd = nextChunk[0, :]

        change = dummyEnd - dummyStart
        dummy = np.hstack([dummyStart, change])
        dumType = -1
        
        if np.abs(chunk[-2, :2] - chunk[-1, :2]).sum() == 0:
            dumType -= 100

        if np.abs(nextChunk[1, :2] - nextChunk[0, :2]).sum() == 0:
            dumType -= 10

        if inclTimes:
            dummy = np.hstack([dummyStart, dummyEnd - dummyStart, dumType])
        else:
            raise NotImplementedError("dummies wihout times not implemented")
        dummies += [dummy]

    dummies = np.vstack(dummies)
    return(dummies)


# returns an N x p array, where p = 3, 4. The first two
# columns are always the location of the pause. The last
# column is always the duration of the pause. If inclTimes = True
# then there are four columns and the third column is the time
# at which the pause starts.
def findPauses(pings, inclTimes = False):

    if pings.shape[1] == 3:
        chunks = findChunks2(pings)
    elif pings.shape[1] == 2:
        chunks = findChunks(pings, times = inclTimes)

    allPauses = [np.empty((0, 3 + inclTimes))]
    for chunk in chunks:

        difx = np.diff(chunk[:, 0])
        dify = np.diff(chunk[:, 1])
        inds = np.intersect1d(np.where(difx == 0)[0], np.where(dify == 0)[0])
        if not len(inds):
            continue

        dInds = np.diff(inds)
        pLocInds = np.hstack((inds[0], inds[np.where(dInds > 1)[0] + 1]))
        pLocs = chunk[pLocInds]

        splitDInds = (np.where(dInds > 1)[0])
        pLengths = np.array([[len(k) for k in np.split(dInds, splitDInds)]]).T
        pLengths[0, 0] += 1
        pauses = np.hstack((pLocs, pLengths))
        # if the chunk ends with a pause we cannot include it because it is
        # not linkable

        if pLengths[-1] + pLocInds[-1] == len(chunk) - 1:
            pauses = pauses[:-1, :]
        # if the chunk starts with a pause it is also not linkable
        if pauses.shape[0] and pLocInds[0] == 0:
            pauses = pauses[1:]
            
        allPauses += [pauses]

    allPauses = np.vstack(allPauses)
    return(allPauses)


# returns an N x 5 array where the first two columns
# are the beginning of the flight and the second two
# are the x and y displacement, respectively. The last
# column is the time 
def findFlights(pings):

    if pings.shape[1] == 3:
        chunks = findChunks2(pings)
    elif pings.shape[1] == 2:
        chunks = findChunks(pings, times = True)

    allFlights = []
    for chunk in chunks:

        difx = np.diff(chunk[:, 0])
        dify = np.diff(chunk[:, 1])
        inds = np.union1d(np.where(difx != 0)[0], np.where(dify != 0)[0])

        incs = np.hstack((difx[inds, np.newaxis], dify[inds, np.newaxis]))
        flights = np.hstack([chunk[inds, :-1], incs, chunk[inds, -1:]])
        allFlights += [flights]

    allFlights = np.vstack(allFlights)
    return(allFlights)


def findChunks(pings, times = False):

    pattern = np.where(~np.isnan(pings[:, 0]))[0]
    dpat = np.diff(pattern)
    dinds = np.where(dpat > 1)[0]
    chunkStarts = np.append(pattern[0], pattern[dinds + 1])
    chunkEnds = np.append(pattern[dinds], pattern[-1])
    nChunks = len(chunkStarts)

    pingChop = []
    for chunkNo in range(nChunks):
        start = chunkStarts[chunkNo]
        stop = chunkEnds[chunkNo]

        chop = pings[start:(stop+1):]
        if times:
            start = sum([ch.shape[0] for ch in pingChop])
            end = start + chop.shape[0]
            n = end - start
            t = np.arange(start, end).reshape((n, 1))
            chop = np.hstack((chop, t))
        pingChop += [chop]

    return pingChop


# pings is an N x 3 array where the first two columns are the
# spatial coordinates and the third column is the time. The
# temporal resolution is taken to be the smallest difference
# between two consecutive time indices. It is assumed that there
# are no nans.
def findChunks2(pings):

    pattern = pings[:, 2]
    dpat = np.diff(pattern)
    dinds = np.where(dpat > np.min(dpat))[0]
    chunkStarts = np.append(0, dinds + 1)
    chunkEnds = np.append(dinds, len(pattern))
    nChunks = len(chunkStarts)

    pingChop = []
    for chunkNo in range(nChunks):
        start = chunkStarts[chunkNo]
        stop = chunkEnds[chunkNo]

        chop = pings[start:(stop+1):]
        pingChop += [chop]

    return pingChop


def getMissingPattern(N, missType, params=None):

    misInds = np.array([])
    if "contiguous" in missType and "perc" in params.keys():
        perc = params["perc"]
        misPat = np.ones(N)
        gapStart = round(N * (0.5 - perc / 2))
        gapStop = round(N * (0.5 + perc / 2))
        misPat[gapStart:gapStop] = 0
        misIndsC = np.where(misPat == 0)[0]
        misInds = np.union1d(misIndsC, misInds)

    if "on-off" in missType and "on" in params.keys() and "off" in params.keys():
        misPat = np.ones(N)
        K = params["on"] + params["off"]
        misPat[np.arange(N) % K >= params["on"]] = 0
        misPat[K * int(N / K):] = 1
        misPat[-2:] = 1
        misIndsOO = np.where(misPat == 0)[0]
        misInds = np.union1d(misIndsOO, misInds)

    misInds = misInds.astype("int")
        
    return(misInds)
