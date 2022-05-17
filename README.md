# pyhMob
The `pyhMob` folder contains implementation of the flight-pause model of human mobility in the form of a Python package. The intended use is to install this package and then use the contents of the `scripts` folder to reproduce the results of the paper and experiment with the model.

## Installation

The following instructions show how to install the package on an Ubuntu system. Perhaps the easiest way to do it is to clone the repository using
```
git clone https://github.com/marcinjurek/pyhMob.git
```
into the desired directory. Then navigate to that directory and run
```
python -m pip install pyhMob
```

## Usage

It is recommended to explore the individual scripts to get an idea of the capabilities of the package. As a simple example let us generate a sample trajectory from the model. The way to do it is shown in the `scripts/exploratory/draw-sample-trajectories.py` file pasted below
```{python}
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhMob.plotting import getAxesWithTrajectory
from pyhMob.sampling import sampleData


class Params:
    MISS_LWD = 0.3
    NORMAL_LWD = 1
    SMALL_SIZE = 10
    mpl.rc('font', size=SMALL_SIZE)
    mpl.rc('axes', titlesize=SMALL_SIZE)
    CMAP = plt.get_cmap("Set1")


class Conf:
    N = 1000
    LAMBDA = 500
    NREP = 22
    NSIMS = 36
    SD = 1.0
    C = 0.95
    PROB_P = 0.1
    PROB_F = 0.1


if __name__ == "__main__":

    truePings, pings = sampleData(Conf.N, Conf.LAMBDA, 1.0, Conf.SD,
                                  Conf.C, Conf.PROB_P, Conf.PROB_F)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax, t = getAxesWithTrajectory(ax, truePings, "grey", "black", Params.NORMAL_LWD)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
```
