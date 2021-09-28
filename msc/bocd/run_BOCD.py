# %% md

# A Unit Test for BOCD


# %%

import os
import numpy
import numpy.random
import numpy.linalg
from functools import partial

from typing import Optional

from .bocd import BOCD, StudentT, constant_hazard
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

LAMBDA = 100
ALPHA = 0.1
BETA = 1.
KAPPA = 1.
MU = 0.
DELAY = 15
THRESHOLD = 0.5


def synthetic_data():
    xs = numpy.random.normal(size=1000)
    xs[len(xs) // 4:len(xs) // 2] += 10.
    xs[len(xs) // 2:3 * len(xs) // 4] -= 10.
    return xs


def get_BOCD_changepoints(series, verbose=True):
    """return detected changepoints.
    """
    bocd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU))
    changepoints = []
    for x in tqdm(series[:DELAY], desc="pre"):
        bocd.update(x)
    for x in tqdm(series[DELAY:], desc="main"):
        bocd.update(x)
        if bocd.growth_probs[DELAY] >= THRESHOLD:
            changepoints.append(bocd.t - DELAY + 1)
    if verbose:
        print(f"changepoints: {changepoints}")
    return changepoints


def plot_BOCD_changepoints(data, changepoints, sfreq, output_path: Optional[str] = None, channel_name: Optional[str] = '?'):
    sns.lineplot(data=data, label=f"EEG[{channel_name}]")
    sns.scatterplot(x=changepoints, y=[0] * len(changepoints), color='orange', zorder=3, label="BOCD changepoints")
    plt.title("Bayesian Online Changepoint Detection on EEG")
    plt.xlabel(f"sample number, sfreq={sfreq}")
    if output_path:
        plt.savefig(os.path.join(output_path, f"{channel_name}_bocd.png"))
    else:
        plt.show()
    plt.clf()
