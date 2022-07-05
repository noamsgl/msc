"""
plot density estimator highest density region (HDR)
# Hyndman, R. J. (1996). 
# Computing and Graphing Highest Density Regions. 
# The American Statistician, 50(2), 120–126.
# https://doi.org/10.2307/2684423
# https://www.jstor.org/stable/2684423
"""

from matplotlib.colors import SymLogNorm
from matplotlib.ticker import SymmetricalLogLocator

import numpy as np
import matplotlib.pyplot as plt

from msc import config

plt.style.use(["science", "no-latex"])


def plot_de_hdr(de, thresh, ax=None):
    """
    plot density estimator highest density region (HDR) above threshold `thresh`.
    """
    if ax is None:
        ax = plt.gca()

    # define x and y limits
    x = np.linspace(-10.0, 15.0, 200)
    y = np.linspace(-10.0, 15.0, 200)

    # create meshgrid
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = de.score_samples(XX)
    Z = Z.reshape(X.shape)

    # plot contour
    norm = SymLogNorm(vmin=np.min(Z), vmax=np.log10(0.5), linthresh=0.03)
    levels = -np.flip(np.logspace(0, np.log10(-np.min(Z)), 20))
    CS = ax.contour(X, Y, Z, norm=norm, cmap="viridis_r", levels=levels)

    CB = plt.colorbar(
        CS,
        location="right",
        ticks=SymmetricalLogLocator(linthresh=0.03, base=10, subs=range(10)),
    )
    CB.set_label("log-likelihood")

    # add axes annotations
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.savefig(
        f"{config['path']['figures']}/density_estimation/hdr.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    # TODO: pickup here
    plot_de_hdr(de, 0.05)
