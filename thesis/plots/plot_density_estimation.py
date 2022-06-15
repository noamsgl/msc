from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter 
from matplotlib.ticker import SymmetricalLogLocator

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA

from msc import config
from msc.cache_handler import get_samples_df
from msc.plot_utils import set_size

def plot(width):
    plt.style.use(['science', 'no-latex'])

    # get samples_df
    samples_df = get_samples_df(config['dataset_id'], with_events=True)  # type: ignore

    # compute PCA and add to samples_df
    pca = PCA(n_components=2)
    components = pca.fit_transform(np.stack(samples_df['embedding']))  # type: ignore
    samples_df[['pca-2d-one', 'pca-2d-two']] = components

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=4, covariance_type="full")
    clf.fit(components)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-10.0, 15.0, 200)
    y = np.linspace(-10.0, 15.0, 200)

    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    # set figure size
    fig = plt.figure(figsize=set_size(width, height_scale=1.4))

    # set drawing limits
    plt.xlim(-7.26, 12.468)
    plt.ylim(-2.942, 11.104)

    norm = SymLogNorm(vmin=np.min(Z), vmax=np.log10(0.5), linthresh=0.03)
    levels = -np.flip(np.logspace(0, np.log10(-np.min(Z)), 20))
    CS = plt.contour(X, Y, Z, norm=norm, cmap='viridis_r', levels=levels)

    # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.SymmetricalLogLocator
    CB = plt.colorbar(CS, location="bottom",ticks = SymmetricalLogLocator(linthresh=0.03, base=10, subs=range(10)))
    # CB.ax.minorticks_off()

    CB.set_label("log-likelihood")
    # plt.scatter(X[:, 0], X[:, 1], 0.8)

    # add axes annotations
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # add title
    # plt.title("log likelihood as predicted by GMM")
    # plt.axis("tight")

    # save fig
    # plt.show()
    plt.savefig(f"{config['path']['figures']}/density_estimation/density_estimation.pdf", bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
        plot(width=380.9)