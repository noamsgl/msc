from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import SymmetricalLogLocator

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import mixture
from sklearn.decomposition import PCA
from tqdm import tqdm

from msc import config
from msc.cache_handler import get_samples_df
from msc.plot_utils import set_size, square_size

plt.style.use(["science", "no-latex"])


def plot_pc1_pc2(width, n_dim=5):
    # get samples_df
    samples_df = get_samples_df(config["dataset_id"], with_wall_time=False, with_events=False)  # type: ignore

    # compute PCA
    pca = PCA(n_components=n_dim)
    X = np.stack(samples_df["embedding"])
    transformed_X = pca.fit_transform(X)  # type: ignore

    # fit a Gaussian Mixture Model with two components
    gmm = mixture.GaussianMixture(
        n_components=4, covariance_type="full", random_state=42
    )
    gmm.fit(transformed_X)

    # define grid extent and resolution
    start = -10.0
    stop = 10.0
    n_steps = 25
    x = np.linspace(start, stop, n_steps)
    step_size = (stop - start) / (n_steps - 1)
    coordinates = [x for _ in range(n_dim)]
    grid = np.meshgrid(*coordinates)

    # unravel the grid into a single 1D array
    XX = np.stack([g.ravel() for g in grid]).T

    # score all points on the grid
    Z = gmm.score_samples(XX)
    Z = Z.reshape(grid[0].shape)

    integrate_over_dims = tuple(range(2, n_dim))
    integrated_Z = scipy.special.logsumexp(Z, axis=integrate_over_dims)

    # set figure size
    fig = plt.figure(figsize=set_size(width, height_scale=1.4))

    norm = SymLogNorm(vmin=np.min(integrated_Z), vmax=np.log10(0.5), linthresh=0.03)
    levels = -np.flip(np.logspace(0, np.log10(-np.min(integrated_Z)), 20))

    # define plot grid extent
    plt_grid = np.meshgrid(*coordinates[0:2])
    CS = plt.contour(
        plt_grid[0],
        plt_grid[1],
        integrated_Z,
        norm=norm,
        cmap="viridis_r",
        levels=levels,
    )

    # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.SymmetricalLogLocator
    CB = plt.colorbar(
        CS,
        location="bottom",
        ticks=SymmetricalLogLocator(linthresh=0.03, base=10, subs=range(10)),
    )
    # CB.ax.minorticks_off()

    CB.set_label("marginal log-likelihood")

    # add axes annotations
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # add title
    # plt.title("log likelihood as predicted by GMM")
    # plt.axis("tight")

    # save fig
    # plt.show()
    print("saving figure...")
    fig_save_path = f"{config['path']['figures']}/density_estimation/de_pc1_pc2.pdf"
    plt.savefig(fig_save_path, bbox_inches="tight")


def plot_pc_pair_plot(width, n_dim=5):
    # get samples_df
    samples_df = get_samples_df(config["dataset_id"], with_wall_time=False, with_events=False)  # type: ignore

    # compute PCA
    pca = PCA(n_components=n_dim)
    X = np.stack(samples_df["embedding"])
    transformed_X = pca.fit_transform(X)  # type: ignore

    # fit a Gaussian Mixture Model with two components
    gmm = mixture.GaussianMixture(
        n_components=4, covariance_type="full", random_state=42
    )
    gmm.fit(transformed_X)

    # define grid extent and resolution
    start = -10.0
    stop = 10.0
    n_steps = 25
    x = np.linspace(start, stop, n_steps)
    step_size = (stop - start) / (n_steps - 1)
    coordinates = [x for _ in range(n_dim)]
    grid = np.meshgrid(*coordinates)

    # unravel the grid into a single 1D array
    XX = np.stack([g.ravel() for g in grid]).T

    # score all points on the grid
    Z = gmm.score_samples(XX)
    Z = Z.reshape(grid[0].shape)

    # create fig and axes
    fig, axes = plt.subplots(
        n_dim,
        n_dim,
        sharex=False,
        sharey=False,
        figsize=set_size(width, transposed=False),
        # figsize=square_size(width)
    )

    for i in tqdm(range(n_dim)):  # component 1
        for j in range(n_dim):  # component 2
            if i > j:             
                integrate_over_dims = tuple([d for d in range(n_dim) if d != i and d != j])
                integrated_Z = scipy.special.logsumexp(Z, axis=integrate_over_dims)

                norm = SymLogNorm(vmin=np.min(integrated_Z), vmax=np.log10(0.5), linthresh=0.03)
                levels = -np.flip(np.logspace(0, np.log10(-np.min(integrated_Z)), 20))

                # define plot grid extent
                plt_grid = np.meshgrid(*coordinates[0:2])
                CS = axes[i, j].contour(
                    plt_grid[0],
                    plt_grid[1],
                    integrated_Z,
                    norm=norm,
                    cmap="viridis_r",
                    levels=levels,
                )

                axes[i, j].set_xticks(x)
                axes[i, j].set_yticks(x)
                # # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.SymmetricalLogLocator
                # CB = plt.colorbar(
                #     CS,
                #     location="bottom",
                #     ticks=SymmetricalLogLocator(linthresh=0.03, base=10, subs=range(10)),
                # )
                # # CB.ax.minorticks_off()

                # CB.set_label("marginal log-likelihood")

                # add axes annotations

                    
                # axes[i, j].set_xlabel("PC1")
                # axes[i, j].set_ylabel("PC2")
            if i == j:
                axes[i, j].text(0.5, 0.5, f"PC{i+1}", ha="center", va="center")
                axes[i, j].set_frame_on(False)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                axes[i, j].set_frame_on(False)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                # axes[i, j].axis("off")
                pass

    for i in tqdm(range(n_dim)):  # component 1
        for j in range(n_dim):  # component 2
            if i == n_dim-1 and j != n_dim-1:
                axes[i, j].set_xlabel(f"PC{j + 1}")

            if j == 0 and i != 0:
                axes[i, j].set_ylabel(f"PC{i + 1}")
            # We change the fontsize of ticks label
            # axes[i, j].tick_params(axis='both', which='major', labelsize=8)
            # axes[i, j].tick_params(axis='both', which='minor', labelsize=6)

    plt.tight_layout()
    plt.savefig(
        f"{config['path']['figures']}/density_estimation/de_pc_pair_plot.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # plot_pc1_pc2(width=380.9)
    plot_pc_pair_plot(width=478)
