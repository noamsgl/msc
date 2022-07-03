import datetime
from functools import partial
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import i0

from msc import config
from msc.data_utils import get_dataset
from msc.plot_utils import set_size
from msc.prior_utils import get_events_df, vm_density, vm_mixture

plt.style.use(["science", "no-latex"])

SEC = 2 * 1e6
MIN = 60 * SEC
HOUR = 60 * MIN


def plot_bar_histogram(width):
    # prepare figure
    fig = plt.figure(figsize=set_size(width))

    plt.bar(range(N), circadian_hist, align="edge", color="royalblue", label="data")
    ax = plt.gca()
    ax.set_facecolor("lightyellow")
    plt.grid()
    # add base rate
    plt.plot(
        np.linspace(0, N, 100),
        np.ones(100) * sum(circadian_hist) / N,
        color="darkorange",
        linestyle="dashed",
        lw=2,
        label="base rate",
    )
    plt.xlim(0, N)
    plt.title("Circadian Seizure Distribution")
    plt.savefig(f"{config['path']['figures']}/prior/hist.pdf", bbox_inches="tight")
    return plt.gcf()


def plot_polar_histogram(fig_width):
    # prepare figure
    fig = plt.figure(figsize=set_size(fig_width))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar", facecolor="white")

    # initialize seizure distribution as periodical
    half_hour_offset = (1 / 2) * (1 / N) * 2 * np.pi
    theta = np.linspace(
        0.0 + half_hour_offset, 2 * np.pi + half_hour_offset, N, endpoint=False
    )
    radii = circadian_hist
    width = 2 * np.pi / N

    # define axis
    ax = plt.subplot(111, projection="polar", facecolor="lightyellow")

    # plot seizures
    bars = ax.bar(
        theta,
        radii,
        width=width,
        bottom=0.0,
        label="events per hour",
        color="royalblue",
    )

    # make the labels start at North
    ax.set_theta_zero_location("N")

    # make the labels go clockwise
    ax.set_theta_direction(-1)

    # clock labels
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels(range(24))

    # add base rate
    ax.plot(
        np.linspace(0, 2 * np.pi, 100),
        np.ones(100) * sum(circadian_hist) / N,
        color="darkorange",
        linestyle="dashed",
        lw=2,
        label="base rate",
    )

    # add legend
    ax.legend(fancybox=True, bbox_to_anchor=(0.5, -0.05))
    # plt.title("Circadian Seizure Distribution")
    plt.savefig(
        f"{config['path']['figures']}/prior/polar_hist.pdf", bbox_inches="tight"
    )


def plot_vmdensity(width, mu=8, k=1 / 0.6):
    X = np.linspace(0, N, 100)
    y = vm_density(X, mu=mu, k=k)

    # prepare figure
    fig = plt.figure(figsize=set_size(width))
    plt.plot(X, y, label="density $f(x | \mu, \kappa ; \omega)$")
    plt.fill_between(X, y, color="lightyellow")
    plt.vlines(
        [mu], 0, max(y), linestyles="dashed", colors="orange", lw=2, label=f"$\mu={mu}$"
    )
    plt.xlabel("x")
    extraticks = [mu, N]
    plt.xticks([0, 6, 12, 18, 24] + extraticks)
    plt.xlim(0, N)
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(
        f"{config['path']['figures']}/prior/vm_density.pdf", bbox_inches="tight"
    )


def plot_von_mises_prior(width, fig=None):
    # prepare figure
    if fig is None:
        fig = plt.figure(figsize=set_size(width))
    else:
        plt.figure(fig)
    X = np.linspace(0, 24, 100)
    y = vm_mixture(X, circadian_hist)
    y = 5 * y / max(y)  # TODO: make this same height as bar chart
    plt.plot(X, y, label="v.M. prior (not at scale)")
    # plt.plot(y)
    # ax.set_xlabel('walltime')
    legend = plt.legend(loc="upper right", framealpha=1)
    # TODO: fix facecolor to white
    frame = legend.get_frame()
    frame.set_facecolor("white")
    plt.xlabel("time")
    plt.title("")
    fig.savefig(f"{config['path']['figures']}/prior/vm_prior.pdf", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # define number of bins
    N = 24
    # get dataset
    ds = get_dataset(config["dataset_id"])
    # get dataset's start time
    start_time = datetime.datetime.fromtimestamp(
        ds.start_time / SEC, datetime.timezone.utc
    )
    # get dataset's seizure annotations
    seizures = ds.get_annotations("seizures")
    # convert annotations to event datetimes
    events = [
        start_time + datetime.timedelta(microseconds=seizure.start_time_offset_usec)
        for seizure in seizures
    ]
    # create events_df
    events_df = get_events_df(events)
    # compute events circadian histogram
    circadian_hist = (
        pd.cut(events_df.hour, N, labels=range(N)).value_counts().sort_index()
    )

    width = 478  # pt
    fig = plot_bar_histogram(width)
    # plot_polar_histogram(width)
    plot_vmdensity(width)
    plot_von_mises_prior(width, fig)
