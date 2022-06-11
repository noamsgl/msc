import datetime
from functools import partial
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import i0

from msc import config
from msc.datamodules.data_utils import IEEGDataFactory
from msc.plot_utils import set_size

plt.style.use(['science', 'no-latex'])

SEC = 1e6
MIN = 60 * SEC
HOUR = 60 * MIN

def get_dataset(dataset_id):
    # get dataset from iEEG.org
    ds = IEEGDataFactory.get_dataset(dataset_id)
    return ds
    
def karoly_prior(t):
    ds = get_dataset(config['dataset_id'])

def get_events_df(events) -> pd.DataFrame:
    events_df = pd.DataFrame(events, columns=['datetime'])
    events_df['year'] = events_df['datetime'].dt.year
    events_df['month'] = events_df['datetime'].dt.month
    events_df['day'] = events_df['datetime'].dt.day
    events_df['hour'] = events_df['datetime'].dt.hour
    events_df['minute'] = events_df['datetime'].dt.minute
    events_df['second'] = events_df['datetime'].dt.second
    return events_df


def plot_bar_histogram(width):
    plt.clf()
    # prepare figure
    fig = plt.figure(figsize=set_size(width))

    plt.bar(range(N), circadian_hist, align='edge')
    ax = plt.gca()
    ax.set_facecolor('lightyellow')
    plt.grid()
    # add base rate
    plt.plot(np.linspace(0, N, 100), np.ones(100)*sum(circadian_hist)/N, color='darkorange', linestyle='dashed', lw=2, label='base rate')
    plt.xlim(0, N)
    plt.title("Circadian Seizure Distribution")
    plt.savefig(f"{config['path']['figures']}/prior/hist.pdf", bbox_inches='tight')


def plot_polar_histogram(fig_width):
    plt.clf()

    # prepare figure
    fig = plt.figure(figsize=set_size(fig_width))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar', facecolor='white')

    # initialize seizure distribution as periodical
    half_hour_offset = (1/2) * (1/N) * 2 * np.pi
    theta = np.linspace(0.0 + half_hour_offset, 2 * np.pi + half_hour_offset, N, endpoint=False)
    radii = circadian_hist
    width = 2 * np.pi / N

    # define axis
    ax = plt.subplot(111, projection='polar', facecolor='lightyellow')

    # plot seizures
    bars = ax.bar(theta, radii, width=width, bottom=0.0, label='events per hour', color='royalblue')

    # make the labels start at North
    ax.set_theta_zero_location('N')

    # make the labels go clockwise
    ax.set_theta_direction(-1)

    # clock labels
    ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
    ax.set_xticklabels(range(24))

    # add base rate
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*sum(circadian_hist)/N, color='darkorange', linestyle='dashed', lw=2, label='base rate')

    # add legend
    ax.legend(fancybox=True, bbox_to_anchor=(0.5, -0.05))
    # plt.title("Circadian Seizure Distribution")
    plt.savefig(f"{config['path']['figures']}/prior/polar_hist.pdf", bbox_inches='tight')


def vmdensity(x, mu=0, k=0.6):
    """von mises density function"""
    omega = 2 * np.pi/N
    return np.exp(k * np.cos(omega * (x - mu)))/(2 * np.pi * i0(k))

def plot_vmdensity(width, mu=8, k=0.6):
    X = np.linspace(0, 24, 100)
    y = vmdensity(X, mu=mu, k=k)

    # vm_mixture = lambda x: sum([circadian_hist[i] * partial(vmdensity, mu=mu)(x) for i, mu in enumerate(mus)])
    
    # prepare figure
    fig = plt.figure(figsize=set_size(width))
    plt.plot(X, y, label='density $f(x | \mu, \kappa ; \omega)$')
    plt.vlines([mu], min(y), max(y), linestyles='dashed', colors='orange', lw=2, label=f'$\mu={mu}$')
    # plt.plot(y)
    plt.xlabel('x')
    extraticks = [mu, N]
    plt.xticks([0, 6, 12, 18, 24] + extraticks)
    plt.xlim(0, N)
    plt.legend()
    plt.savefig(f"{config['path']['figures']}/prior/vm_density.pdf", bbox_inches='tight')


def plot_von_mises_prior(width):
    X = np.linspace(0, 24, 100)
    y = vmdensity(X)
    weights = circadian_hist
    mus = np.arange(N) + 0.5

    vm_mixture = lambda x: sum([circadian_hist[i] * partial(vmdensity, mu=mu)(x) for i, mu in enumerate(mus)])
    
    # prepare figure
    fig = plt.figure(figsize=set_size(width))
    plt.plot(X, y)
    # plt.plot(y)
    plt.xlabel('walltime')

    plt.savefig(f"{config['path']['figures']}/prior/vm_prior.pdf", bbox_inches='tight')


if __name__ == "__main__":
    ds = get_dataset(config['dataset_id'])
    start_time = datetime.datetime.fromtimestamp(ds.start_time / SEC, datetime.timezone.utc)

    seizures = ds.get_annotations('seizures')
    events =  [start_time + datetime.timedelta(microseconds=seizure.start_time_offset_usec) for seizure in seizures]
    
    events_df = get_events_df(events)
    circadian_hist = pd.cut(events_df.hour, 24, labels=range(24)).value_counts().sort_index()
    N = 24

    width = 478  #pt
    # plot_bar_histogram(width)
    # plot_polar_histogram(width)
    plot_vmdensity(width)
    plot_von_mises_prior(width)
