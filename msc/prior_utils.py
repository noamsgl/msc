from scipy.special import i0
import numpy as np
import pandas as pd


def vm_density(x, mu, k=1 / 0.6):
    """von mises density function over 24 hours"""
    omega = 2 * np.pi / 24
    return np.exp(k * np.cos(omega * (x - mu))) / (2 * np.pi * i0(k))


def get_events_df(events) -> pd.DataFrame:
    events_df = pd.DataFrame(events, columns=['datetime'])
    events_df['year'] = events_df['datetime'].dt.year
    events_df['month'] = events_df['datetime'].dt.month
    events_df['day'] = events_df['datetime'].dt.day
    events_df['hour'] = events_df['datetime'].dt.hour
    events_df['minute'] = events_df['datetime'].dt.minute
    events_df['second'] = events_df['datetime'].dt.second
    return events_df


def events_to_circadian_hist(events, N=24):
    """

    Parameters
    ----------
    events (as list of datetimes)
    N (number of bins)

    Returns
    -------

    """
    # use pandas datetime tools to get event hours
    events_df = get_events_df(events)
    # compute events circadian histogram
    circadian_hist = pd.cut(events_df.hour, N, labels=range(N)).value_counts().sort_index()
    return circadian_hist.to_numpy()
