import datetime
from functools import partial
from scipy.special import i0
import numpy as np
import pandas as pd

from .data_utils import get_dataset
from .time_utils import SEC, MIN, HOUR


class PercentileOfScore(object):
    """a vectorized implementation of stats.percentileofscore"""
    def __init__(self, aList):
        self.a = np.array( aList )
        self.a.sort()
        self.n = float(len(self.a))
        self.pct = self.__rank_searchsorted_list
    # end def __init__

    def __rank_searchsorted_list(self, score_list):
        adx = np.searchsorted(self.a, score_list, side='right')
        pct = []
        for idx in adx:
            # Python 2.x needs explicit type casting float(int)
            pct.append( (float(idx) / self.n) * 100.0 )

        return pct
    # end def _rank_searchsorted_list
# end class PercentileOfScore


def vm_density(x, mu, k=1 / 0.6):
    """von mises density function over 24 hours"""
    omega = 2 * np.pi / 24
    density_func = lambda x: np.exp(k * np.cos(omega * (x - mu))) / (2 * np.pi * i0(k))  # not normalized
    normalizing_constant = np.trapz(density_func(np.linspace(0,24,100)))
    return density_func(x) / normalizing_constant


def vm_mixture(x, circadian_hist): 
    mus = np.arange(24) + 0.5
    density_func = lambda x: sum([circadian_hist[i] * partial(vm_density, mu=mu)(x) for i, mu in enumerate(mus)])  # not normalized
    normalizing_constant = np.trapz(density_func(np.linspace(0, 24, 100)))
    return density_func(x) / normalizing_constant


def get_events_df(events) -> pd.DataFrame:
    events_df = pd.DataFrame(events, columns=['datetime'])
    events_df['year'] = events_df['datetime'].dt.year
    events_df['month'] = events_df['datetime'].dt.month
    events_df['day'] = events_df['datetime'].dt.day
    events_df['hour'] = events_df['datetime'].dt.hour
    events_df['minute'] = events_df['datetime'].dt.minute
    events_df['second'] = events_df['datetime'].dt.second
    return events_df


def get_events_df_from_config() -> pd.DataFrame:
    from msc import config
    SEC = 1e6
    
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
    return events_df


def event_times_to_circadian_hist(event_times: np.ndarray, N=24):
    """

    Parameters
    ----------
    events : array_like (representing seconds)
    N (number of bins)

    Returns
    -------

    """
    event_hours = np.floor_divide(event_times, HOUR)
    event_walltime_hour = np.mod(event_hours, 24)

    # compute events circadian histogram
    circadian_hist = np.histogram(event_walltime_hour, np.arange(25))[0]
    return circadian_hist

def events_df_to_circadian_hist(events_df, N=24):
    """

    Parameters
    ----------
    events_df : pandas.DataFrame
    N (number of bins)

    Returns
    -------

    """
    # compute events circadian histogram
    circadian_hist = events_df.hour.value_counts().sort_index().reindex(range(N), fill_value=0)
    circadian_hist = circadian_hist.to_numpy()
    assert len(circadian_hist) == N, f"error: circadian_hist has wrong length ({len(circadian_hist)=})"
    return circadian_hist
    
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
