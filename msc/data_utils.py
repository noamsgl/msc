import numpy as np
from numpy.random import default_rng

from msc import config
from .datamodules.data_utils import IEEGDataFactory

def count_nans(data):
    """return count of nan entries"""
    return np.count_nonzero(np.isnan(data))

def prop_nans(data):
    """"return proportion of nan entries"""
    return count_nans(data) / data.size

def get_config_dataset():
    # get dataset from iEEG.org
    ds_id = config['dataset_id']
    ds = IEEGDataFactory.get_dataset(ds_id)
    return ds

def get_sample_times(N, mode, t_end=None) -> np.ndarray:
    """subsample uniformly N time points between 0 and t_max or from t_max to t_end"""
    rng = default_rng(config['random_seed'])  # type: ignore
    if mode == "offline":
        times = np.array(rng.integers(0, config['t_max'], size=N))  # type: ignore
    elif mode == "online":
        assert t_end is not None
        times = np.array(rng.integers(config['t_max'], t_end, size=N))  # type: ignore
    else:
        raise ValueError(f"{mode=} is unsupported")
    return times

def get_event_sample_times(ds, augment=False) -> np.ndarray:
    """get event times, possibly augment with pre-event times"""
    seizures = ds.get_annotations('seizures')
    seizure_onsets_usec = np.array([seizure.start_time_offset_usec for seizure in seizures])
    # convert usec to sec
    times = seizure_onsets_usec / 1e6
    # append pre-ictal segments
    if augment:
        times = np.concatenate([times, times-5, times-10, times-15, times-20, times-25, times-30])
    return times.astype(int)