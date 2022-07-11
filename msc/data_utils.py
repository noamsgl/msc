from dataclasses import dataclass
import numpy as np
from numpy.random import default_rng

from ieegpy.ieeg.auth import Session
from ieegpy.ieeg.dataset import Dataset

from .config_utils import get_authentication, config


def count_nans(data):
    """return count of nan entries"""
    return np.count_nonzero(np.isnan(data))

def prop_nans(data):
    """"return proportion of nan entries"""
    return count_nans(data) / data.size

class IEEGDataFactory:
    def __init__(self, dataset_id) -> None:
        self.dataset_id = dataset_id
        
    @classmethod 
    def get_dataset(cls, dataset_id) -> Dataset:
        username, password = get_authentication()
        with Session(username, password) as s:# start streaming session
            ds = s.open_dataset(dataset_id)  # open dataset stream
        return ds

def get_config_dataset():
    # get dataset from iEEG.org
    ds_id = config['dataset_id']
    ds = IEEGDataFactory.get_dataset(ds_id)
    return ds

def get_dataset(dataset_id):
    # get dataset from iEEG.org
    ds = IEEGDataFactory.get_dataset(dataset_id)
    return ds


def get_sample_times(N, mode, t_end=None) -> np.ndarray:
    """subsample uniformly N time points between 0 and t_max or from t_max to t_end"""
    rng = default_rng(config['random_seed'])  # type: ignore
    if mode == "offline":
        times = np.array(rng.integers(0, config['t_max'], size=N))  # type: ignore
    elif mode == "online":
        assert t_end is not None
        assert config['t_max'] <= t_end, f"error: {config['t_max']=} is not <= than {t_end=}"
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

def get_event_times(dataset_id):
    pass

@dataclass
class EvalData:
    train_X: np.ndarray
    train_events: np.ndarray
    test_X: np.ndarray
    test_times: np.ndarray
    test_y: np.ndarray
