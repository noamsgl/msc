import os
from dataclasses import dataclass
from typing import Tuple, Union, Sequence, List

import mne
import numpy as np
import pandas as pd
import portion
import torch
from mne.io import Raw, BaseRaw
from numpy.typing import NDArray
from pandas import Series, DataFrame
from portion import Interval

from msc.config import get_config
from msc.data_utils.download import download_file_scp

mne.set_log_level(False)

ALL_CHANNELS = ('FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                'FZ', 'CZ', 'PZ', 'SP1', 'SP2', 'RS', 'T1', 'T2', 'EOG1', 'EOG2', 'EMG', 'ECG', 'PHO', 'CP1', 'CP2',
                'CP5', 'CP6', 'PO1', 'PO2', 'PO5', 'PO6')

COMMON_CHANNELS = (
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
    'T7', 'T8', 'P7', 'P8')

# read local file `config.ini`
config = get_config()

# FPATH = r'C:\raw_data\surf30\pat_103002\adm_1030102\rec_103001102\103001102_0113.data'
FPATH = config.get('DATA', 'RAW_PATH')

# LFREQ = 0.5  # low frequency for highpass filter
# HFREQ = 5  # high frequency for lowpass filter
LFREQ = None  # low frequency for highpass filter
HFREQ = None  # high frequency for lowpass filter
CROP = 10  # segment length measured in seconds
TRAIN_LENGTH = 10  # segment length measured in seconds
TEST_LENGTH = 2  # segment length measured in seconds
RESAMPLE = None


@dataclass
class PicksOptions:
    """Class for picks options"""
    all_channels: Tuple[str] = ALL_CHANNELS
    one_channel: Tuple[str] = ('C3',)
    two_channels: Tuple[str] = ('C3', 'C4')
    common_channels: Tuple[str] = COMMON_CHANNELS
    non_eeg_channels: Tuple[str] = ('EOG1', 'EOG2', 'EMG', 'ECG', 'PHO')
    eeg_channels: Tuple[str] = tuple(set(ALL_CHANNELS) - set(non_eeg_channels))


PICKS = PicksOptions.one_channel


class EEGdata:
    def __init__(self, fpath: str = FPATH, offset: float = 0, crop: int = CROP,
                 picks: Tuple[str] = PicksOptions.common_channels, verbose: bool = False, l_freq=None, h_freq=None):
        self.fpath = fpath
        self.offset = offset
        self.crop = crop
        self.picks = picks
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.raw: BaseRaw = self._init_raw_data()
        # set verbosity
        self.verbose = verbose
        mne.set_log_level(verbose)

    def _init_raw_data(self):
        """
        load raw data in nicolet (.data & .head) format
        Returns:

        """
        extension = os.path.splitext(self.fpath)[1]
        assert extension == '.data', "unknown extension. currently supports .data"

        raw: Raw = mne.io.read_raw_nicolet(self.fpath, ch_type='eeg', preload=True)
        raw = raw.pick(self.picks).crop(self.offset, self.offset + self.crop).filter(self.l_freq, self.h_freq)
        return raw

    def get_data_as_array(self, T, fs, d, return_times=False) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        raw = self.raw.crop(tmax=T).resample(fs)
        if return_times:
            data, times = raw.get_data(return_times=True)
            return data[:d], times
        else:
            return raw.get_data()[:d]


def load_raw_data(fpath: str = FPATH, offset: float = 0, crop: int = CROP, picks: Tuple[str] = PICKS,
                  verbose: bool = False) -> Raw:
    """ get sample data for testing

    Args:
        fpath: path to load data from
        crop: time length in seconds to crop
        picks: channels to select

    Returns:
        Tensor: test data
    """
    mne.set_log_level(verbose)
    if verbose:
        print(f"picks: {picks}")
    extension = os.path.splitext(fpath)[1]
    if extension == '.data':
        # file is stored in raw nicolet format
        raw: Raw = mne.io.read_raw_nicolet(fpath, ch_type='eeg', preload=True)
    elif extension == ".edf":
        raw = mne.io.read_raw_edf(fpath, picks=picks, preload=True)
    else:
        raise ValueError("unknown extension. currently supports: .data, .edf")

    # crop and filter raw
    raw = raw.pick(picks).crop(offset, offset + crop).filter(l_freq=LFREQ, h_freq=HFREQ)
    return raw


def load_tensor_dataset(fpath: str = FPATH, train_length: int = TRAIN_LENGTH,
                        test_length: int = TEST_LENGTH, picks: Tuple[str] = PICKS) -> dict:
    """ Returns a standardized dataset

    Args:
        fpath: filepath where .data and .head are located
        train_length: time length in seconds
        test_length: time length in seconds
        picks: list of EEG channels to select from file

    Returns: dict which includes train_x, train_y, test_x, test_y

    """
    raw: Raw = load_raw_data(fpath, 0, train_length + test_length, picks)
    data, times = raw.get_data(return_times=True)

    data = (data - np.mean(data)) / np.std(data)

    split_index = np.argmax(times > train_length)

    dataset = {"train_x": torch.Tensor(times[:split_index]),
               "train_y": torch.Tensor(data[:, :split_index]).T.squeeze(),
               "test_x": torch.Tensor(times[split_index:]),
               "test_y": torch.Tensor(data[:, split_index:]).T.squeeze()
               }
    return dataset


def get_index_from_time(t, times):
    return np.argmax(times >= t)


def datasets(H, F, L, dt, offset, device, *, fpath: str = FPATH, resample_sfreq=RESAMPLE, picks: Tuple[str] = PICKS):
    raw: Raw = mne.io.read_raw_nicolet(fpath, ch_type='eeg', preload=True).pick(picks)
    raw = raw.resample(resample_sfreq)

    sfreq = int(raw.info['sfreq'])
    data, times = raw.get_data(return_times=True)
    data = (data - np.mean(data)) / np.std(data)

    t_start = offset + H
    t_stop = t_start + L
    t_start_idx = get_index_from_time(t_start, times)
    t_stop_idx = get_index_from_time(t_stop, times)

    stepsize = int(dt * sfreq)
    for t in times[t_start_idx:t_stop_idx:stepsize]:
        print(f"serving dataset at time {t} seconds")
        t_ix = get_index_from_time(t, times)
        train_x = torch.tensor(times[int(t_ix - H * sfreq):t_ix], device=device).float()
        train_y = torch.tensor(data[:, int(t_ix - H * sfreq):t_ix], device=device).float().T.squeeze()
        test_x = torch.tensor(times[t_ix:int(t_ix + F * sfreq)], device=device).float()
        test_y = torch.tensor(data[:, t_ix:int(t_ix + F * sfreq)], device=device).float().T.squeeze()
        yield t, train_x, train_y, test_x, test_y


def load_raw_seizure(package="surf30", patient="pat_92102", seizure_num=3, delta=0.0) -> Raw:
    """
    Return a mne.io.Raw EEG data of the seizure based on the EPILEPSIAE seizure index.
    Args:
        package:
        patient:
        seizure_num:
        delta: time (in percentage of seizure length) of EEG before and after a seizure to return

    Returns:

    """
    import pandas as pd
    import re

    # get seizure index
    config = get_config()
    seizures_index_path = r"C:\raw_data/epilepsiae/seizures_index.csv"
    df = pd.read_csv(seizures_index_path, index_col=0, parse_dates=['onset', 'offset'])

    # filter out seizure row
    seizure_row = df.loc[(df["package"] == package) & (df["patient"] == patient) & (df["seizure_num"] == seizure_num)]

    # get seizure file path
    remote_fpath = seizure_row["onset_fpath"].item()[3:-2]  # extract path from string
    datasets_path, dataset, relative_file_path = re.split('(epilepsiae/)', remote_fpath)
    local_path = download_file_scp(relative_file_path)

    # get seizure times
    onset = seizure_row["onset"].item()
    offset = seizure_row["offset"].item()

    # load and crop seizure data
    raw = mne.io.read_raw_nicolet(local_path, ch_type='eeg', preload=True)
    start_time = (onset - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    end_time = (offset - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    seizure_length = end_time - start_time
    raw = raw.crop(start_time - delta * seizure_length, end_time + delta * seizure_length)
    return raw


def raw_to_array(raw, T, fs, d, return_times=False) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    raw = raw.crop(tmax=T).resample(fs)
    if return_times:
        data, times = raw.get_data(return_times=True)
        return data[:d], times
    else:
        return raw.get_data()[:d]


def load_raw_from_data_series(row: Series, interval: Interval):
    dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}"
    # load raw data
    raw_path = f"{dataset_path}/{row['package']}/{row['patient']}/{row['admission']}/{row['recording']}/{row['fname']}"
    raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True)
    # trim interval to intersection with data
    interval = interval.intersection(portion.closed(raw.info["meas_date"], raw.info["end_date"]))
    # crop raw data to interval
    start_time = (interval.lower - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    end_time = (interval.upper - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    raw = raw.crop(start_time, end_time)
    return raw


def get_raw_from_data_files(data_df: DataFrame, interval: Interval) -> Raw:
    if len(data_df) > 2:
        raise ValueError("Interval crosses more than 2 data files")

    if len(data_df) == 0:
        raise ValueError("Requested interval with 0 data files")

    if len(data_df) == 2:
        data_df = data_df.sort_values(by=['meas_date']).reset_index()
        raw_1: Raw = load_raw_from_data_series(data_df.loc[0], interval)
        raw_2: Raw = load_raw_from_data_series(data_df.loc[1], interval)
        raw = mne.concatenate_raws([raw_1, raw_2])
        return raw

    if len(data_df) == 1:
        raw = load_raw_from_data_series(data_df.reset_index().loc[0], interval)
        return raw


def get_overlapping_data_files(package: str, patient: str, interval: Interval) -> DataFrame:
    dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}"
    data_index_path = f"{dataset_path}/data_index.csv"

    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    begins_in = lambda x: x['meas_date'] in interval
    ends_in = lambda x: x['end_date'] in interval

    def is_overlap(row):
        p = portion.closed(row['meas_date'], row['end_date'])
        return p.overlaps(interval)

    recordings_df = patient_data_df[patient_data_df.apply(is_overlap, axis=1)]
    return recordings_df


def get_raw_from_interval(package: str, patient: str, interval: Interval) -> Raw:
    """
    Return a raw EEG of a time interval
    Args:
        package: package name
        patient: patient name
        interval: time interval

    Returns: raw

    """
    recordings_df = get_overlapping_data_files(package, patient, interval)
    raw = get_raw_from_data_files(recordings_df, interval)
    return raw


def get_raws_from_intervals(package: str, patient: str, intervals: Sequence[Interval]) -> List[Raw]:
    """
    Return a list of raw EEGs from a list of time intervals
    Args:
        package: package name
        patient: patient name
        intervals: time intervals

    Returns:

    """
    return [get_raw_from_interval(package, patient, interval) for interval in intervals]
