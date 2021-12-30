import os
import sys
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Tuple, Union, Sequence, List

import mne
import numpy as np
import pandas as pd
import portion
from mne.io import Raw, BaseRaw
from numpy import ndarray
from pandas import Series, DataFrame
from portion import Interval
from tqdm import tqdm

from msc.config import get_config

mne.set_log_level(False)

ALL_CHANNELS = ('Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                'Fz', 'Cz', 'Pz', 'SP1', 'SP2', 'RS', 'T1', 'T2', 'EOG1', 'EOG2', 'EMG', 'ECG', 'PHO', 'CP1', 'CP2',
                'CP5', 'CP6', 'PO1', 'PO2', 'PO5', 'PO6')

COMMON_CHANNELS = (
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz',
    'T7', 'T8', 'P7', 'P8')


@dataclass
class PicksOptions:
    """Class for picks options"""
    all_channels: Tuple[str] = ALL_CHANNELS
    one_channel: Tuple[str] = ('C3',)
    two_channels: Tuple[str] = ('C3', 'C4')
    common_channels: Tuple[str] = COMMON_CHANNELS
    non_eeg_channels: Tuple[str] = ('EOG1', 'EOG2', 'EMG', 'ECG', 'PHO')
    eeg_channels: Tuple[str] = tuple(set(ALL_CHANNELS) - set(non_eeg_channels))


class EEGdata:
    def __init__(self, fpath: str, offset: float = 0, crop: int = 10,
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

    def get_data_as_array(self, T, fs, d, return_times=False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        raw = self.raw.crop(tmax=T).resample(fs)
        if return_times:
            data, times = raw.get_data(return_times=True)
            return data[:d], times
        else:
            return raw.get_data()[:d]


def load_raw_data(fpath: str, offset: float = 0, crop: int = 10, picks: Tuple[str] = None,
                  verbose: bool = False, lfreq=None, hfreq=None) -> Raw:
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
    raw = raw.pick(picks).crop(offset, offset + crop).filter(l_freq=lfreq, h_freq=hfreq)
    return raw


def load_tensor_dataset(fpath: str, train_length: int = 20,
                        test_length: int = 10, picks: Tuple[str] = None) -> dict:
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
    """Get the first index at which `times` (a sorted array of time values) crosses a given time t."""
    return np.argmax(times >= t)


def datasets(H, F, L, dt, offset, device, *, fpath: str, resample_sfreq, picks: Tuple[str]):
    """Deprecated
        #todo: remove
     """
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


def get_interval_from_raw(raw: Raw) -> Interval:
    """
    # todo: test this
    Return a portion.Interval with the end points equal to the beginning and end of the raw segment, respectively.
    Args:
        raw:

    Returns:

    """
    lower = raw.info["meas_date"].replace(tzinfo=None)
    upper = raw.info["meas_date"].replace(tzinfo=None) + timedelta(seconds=int(raw.times[-1]))
    return portion.closedopen(lower, upper)


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


def raw_to_array(raw, T, fs, d, return_times=False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    raw = raw.crop(tmax=T).resample(fs)
    if return_times:
        data, times = raw.get_data(return_times=True)
        return data[:d], times
    else:
        return raw.get_data()[:d]


def load_raw_from_data_row_and_interval(row: Series, interval: Interval) -> Raw:
    """
    Return a raw EEG from a data_index row and it's intersection with an interval
    Args:
        row:
        interval:

    Returns:

    """
    config = get_config()
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    # load raw data
    raw_path = f"{dataset_path}/{row['package']}/{row['patient']}/{row['admission']}/{row['recording']}/{row['fname']}"
    raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True)
    raw = raw.resample(config['TASK']['RESAMPLE'])
    # trim interval to intersection with data
    interval = interval.intersection(portion.closedopen(raw.info["meas_date"].replace(tzinfo=None),
                                                        raw.info["meas_date"].replace(tzinfo=None) + timedelta(
                                                            seconds=int(raw.times[-1]))))

    # crop raw data to interval
    start_time = (interval.lower - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    end_time = (interval.upper - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    raw = raw.crop(start_time, end_time)
    return raw


def get_raw_from_data_files(data_df: DataFrame, interval: Interval) -> Union[Raw, None]:
    """
    Get raw data from subset of data_index data_rows, and intersect the data with the given interval
    If len(data_df)==0, return None.
    Args:
        data_df:
        interval:

    Returns: Raw (if data_df is empty, return None)

    """
    if len(data_df) > 2:
        data_df = data_df.sort_values(by=['meas_date']).reset_index()
        raws = [load_raw_from_data_row_and_interval(data_df.loc[i], interval) for i in range(len(data_df))]
        raw = mne.concatenate_raws(raws)
        return raw

    if len(data_df) == 0:
        print(f"Requested interval {interval} has 0 data files, skipping...", file=sys.stderr)
        return None

    if len(data_df) == 2:
        data_df = data_df.sort_values(by=['meas_date']).reset_index()
        raw_1: Raw = load_raw_from_data_row_and_interval(data_df.loc[0], interval)
        raw_2: Raw = load_raw_from_data_row_and_interval(data_df.loc[1], interval)
        raw = mne.concatenate_raws([raw_1, raw_2])
        return raw

    if len(data_df) == 1:
        raw = load_raw_from_data_row_and_interval(data_df.reset_index().loc[0], interval)
        return raw


def get_overlapping_data_files(patient_data_df: DataFrame, interval: Interval) -> DataFrame:
    """
    Get a DataFrame of all data files which belong to patient, and overlap the interval
    Args:
        patient_data_df:
        interval:

    Returns:

    """

    def is_overlap(row):
        p = portion.closedopen(row['meas_date'], row['end_date'])
        return p.overlaps(interval)

    recordings_df = patient_data_df[patient_data_df.apply(is_overlap, axis=1)]
    return recordings_df


def get_raw_from_interval(patient_data_df, interval: Interval) -> Raw:
    """
    Return a raw EEG of a time interval
    Args:
        patient_data_df:

        interval: time interval

    Returns: raw

    """
    recordings_df = get_overlapping_data_files(patient_data_df, interval)

    raw = get_raw_from_data_files(recordings_df, interval)
    return raw


def get_raws_from_data_and_intervals(patient_data_df: DataFrame, picks, intervals, fast_dev_mode: bool = False):
    """
    # todo: implement this
    Returns a list of raws which correspond to the given intervals which occur completely during a single data file
    Args:
        picks:
        patient_data_df:
        intervals:
        fast_dev_mode:

    Returns:

    """
    raise NotImplementedError("under revision")
    config = get_config()
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    raws = []
    patient_data_df_rows = list(patient_data_df.iterrows())
    # if fast_dev_mode:
    #     patient_data_df_rows = patient_data_df_rows[:1]
    for idx, row in tqdm(patient_data_df_rows, desc="loading patient data"):

        raw_interval = portion.closedopen(row["meas_date"], row["end_date"])

        intervals_during_raw = [interval for interval in intervals if
                                interval in raw_interval]
        if len(intervals_during_raw) == 0:
            continue

        # load raw data file
        raw_path = f"{dataset_path}/{row['package']}/{row['patient']}/{row['admission']}/{row['recording']}/{row['fname']}"
        raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True)
        picks = [p for p in picks if p in raw.info["ch_names"]]
        raw = raw.pick(picks)
        raw = raw.resample(config['TASK']['RESAMPLE'])


        for interval in intervals_during_raw:
            # crop raw data to interval
            start_time = (interval.lower - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
            end_time = (interval.upper - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
            raw_interval = raw.copy().crop(start_time, end_time)
            raws.append(raw_interval)

    return raws


def get_patient_data_index(patient: str) -> DataFrame:
    """
    Return a DataFrame of patient's data index
    Args:
        patient: patient name

    Returns:

    """
    config = get_config()
    package = get_package_from_patient(patient)

    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    data_index_path = f"{dataset_path}/data_index.csv"

    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]
    return patient_data_df


def get_raws_from_intervals(package: str, patient: str, intervals: Sequence[Interval]) -> List[Raw]:
    """
    Return a list of raw EEGs from a list of time intervals
    Args:
        package: package name
        patient: patient name
        intervals: time intervals

    Returns:

    """
    config = get_config()
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    data_index_path = f"{dataset_path}/data_index.csv"

    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    raws = [get_raw_from_interval(patient_data_df, interval) for interval in intervals if interval != portion.empty()]
    return [raw for raw in raws if raw]  # remove None values


def get_package_from_patient(patient: str) -> str:
    config = get_config()
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    patients_index_path = f"{dataset_path}/patients_index.csv"
    patients_df = pd.read_csv(patients_index_path)
    patient_row = patients_df.loc[patients_df['pat_id'] == patient, 'package']
    assert len(patient_row) == 1, "check patient in patients_index.csv because patient was not found exactly once"
    package = patient_row.item()
    return package


def get_time_as_str(fmt=None):
    iso_8601_format = '%Y%m%dT%H%M%S'  # e.g., 20211119T221000
    if fmt is None:
        fmt = iso_8601_format
    return datetime.now().strftime(fmt)
