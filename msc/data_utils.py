import itertools
import os
import sys
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Tuple, Union, Sequence, List

import mne
import mne_features
import numpy as np
import pandas as pd
import portion
import portion as P
from mne.io import Raw
from numpy import ndarray
from pandas import Series, DataFrame
from portion import Interval
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from msc import config

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


def get_index_from_time(t, times):
    """Get the first index at which `times` (a sorted array of time values) crosses a given time t."""
    return np.argmax(times >= t)


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
    raise NotImplementedError("needs refactoring")
    # import re
    # # get seizure index
    # seizures_index_path = r"C:\raw_data/epilepsiae/seizures_index.csv"
    # df = pd.read_csv(seizures_index_path, index_col=0, parse_dates=['onset', 'offset'])
    #
    # # filter out seizure row
    # seizure_row = df.loc[(df["package"] == package) & (df["patient"] == patient) & (df["seizure_num"] == seizure_num)]
    #
    # # get seizure file path
    # remote_fpath = seizure_row["onset_fpath"].item()[3:-2]  # extract path from string
    # datasets_path, dataset, relative_file_path = re.split('(epilepsiae/)', remote_fpath)
    # local_path = download_file_scp(relative_file_path)
    #
    # # get seizure times
    # onset = seizure_row["onset"].item()
    # offset = seizure_row["offset"].item()
    #
    # # load and crop seizure data
    # raw = mne.io.read_raw_nicolet(local_path, ch_type='eeg', preload=True)
    # start_time = (onset - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    # end_time = (offset - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
    # seizure_length = end_time - start_time
    # raw = raw.crop(start_time - delta * seizure_length, end_time + delta * seizure_length)
    # return raw


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


def add_raws_to_intervals_df(intervals_df: DataFrame, picks, fast_dev_mode=False) -> DataFrame:
    """
    Returns an expanded dataframe with raws which correspond to the given intervals
    Args:
        intervals_df:
        picks:
        fast_dev_mode:

    Returns:

    """
    fast_dev_counter = itertools.count()

    from msc.dataset import get_data_index_df
    data_index_df = get_data_index_df()
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"

    for patient, intervals in intervals_df.groupby('patient_name'):
        patient_data_index = data_index_df.loc[data_index_df.patient == patient]
        data_index_df_rows = list(patient_data_index.iterrows())
        for data_idx, data_row in tqdm(data_index_df_rows, desc=f"loading data files for patient {patient}"):
            row_interval = portion.closedopen(data_row["meas_date"], data_row["end_date"])

            # add a column specifying whether the seizure interval is in the data row
            intervals_df['in_data_file'] = intervals_df.window_interval.apply(lambda interval: interval in row_interval)

            if not intervals_df['in_data_file'].any():
                # don't load file if no intervals are requested from it
                continue

            # fast dev mode break after 3 data files
            counter_id = next(fast_dev_counter)
            if fast_dev_mode and counter_id > 1:
                break

            # load raw data file
            raw_path = f"{dataset_path}/{data_row['package']}/{data_row['patient']}/{data_row['admission']}/{data_row['recording']}/{data_row['fname']}"
            print(f"reading file {raw_path}")
            raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True)
            picks = [p for p in picks if p in raw.info["ch_names"]]
            raw = raw.pick(picks)
            raw = raw.resample(config['TASK']['RESAMPLE'])

            def add_raw_intervals(interval_row):
                # crop raw data to interval
                start_time = (interval_row.lower - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
                end_time = (interval_row.upper - raw.info["meas_date"].replace(tzinfo=None)).total_seconds()
                raw_interval = raw.copy().crop(start_time, end_time)
                return raw_interval

            # add raw window_interval
            intervals_df.loc[intervals_df['in_data_file'], "raw"] = intervals_df[
                intervals_df['in_data_file']].window_interval.apply(add_raw_intervals)

    return intervals_df.drop(columns='in_data_file')


def get_patient_data_index(patient: str) -> DataFrame:
    """
    Return a DataFrame of patient's data index
    Args:
        patient: patient name

    Returns:

    """
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
    dataset_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}"
    data_index_path = f"{dataset_path}/data_index.csv"

    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    raws = [get_raw_from_interval(patient_data_df, interval) for interval in intervals if interval != portion.empty()]
    return [raw for raw in raws if raw]  # remove None values


def get_package_from_patient(patient: str) -> str:
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


def get_recording_start(patient: str) -> datetime:
    """
    Get first measurement timestamp for patient from data_index.csv
    Args:
        patient:

    Returns:

    """
    from msc.dataset import get_data_index_df
    data_index_df = get_data_index_df()

    patient_data_df = data_index_df.loc[data_index_df['patient'] == patient]
    assert len(patient_data_df) > 0, f"Error: no data files for {patient=} found"
    return min(patient_data_df.meas_date)


def get_recording_end(patient: str) -> datetime:
    """
    Get last measurement timestamp for the patient from data_index.csv
    Args:
        patient:

    Returns:

    """
    data_index_path = f"{config['PATH'][config['INDEX_MACHINE']]['RAW_DATASET']}/data_index.csv"
    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[data_index_df['patient'] == patient]

    return max(patient_data_df.end_date)


def get_interictal_intervals(patient_name: str) -> DataFrame:
    """
    return interictal time intervals

    Args:
        patient_name: | example "pat_4000"

    Returns:

    """
    min_diff = timedelta(hours=float(config['TASK']['INTERICTAL_MIN_DIFF_HOURS']))
    recording_start = get_recording_start(patient_name)
    recording_end = get_recording_end(patient_name)

    onsets = get_seiz_onsets(patient_name)

    first_interictal = P.open(recording_start, onsets[0] - min_diff)
    middle_interictals = [P.open(onsets[i] + min_diff, onsets[i + 1] - min_diff) for i in range(0, len(onsets) - 1)]
    last_interictal = P.open(onsets[-1] + min_diff, recording_end)
    interictals = [first_interictal] + middle_interictals + [last_interictal]
    return DataFrame({"interval": interictals, "label_desc": "interictal"})


def get_preictal_intervals(patient_name: str) -> DataFrame:
    """
    return preictal time intervals

    Args:
        patient_name: | example "pat_4000"

    Returns:

    """
    onsets = get_seiz_onsets(patient_name)
    preictals = [P.open(onset - timedelta(hours=float(config['TASK']['PREICTAL_MIN_DIFF_HOURS'])), onset) for onset in
                 onsets]
    return DataFrame({"interval": preictals, "label_desc": "preictal"})


def get_seiz_onsets(patient_name: str) -> List[datetime]:
    """
    returns seizure onset times.
    Args:
        patient_name: | example "pat_4000"

    Returns:

    """
    seizures_index_path = f"{config['PATH'][config['INDEX_MACHINE']]['RAW_DATASET']}/seizures_index.csv"

    seizures_index_df = pd.read_csv(seizures_index_path, parse_dates=['onset', 'offset'])

    patient_seizures_df = seizures_index_df.loc[seizures_index_df['patient'] == patient_name]

    return list(patient_seizures_df.onset)


def standardize(X: ndarray) -> ndarray:
    """
    shift and scale X to 0 mean and 1 std
    Args:
        X:

    Returns:

    """
    X = X - np.mean(X)
    X = X / np.std(X)
    return X


def cross_correlation(chan_1, chan_2, Fs, tau):
    """
    Compute the cross correlation between two channels.
    It is a linear measure of dependence between two signals, that also
    allows fixed delays between two spatially distant EEG signals to accommodate
    potential signal propagation.
    Args:
        chan_1:
        chan_2:
        Fs: signal sample frequency
        tau:

    Returns:
    """
    assert len(chan_1) == len(chan_2)
    N = len(chan_1)
    if tau < 0:
        return cross_correlation(chan_2, chan_1, Fs, -tau)
    else:
        cc = 0
        for t in range(1, N - Fs * tau):
            cc += chan_1[t + Fs * tau] * chan_2[Fs * tau]
        cc = cc / (N - Fs * tau)
        return cc


def maximal_cross_correlation(chan_1: ndarray, chan_2: ndarray, Fs, tau_min=-0.5, tau_max=0.5) -> float:
    """Return the maximal cross correlation between two channels.
    It is a linear measure of dependence between two signals, that also
    allows fixed delays between two spatially distant EEG signals to accommodate
    potential signal propagation.

    Mormann, Florian, et al. "On the predictability of epileptic seizures."
    Clinical neurophysiology 116.3 (2005): 569-587.


    Args:
        chan_1: first EEG channel
        chan_2: second EEG channel
        Fs: signal sample frequency
        tau_min: bottom range of taus to check (s)
        tau_max: top rang of taus to check (s)

    Returns:
    """
    taus = np.linspace(tau_min, tau_max, num=50, endpoint=True)
    return np.max([cross_correlation(chan_1, chan_2, tau, Fs) for tau in taus])


class SynchronicityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_code):
        assert feature_code in ["C", "S", "DSTL", "SPLV", "H", "Coh"], "feature_code invalid, check paper"
        self.feature_code = feature_code

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        n_channels = len(X)
        a = np.arange(n_channels)
        b = np.arange(n_channels)
        xa, xb = np.meshgrid(a, b)
        z = maximal_cross_correlation(xa, xb)
        if self.feature_code == "C":
            return maximal_cross_correlation(X)


def extract_feature_from_numpy(X: ndarray, selected_func: str, sfreq, frame_length_sec=5) -> ndarray:
    """
    Extract features at every 5s frame and concatenate into feature window
    Args:
        X:
        selected_func: function name (see mne-features)
        sfreq:
        frame_length_sec:

    Returns:

    """
    frames = np.array_split(X, X.shape[-1] // (sfreq * frame_length_sec), axis=1)
    features = [mne_features.get_bivariate_funcs(sfreq=sfreq)[selected_func](f) for f in
                frames]
    X = np.vstack(features).T
    return X
