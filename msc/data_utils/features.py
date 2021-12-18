import itertools
import os
import pickle
from datetime import timedelta, datetime
from itertools import chain

import mne_features.feature_extraction
import numpy as np
import pandas as pd
import portion
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from msc.data_utils import get_interictal_intervals, get_preictal_intervals
from msc.data_utils.load import get_package_from_patient, get_patient_data_index, \
    get_raws_from_data_and_intervals, get_interval_from_raw, get_time_as_str


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


def intervals_to_windows(intervals, time_minutes=5):
    start_times = [list(portion.iterate(intervals[i], step=timedelta(minutes=time_minutes))) for i in
                   range(len(intervals))]
    windows = [[portion.closedopen(times[i], times[i + 1]) for i in range(len(times) - 1)] for times in start_times]
    return list(chain.from_iterable(windows))  # chain together sublists


def write_metadata(data_dir, pat_id, picks, config, fast_dev_mode, dataset_timestamp, features_desc):
    # The path of the metadata file
    path = os.path.join(data_dir, 'dataset.txt')

    with open(path, 'w') as file:
        # Datetime
        file.write(f'Dataset Creation DateTime: {dataset_timestamp}\n\n')

        # Dataset metdata
        file.write('\nDataset Metadata\n')
        file.write('***************\n')
        file.write(f'Fast Dev Mode: {fast_dev_mode}\n')
        file.write(f'Patient Id: {pat_id}\n')
        file.write(f'Features Type: {features_desc}\n')
        file.write(f'Channel Selection: {picks}\n')
        file.write(f'Resample Frequency: {config.get("DATA", "RESAMPLE")}\n')
        file.write(f'Preictal Min. Diff. (hours): {config.get("TASK", "PREICTAL_MIN_DIFF_HOURS")}\n')
        file.write(f'Interictal Min. Diff. (hours): {config.get("TASK", "INTERICTAL_MIN_DIFF_HOURS")}\n')
        file.write(f'Preictal Label: {config.get("TASK", "PREICTAL_LABEL")}\n')
        file.write(f'Interictal Label: {config.get("TASK", "INTERICTAL_LABEL")}\n')



def extract_feature_from_numpy(X: ndarray, selected_func: str, sfreq, frame_length_sec=5) -> ndarray:
    """
    Extract features at every 5s frame and concatenate into feature window
    Args:
        X:
        selected_func:
        sfreq:
        frame_length_sec:

    Returns:

    """
    frames = np.array_split(X, X.shape[-1] // (sfreq * frame_length_sec), axis=1)
    features = [mne_features.get_bivariate_funcs(sfreq=sfreq)[selected_func](f) for f in
                frames]
    X = np.vstack(features).T
    return X


def save_dataset_to_disk(patient, picks, selected_func, dataset_timestamp, config, fast_dev_mode=False):
    """
    Gets the features Xs and labels Ys for a partitioned and feature extracted dataset
    Args:
        patient: the patient's id
        picks:
        selected_func: the feature function
        dataset_timestamp:
        fast_dev_mode:

    Returns: samples_df, Xs, Ys

    """
    if fast_dev_mode:
        print(f"WARNING! {fast_dev_mode=} !!! Results are incomplete.")
    package = get_package_from_patient(patient)

    data_dir = f"{config.get('RESULTS', 'RESULTS_DIR')}/{config.get('DATA', 'DATASET')}/{selected_func}/{package}/{patient}/{dataset_timestamp}"
    print(f"dumping results to {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    samples_df = pd.DataFrame(columns=['package', 'patient', 'interval',
                                       'window_id', 'fname', 'label', 'label_desc'])
    counter = itertools.count()

    print(f"getting {selected_func=} for {patient=} from {package=}")
    # get intervals
    preictal_intervals = get_preictal_intervals(package, patient, fast_dev_mode)
    print(f"{preictal_intervals=}")
    interictal_intervals = get_interictal_intervals(package, patient, fast_dev_mode)
    print(f"{interictal_intervals=}")

    # get windowed intervals
    preictal_window_intervals = intervals_to_windows(preictal_intervals)
    print(f"{preictal_window_intervals=}")
    interictal_window_intervals = intervals_to_windows(interictal_intervals)
    print(f"{interictal_window_intervals=}")

    # get patient data files
    patient_data_df = get_patient_data_index(patient)
    # load preictal data
    preictal_raws = get_raws_from_data_and_intervals(patient_data_df, picks, preictal_window_intervals, fast_dev_mode)
    print(f"{preictal_raws=}")
    write_metadata(data_dir, patient, preictal_raws[0].info["ch_names"], config, fast_dev_mode, dataset_timestamp, selected_func)


    print("starting to process preictal raws")
    for raw in preictal_raws[:2 if fast_dev_mode else len(preictal_raws)]:
        interval = get_interval_from_raw(raw)
        window_id = next(counter)
        fname = f"{data_dir}/window_{window_id}.pkl"
        y = config.get("DATA", "PREICTAL_LABEL")
        row = {"package": package,
               "patient": patient,
               "interval": interval,
               "window_id": window_id,
               "fname": fname,
               "label": y,
               "label_desc": "preictal"}
        samples_df = samples_df.append(row, ignore_index=True)
        X = raw.get_data()
        X = StandardScaler().fit_transform(X)
        print(f"dumping {window_id=} to {fname=}")
        # X = mne_features.feature_extraction.FeatureExtractor(sfreq=config.get("DATA", "RESAMPLE"),
        #                                                      selected_funcs=selected_funcs).fit_transform(X)
        X = extract_feature_from_numpy(X, selected_func, float(config.get("DATA", "RESAMPLE")))
        pickle.dump(X, open(fname, 'wb'))

    # clear memory from preictal raws
    del preictal_raws

    interictal_raws = get_raws_from_data_and_intervals(patient_data_df, picks, interictal_window_intervals,
                                                       fast_dev_mode)
    print(f"{interictal_raws}")
    print("starting to process interictal raws")
    for raw in interictal_raws[:2 if fast_dev_mode else len(interictal_raws)]:
        interval = get_interval_from_raw(raw)
        window_id = next(counter)
        fname = f"{data_dir}/window_{window_id}.pkl"
        y = config.get("TASK", "INTERICTAL_LABEL")
        row = {"package": package,
               "patient": patient,
               "interval": interval,
               "window_id": window_id,
               "fname": fname,
               "label": y,
               "label_desc": "interictal"}
        samples_df = samples_df.append(row, ignore_index=True)
        X = raw.get_data()
        X = StandardScaler().fit_transform(X)
        # X = mne_features.feature_extraction.FeatureExtractor(sfreq=float(config.get("DATA", "RESAMPLE")),
        #                                                      selected_funcs=selected_funcs).fit_transform(X)
        X = extract_feature_from_numpy(X, selected_func, float(config.get("DATA", "RESAMPLE")))
        print(f"dumping {window_id=} to {fname=}")
        pickle.dump(X, open(fname, 'wb'))
    samples_df_path = f"{data_dir}/dataset.csv"
    print(f"saving samples_df to {samples_df_path=}")
    samples_df.to_csv(samples_df_path)
    return

