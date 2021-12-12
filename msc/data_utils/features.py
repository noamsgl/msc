from datetime import timedelta
from itertools import chain
from typing import List

import numpy as np
import portion
from mne_features.feature_extraction import FeatureExtractor
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msc.data_utils import get_interictal_intervals, get_preictal_intervals
from msc.data_utils.load import config, get_raws_from_intervals, get_package_from_patient


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
    windows = [[portion.closed(times[i], times[i + 1]) for i in range(len(times) - 1)] for times in start_times]
    return list(chain.from_iterable(windows))  # chain together sublists


def get_features_and_labels(patient, selected_funcs):
    package = get_package_from_patient(patient)

    print(f"getting raws for {patient=} from {package=}")
    # get intervals
    preictal_intervals = get_preictal_intervals(package, patient)
    print(f"{preictal_intervals=}")
    interictal_intervals = get_interictal_intervals(package, patient)
    print(f"{interictal_intervals=}")

    # get windowed intervals
    preictal_window_intervals = intervals_to_windows(preictal_intervals)
    interictal_window_intervals = intervals_to_windows(interictal_intervals)

    # load resampled raw datas
    preictal_raws = get_raws_from_intervals(package, patient, preictal_window_intervals)
    print(f"{preictal_raws=}")  # todo: sanity check here
    interictal_raws = get_raws_from_intervals(package, patient, interictal_window_intervals)
    print(f"{interictal_raws=}")

    # convert to numpy arrays
    preictal_Xs: List[ndarray] = [raw.get_data() for raw in preictal_raws]
    print(f"{preictal_Xs=}")
    interictal_Xs: List[ndarray] = [raw.get_data() for raw in interictal_raws]
    print(f"{interictal_Xs=}")

    # build labels
    preictal_Ys = [int(config.get("DATA", "PREICTAL_LABEL")) for _ in preictal_Xs]
    interictal_Ys = [int(config.get("DATA", "INTERICTAL_LABEL")) for _ in interictal_Xs]

    # standardize Xs
    # preictal_Xs = [standardize(X) for X in preictal_Xs]
    # interictal_Xs = [standardize(X) for X in interictal_Xs]

    # concat classes
    Xs = preictal_Xs + interictal_Xs
    Ys = preictal_Ys + interictal_Ys

    # build data transform and classification pipeline

    pipe = Pipeline(steps=[('standardize', StandardScaler()),
                           ('fe', FeatureExtractor(sfreq=config.get("DATA", "RESAMPLE"),
                                                   selected_funcs=selected_funcs))])

    Xs = pipe.fit_transform(Xs)
    return Xs, Ys
