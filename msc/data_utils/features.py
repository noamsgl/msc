import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin


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
        cc = cc/(N - Fs * tau)
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
        xa, xb = np.meshgrid(a,b)
        z = maximal_cross_correlation(xa, xb)
        if self.feature_code == "C":
            return maximal_cross_correlation(X)



