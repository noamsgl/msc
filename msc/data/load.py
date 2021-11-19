import os
from dataclasses import dataclass
from typing import Tuple

import mne
import numpy as np
import torch
from mne.io import Raw

ALL_CHANNELS = ('FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                'FZ', 'CZ', 'PZ', 'SP1', 'SP2', 'RS', 'T1', 'T2', 'EOG1', 'EOG2', 'EMG', 'ECG', 'PHO', 'CP1', 'CP2',
                'CP5', 'CP6', 'PO1', 'PO2', 'PO5', 'PO6')

COMMON_CHANNELS = (
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
    'T7', 'T8', 'P7', 'P8')

FPATH = r'C:\raw_data\surf30\pat_103002\adm_1030102\rec_103001102\103001102_0113.data'

# LFREQ = 0.5  # low frequency for highpass filter
# HFREQ = 5  # high frequency for lowpass filter
LFREQ = None  # low frequency for highpass filter
HFREQ = None  # high frequency for lowpass filter
CROP = 10  # segment length measured in seconds
TRAIN_LENGTH = 10  # segment length measured in seconds
TEST_LENGTH = 2  # segment length measured in seconds


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
    return np.argmax(times > t)

def datasets(H, F, L, dt, offset, device, *, fpath: str = FPATH, picks: Tuple[str] = PICKS):
    raw: Raw = mne.io.read_raw_nicolet(fpath, ch_type='eeg', preload=True).pick(picks)
    sfreq = int(raw.info['sfreq'])
    data, times = raw.get_data(return_times=True)
    data = (data - np.mean(data)) / np.std(data)

    t_start = offset + H
    t_stop = t_start + L
    t_start_idx = np.argmax(times > t_start)
    t_stop_idx = np.argmin(times < t_stop)

    stepsize = int(dt * sfreq)
    for t in times[t_start_idx:t_stop_idx:stepsize]:
        print(f"serving dataset at time {t} seconds")
        t_ix = get_index_from_time(t, times)
        train_x = torch.Tensor(times[t_ix - H * sfreq:t_ix]).to(device=device)
        train_y = torch.Tensor(data[:, t_ix - H * sfreq:t_ix]).T.squeeze().to(device=device)
        test_x = torch.Tensor(times[t_ix:t_ix + F * sfreq]).to(device=device)
        test_y = torch.Tensor(data[:, t_ix:t_ix + F * sfreq]).T.squeeze().to(device=device)
        yield train_x, train_y, test_x, test_y
