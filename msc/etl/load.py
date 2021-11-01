from dataclasses import dataclass
from typing import List, Tuple

import mne
import os

from mne.io import Raw

__ALL_CHANNELS__ = ('FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'FZ', 'CZ', 'PZ', 'SP1', 'SP2', 'RS', 'T1', 'T2', 'EOG1', 'EOG2', 'EMG', 'ECG', 'PHO', 'CP1', 'CP2',
                    'CP5', 'CP6', 'PO1', 'PO2', 'PO5', 'PO6')

__COMMON_CHANNELS__ = (
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
    'T7', 'T8', 'P7', 'P8')
__FPATH__ = r'C:\temp\surf30\pat_103002\adm_1030102\rec_103001102\103001102_0113.data'

__CROP__ = 60


@dataclass
class PicksOptions:
    """Class for picks options"""
    all_channels: Tuple[str] = __ALL_CHANNELS__
    one_channel: Tuple[str] = ('C3',)
    two_channels: Tuple[str] = ('C3', 'C4')
    common_channels: Tuple[str] = __COMMON_CHANNELS__
    non_eeg_channels: Tuple[str] = ('EOG1', 'EOG2', 'EMG', 'ECG', 'PHO')
    eeg_channels: Tuple[str] = tuple(set(__ALL_CHANNELS__) - set(non_eeg_channels))


__PICKS__ = PicksOptions.eeg_channels


def load_raw_data(fpath: str = __FPATH__, crop: int = __CROP__, picks: Tuple[str] = __PICKS__):
    """ get sample data for testing

    Args:
        fpath: path to load data from
        crop: time length in seconds to crop
        picks: channels to select

    Returns:
        Tensor: test data
    """
    print(f"picks: {picks}")
    extension = os.path.splitext(fpath)[1]
    preload = False
    if extension == '.data':
        # file is stored in raw nicolet format
        raw: Raw = mne.io.read_raw_nicolet(fpath, ch_type='eeg', preload=preload).pick(picks).crop(0, crop)
    elif extension == ".edf":
        raw = mne.io.read_raw_edf(fpath, picks=picks, preload=preload).crop(0, crop)
    else:
        raise ValueError("unknown extension. currently supports: .data, .edf")
    return raw


def load_tensor_dataset(fpath: str = __FPATH__, crop: int = __CROP__, picks: Tuple[str] = __PICKS__):
    raw: Raw = load_raw_data(fpath, crop, picks)
    data, times, = raw.get_data(return_times=True)
    dataset = {"x_train": times,
               "y_train": data}
    return dataset
