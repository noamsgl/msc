"""
Building Datasets

* data cleaning and standardization procedures
* data resampling
"""

from datetime import timedelta
from itertools import chain

import portion


def intervals_to_windows(intervals, time_minutes=5):
    """
    todo: implement this
    Args:
        intervals:
        time_minutes:

    Returns:

    """
    raise NotImplementedError()
    start_times = [list(portion.iterate(intervals[i], step=timedelta(minutes=time_minutes))) for i in
                   range(len(intervals))]
    windows = [[portion.closedopen(times[i], times[i + 1]) for i in range(len(times) - 1)] for times in start_times]
    return list(chain.from_iterable(windows))  # chain together sublists


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
