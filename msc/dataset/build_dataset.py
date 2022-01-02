"""
Building Datasets

* data cleaning and standardization procedures
* data resampling
"""

from datetime import timedelta
from typing import List

import portion
from portion import Interval


def intervals_to_windows(intervals, time_minutes_before=0, time_minutes_after=0) -> List[Interval]:
    """
    Returns the windows. Optionally add constant buffer time before and after intervals.
    Args:
        intervals:
        time_minutes_before:
        time_minutes_after:
    Returns:
    """

    def expand(interval):
        return portion.closedopen(interval.lower - timedelta(minutes=time_minutes_before),
                                  interval.upper + timedelta(minutes=time_minutes_after))

    return intervals.apply(expand)


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
