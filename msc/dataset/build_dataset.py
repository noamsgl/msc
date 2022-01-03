"""
Building Datasets

* data cleaning and standardization procedures
* data resampling
"""

from datetime import timedelta

import portion
from pandas import DataFrame


def add_window_intervals(intervals_df: DataFrame, time_minutes_before=0, time_minutes_after=0) -> DataFrame:
    """
    Adds window_interval to dataframe with ictal_intervals.
    Args:
        intervals_df:
        time_minutes_before:
        time_minutes_after:
    Returns:
    """

    assert 'ictal_interval' in intervals_df

    def expand(interval):
        return portion.closedopen(interval.lower - timedelta(minutes=time_minutes_before),
                                  interval.upper + timedelta(minutes=time_minutes_after))

    intervals_df["window_interval"] = intervals_df['ictal_interval'].apply(expand)

    return intervals_df


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
