"""
Building Datasets

* data cleaning and standardization procedures
* data resampling
"""

from datetime import timedelta

import portion
from pandas import DataFrame
import numpy as np
from portion import Interval


def get_random_intervals(N=1000, L=1000):
    """
    Samples N random time intervals from data_df
    Args:
        N: Number of intervals to sample
        L: number of samples per interval

    Returns:

    """
    from msc.dataset.dataset import get_data_index_df

    data_index_df = get_data_index_df()

    samples = data_index_df.sample(N, replace=True)

    def get_interval(data_file_row) -> Interval:
        sfreq = data_file_row['sfreq']
        interval_length = L / data_file_row['sfreq']  # length of segment in seconds

        recording_start = data_file_row['meas_date']
        recording_end = data_file_row['end_date']
        recording_length = (recording_end - recording_start).total_seconds()

        interval_start = recording_start + timedelta(seconds=recording_length * np.random.uniform(0, 0.5))
        interval_end = interval_start + timedelta(seconds=sfreq*interval_length)
        return portion.closedopen(interval_start, interval_end)

    samples['window_interval'] = samples.apply(get_interval, axis=1)
    return samples



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
