"""
Conditions a generative model to produce seizures of shape (n_channels, n_times)

Output a single, randomly generated seizure


"""
from collections import Sequence

from msc.dataset.dataset import get_data_index_df, SeizuresDataset, get_seizures_index_df

if __name__ == '__main__':
    sfreq = 128
    T = 10
    n_channels = 2
    n_times : Sequence = sfreq * T

    # load data
    data_index_df = get_data_index_df()
    seizures_index_df = get_seizures_index_df()
    dataset = SeizuresDataset.generate_dataset(seizures_index_df,
                                               time_minutes_before=2, time_minutes_after=2, fast_dev_mode=True)
