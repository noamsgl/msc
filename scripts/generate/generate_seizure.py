"""
Conditions a generative model to produce seizures of shape (n_channels, n_times)

Output a single, randomly generated seizure


"""
from msc.dataset.dataset import get_data_index_df

if __name__ == '__main__':
    sfreq = 128
    T = 10
    n_channels = 2
    n_times = sfreq * T

    # load data
    data_index_df = get_data_index_df()
    # dataset = SeizuresDataset
    x = 5
