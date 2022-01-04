"""
Conditions a generative model to produce seizures of shape (n_channels, n_times)

Output a single, randomly generated seizure


"""

from msc.dataset.dataset import get_data_index_df, SeizuresDataset, get_seizures_index_df


if __name__ == '__main__':
    sfreq = 128
    T = 10
    n_channels = 2
    n_times = sfreq * T

    # load data
    # dataset_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\SEIZURES\20220103T101554"
    dataset_dir = None

    if dataset_dir is None:
        data_index_df = get_data_index_df()
        seizures_index_df = get_seizures_index_df()
        dataset = SeizuresDataset.generate_dataset(seizures_index_df,
                                                   time_minutes_before=2, time_minutes_after=2,
                                                   fast_dev_mode=True)
    else:
        dataset = SeizuresDataset(dataset_dir)

    # X = dataset.get_X()