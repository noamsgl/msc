"""
Conditions a generative model to produce seizures of shape (n_channels, n_times)

Output a single, randomly generated seizure


"""

from msc.dataset.dataset import get_data_index_df, SeizuresDataset, get_seizures_index_df

if __name__ == '__main__':
    data_dir = None
    data_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\SEIZURES\20220103T101554"

    if data_dir is None:
        # load data
        data_index_df = get_data_index_df()
        seizures_index_df = get_seizures_index_df()
        dataset = SeizuresDataset.generate_dataset(seizures_index_df,
                                                   time_minutes_before=0, time_minutes_after=0,
                                                   fast_dev_mode=True)

    else:
        dataset = SeizuresDataset(data_dir)

    X = dataset.get_X()


