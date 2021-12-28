from msc.dataset import PSPDataset

ds_path = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\max_cross_corr\surfCO\pat_3500\20211213T182128"
dataset = PSPDataset(ds_path, add_next_seizure_info=True)
samples_df = dataset.samples_df
