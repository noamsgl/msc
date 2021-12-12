"""Build the dataset for the Advanced Topics in Physiological Signal Processing Course
* Clean and standardize the data (artifact removal, zero mean & unit variance per data file)
* Resample to sfreq (256 in original paper)
* Split the dataset to train/test
* Split the dataset into 5 min windows
* Transform each 5s epoch into a feature vector
* Save each 60 feature vectors as a pattern

References
[1] Mirowski, Piotr, et al. "Classification of patterns of EEG synchronization for seizure prediction." Clinical neurophysiology 120.11 (2009): 1927-1940.
"""
import os.path
import pickle

from msc.config import get_config
from msc.data_utils.features import get_features_and_labels
from msc.data_utils.load import get_package_from_patient

config = get_config()
print(f"Starting {os.path.basename(__file__)} with {config=}")
dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}"

patients = ["pat_3500", "pat_3700", "pat_4000"]
# patients = ["pat_9021002"]

for patient in patients:
    # get package
    selected_funcs = ['mean']

    Xs, Ys = get_features_and_labels(patient, selected_funcs)
    package = get_package_from_patient(patient)
    features_dump_dir = f"{config.get('RESULTS', 'RESULTS_DIR')}/{config.get('DATA', 'DATASET')}/{'_'.join(selected_funcs)}/{package}/{patient}"
    os.makedirs(features_dump_dir, exist_ok=True)

    print(f"dumping Xs to {features_dump_dir}/Xs.pkl")
    pickle.dump(Xs, open(f"{features_dump_dir}/Xs.pkl", 'wb'))
    print(f"dumping Ys to {features_dump_dir}/Ys.pkl")
    pickle.dump(Ys, open(f"{features_dump_dir}/Ys.pkl", 'wb'))
