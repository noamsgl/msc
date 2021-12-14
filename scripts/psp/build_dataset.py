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
from msc.data_utils.features import save_dataset_to_disk
from msc.data_utils.load import get_package_from_patient, PicksOptions, get_time_as_str

fast_dev_mode = False
config = get_config()

dataset_timestamp = get_time_as_str()

print(f"Starting {os.path.basename(__file__)} with {config=}, {fast_dev_mode=} at time {dataset_timestamp}")  #todo: make config printable
dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}"
results_dir = f"{config.get('RESULTS', 'RESULTS_DIR')}"

patients = ["pat_3500", "pat_3700", "pat_4000"]
# patients = ["pat_9021002"]
for selected_func in ['max_cross_corr', 'phase_lock_val', 'nonlin_interdep', 'time_corr', 'spect_corr']:
    for patient in patients:
        # get package
        picks = PicksOptions.common_channels
        # picks = None
        save_dataset_to_disk(patient, picks, selected_func, dataset_timestamp, fast_dev_mode=fast_dev_mode)
