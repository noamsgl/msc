"""Build the dataset for the Advanced Topics in Physiological Signal Processing Course
* Clean and standardize the data (artifact removal, zero mean & unit variance per patient)
* Split the dataset to train/test
* Split the dataset into 5 min windows
* Transform each 5s epoch into a feature vector
* Save each 60 feature vectors as a pattern

References
[1] Mirowski, Piotr, et al. "Classification of patterns of EEG synchronization for seizure prediction." Clinical neurophysiology 120.11 (2009): 1927-1940.
"""
from msc.config import get_config
import pandas as pd

from msc.data_utils import get_preictal_intervals, get_interictal_intervals
from msc.data_utils.load import get_raws_from_intervals

config = get_config()
dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA','DATASET')}"

# patients = ["pat_3500", "pat_3700", "pat_4000"]
patients = ["pat_9021002"]

for patient in patients:
    #
    print(f"getting raws for {patient=}")
    # get package
    patients_index_path = f"{dataset_path}/patients_index.csv"
    patients_df = pd.read_csv(patients_index_path)
    patient_row = patients_df.loc[patients_df['pat_id'] == patient, 'package']
    assert len(patient_row) == 1, "check patient in patients_index.csv because patient was not found exactly once"
    package = patient_row.item()

    # get intervals
    preictal_intervals = get_preictal_intervals(package, patient)
    interictal_intervals = get_interictal_intervals(package, patient)

    # load raw data
    raws = get_raws_from_intervals(package, patient, preictal_intervals)
    print(raws)



