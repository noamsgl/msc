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
from typing import List

import pandas as pd
from mne_features.feature_extraction import FeatureExtractor
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msc.config import get_config
from msc.data_utils.load import get_raws_from_intervals
from msc.data_utils.windower import get_preictal_intervals, get_interictal_intervals

config = get_config()
dataset_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}"

patients = ["pat_3500", "pat_3700", "pat_4000"]
# patients = ["pat_9021002"]

for patient in patients:
    # get package
    patients_index_path = f"{dataset_path}/patients_index.csv"
    patients_df = pd.read_csv(patients_index_path)
    patient_row = patients_df.loc[patients_df['pat_id'] == patient, 'package']
    assert len(patient_row) == 1, "check patient in patients_index.csv because patient was not found exactly once"
    package = patient_row.item()

    print(f"getting raws for {patient=} from {package=}")
    # get intervals
    preictal_intervals = get_preictal_intervals(package, patient)
    print(f"{preictal_intervals=}")
    interictal_intervals = get_interictal_intervals(package, patient)
    print(f"{interictal_intervals=}")
    # load resampled raw datas
    preictal_raws = get_raws_from_intervals(package, patient, preictal_intervals)
    print(preictal_raws)  # todo: sanity check here
    interictal_raws = get_raws_from_intervals(package, patient, interictal_intervals)
    print(interictal_raws)

    # convert to numpy arrays
    preictal_Xs: List[ndarray] = [raw.get_data() for raw in preictal_raws]
    interictal_Xs: List[ndarray] = [raw.get_data() for raw in interictal_raws]

    # build labels
    preictal_Ys = [int(config.get("DATA", "PREICTAL_LABEL")) for X in preictal_Xs]
    interictal_Ys = [int(config.get("DATA", "INTERICTAL_LABEL")) for X in interictal_Xs]

    # standardize Xs
    # preictal_Xs = [standardize(X) for X in preictal_Xs]
    # interictal_Xs = [standardize(X) for X in interictal_Xs]

    # concat classes
    Xs = preictal_Xs + interictal_Xs
    Ys = preictal_Ys + interictal_Ys

    # build data transform and classification pipeline
    selected_funcs = ['mean']

    pipe = Pipeline(steps=[('standardize', StandardScaler()),
                           ('fe', FeatureExtractor(sfreq=config.get("DATA", "RESAMPLE"),
                                                   selected_funcs=selected_funcs))])

    Xs = pipe.fit_transform(Xs)
    features_dump_dir = f"{config.get('RESULTS', 'RESULTS_DIR')}/{config.get('DATA', 'DATASET')}/{'_'.join(selected_funcs)}/{package}/{patient}"
    os.makedirs(features_dump_dir, exist_ok=True)

    print(f"dumping Xs to {features_dump_dir}/Xs.pkl")
    pickle.dump(Xs, open(f"{features_dump_dir}/Xs.pkl", 'wb'))
    print(f"dumping Ys to {features_dump_dir}/Ys.pkl")
    pickle.dump(Ys, open(f"{features_dump_dir}/Ys.pkl", 'wb'))