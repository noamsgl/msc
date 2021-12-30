"""
Building Datasets

* data cleaning and standardization procedures
* data resampling
"""

import itertools
import os
import pickle
from datetime import timedelta
from itertools import chain

import pandas as pd
import portion
from sklearn.preprocessing import StandardScaler

from msc.data_utils import get_interictal_intervals, get_preictal_intervals
from msc.data_utils.features import extract_feature_from_numpy
from msc.data_utils.load import get_package_from_patient, get_patient_data_index, \
    get_raws_from_data_and_intervals, get_interval_from_raw


def intervals_to_windows(intervals, time_minutes=5):
    start_times = [list(portion.iterate(intervals[i], step=timedelta(minutes=time_minutes))) for i in
                   range(len(intervals))]
    windows = [[portion.closedopen(times[i], times[i + 1]) for i in range(len(times) - 1)] for times in start_times]
    return list(chain.from_iterable(windows))  # chain together sublists


def write_metadata(data_dir, pat_id, picks, config, fast_dev_mode, dataset_timestamp, features_desc):
    # The path of the metadata file
    path = os.path.join(data_dir, 'dataset.txt')

    with open(path, 'w') as file:
        # Datetime
        file.write(f'Dataset Creation DateTime: {dataset_timestamp}\n\n')

        # Dataset metdata
        file.write('\nDataset Metadata\n')
        file.write('***************\n')
        file.write(f'Fast Dev Mode: {fast_dev_mode}\n')
        file.write(f'Patient Id: {pat_id}\n')
        file.write(f'Features Type: {features_desc}\n')
        file.write(f'Channel Selection: {picks}\n')
        file.write(f"Resample Frequency: {config['TASK']['RESAMPLE']}\n")
        file.write(f"Preictal Min. Diff. (hours): {config['TASK']['PREICTAL_MIN_DIFF_HOURS']}\n")
        file.write(f"Interictal Min. Diff. (hours): {config['TASK']['INTERICTAL_MIN_DIFF_HOURS']}\n")
        file.write(f"Preictal Label: {config['TASK']['PREICTAL_LABEL']}\n")
        file.write(f"Interictal Label: {config['TASK']['INTERICTAL_LABEL']}\n")


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
    if fast_dev_mode:
        print(f"WARNING! {fast_dev_mode=} !!! Results are incomplete.")
    package = get_package_from_patient(patient)

    data_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}/{selected_func}/{package}/{patient}/{dataset_timestamp}"
    print(f"dumping results to {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    samples_df = pd.DataFrame(columns=['package', 'patient', 'interval',
                                       'window_id', 'fname', 'label', 'label_desc'])
    counter = itertools.count()

    print(f"getting {selected_func=} for {patient=} from {package=}")
    # get intervals
    preictal_intervals = get_preictal_intervals(package, patient)
    print(f"{len(preictal_intervals)=}")
    interictal_intervals = get_interictal_intervals(package, patient)
    print(f"{len(interictal_intervals)=}")

    # get windowed intervals
    preictal_window_intervals = intervals_to_windows(preictal_intervals)
    preictal_window_intervals = preictal_window_intervals[:2 if fast_dev_mode else len(preictal_window_intervals)]
    print(f"{len(preictal_window_intervals)=}")
    interictal_window_intervals = intervals_to_windows(interictal_intervals)
    interictal_window_intervals = interictal_window_intervals[:2 if fast_dev_mode else len(interictal_window_intervals)]
    print(f"{len(interictal_window_intervals)=}")

    # get patient data files
    patient_data_df = get_patient_data_index(patient)
    # load preictal data
    preictal_raws = get_raws_from_data_and_intervals(patient_data_df, picks, preictal_window_intervals, fast_dev_mode)
    print(f"{len(preictal_raws)=}")
    write_metadata(data_dir, patient, preictal_raws[0].info["ch_names"], config, fast_dev_mode, dataset_timestamp,
                   selected_func)

    print("starting to extract features for preictal raws")
    for raw in preictal_raws:
        interval = get_interval_from_raw(raw)
        window_id = next(counter)
        fname = f"{data_dir}/window_{window_id}.pkl"
        y = config['TASK']['PREICTAL_LABEL']
        row = {"package": package,
               "patient": patient,
               "interval": interval,
               "window_id": window_id,
               "fname": fname,
               "label": y,
               "label_desc": "preictal"}
        samples_df = samples_df.append(row, ignore_index=True)
        X = raw.get_data()
        X = StandardScaler().fit_transform(X)
        print(f"dumping {window_id=} to {fname=}")
        # X = mne_features.feature_extraction.FeatureExtractor(sfreq=config['TASK']['RESAMPLE'],
        #                                                      selected_funcs=selected_funcs).fit_transform(X)
        X = extract_feature_from_numpy(X, selected_func, float(config['TASK']['RESAMPLE']))
        pickle.dump(X, open(fname, 'wb'))

    # clear memory from preictal raws
    del preictal_raws

    interictal_raws = get_raws_from_data_and_intervals(patient_data_df, picks, interictal_window_intervals,
                                                       fast_dev_mode)
    print(f"{interictal_raws}")
    print("starting to extract features for interictal raws")
    for raw in interictal_raws:
        interval = get_interval_from_raw(raw)
        window_id = next(counter)
        fname = f"{data_dir}/window_{window_id}.pkl"
        y = config['TASK']['INTERICTAL_LABEL']
        row = {"package": package,
               "patient": patient,
               "interval": interval,
               "window_id": window_id,
               "fname": fname,
               "label": y,
               "label_desc": "interictal"}
        samples_df = samples_df.append(row, ignore_index=True)
        X = raw.get_data()
        X = StandardScaler().fit_transform(X)
        # X = mne_features.feature_extraction.FeatureExtractor(sfreq=float(config['TASK']['RESAMPLE']),
        #                                                      selected_funcs=selected_funcs).fit_transform(X)
        X = extract_feature_from_numpy(X, selected_func, float(config['TASK']['RESAMPLE']))
        print(f"dumping {window_id=} to {fname=}")
        pickle.dump(X, open(fname, 'wb'))
    samples_df_path = f"{data_dir}/dataset.csv"
    print(f"saving samples_df to {samples_df_path=}")
    samples_df.to_csv(samples_df_path)
    return
