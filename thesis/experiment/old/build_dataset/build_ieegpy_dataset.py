"""
Build a dataset of intracranial EEG from the ieeg-portal api.
"""
import argparse
import os
print(os.getcwd())
import numpy as np
import pandas as pd
from ieeg.auth import Session
from ieeg.dataset import Dataset
import pickle
from tqdm import tqdm

from msc.config import get_authentication
from msc import config

def generate_data_intervals(ds: Dataset, duration=1e6, extension='npy', N=1000):
    onsets = np.arange(0, ds.end_time, step=duration)
    data_intervals = []
    for idx, onset in tqdm(enumerate(onsets)):
        data = ds.get_data(onset, duration, np.arange(len(ds.ch_labels)))
        if np.isnan(data).any():
            continue
        data_row = pd.Series({"interval_idx": idx, "onset": onset, "fname": f"{onset}.{extension}"}).to_frame().T
        yield data_row, data


if __name__ == '__main__':
    np.random.seed(42)

    username, password = get_authentication()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patient', help="patient name", default='I004_A0003_D001')
    args = parser.parse_args()
    patient_name = args.patient

    with Session(username, password) as s:
        ds = s.open_dataset(patient_name)
        
        dataset_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['IEEG_DATASET']}/{patient_name}"
        os.makedirs(dataset_dir, exist_ok=True)

        intervals_rows = []
        for data_row, data in generate_data_intervals(ds, N=2000, extension="npy"):
            intervals_rows.append(data_row) 
            # pickle.dump(data, open(f"{dataset_dir}/{data_row['fname'].item()}", 'wb'))
            np.save(f"{dataset_dir}/{data_row['fname'].item()}", data)
            print(data_row['interval_idx'])
            print(data)

 
        dataset_df = pd.concat(intervals_rows, axis=0)

        dataset_df.to_csv(f"{dataset_dir}/dataset.csv")
