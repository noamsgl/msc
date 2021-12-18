import glob
import os
import pickle
import re

import mne.io
import numpy as np
import pandas as pd
from mne.io import Raw
from numpy import ndarray
from pandas import Series

from msc.config import get_config


def get_datasets_df(feature_names=('max_cross_corr', 'phase_lock_val', 'spect_corr', 'time_corr'),
                    patient_names=('pat_3500', 'pat_3700', 'pat_7200')):
    # get config
    config = get_config()

    # initialize datasets
    feature_names = feature_names
    patient_names = patient_names
    index = pd.MultiIndex.from_product([feature_names, patient_names], names=["feature_name", "patient_name"])
    datasets_df = pd.DataFrame(index=index).reset_index()

    def get_data_dir(row):
        patient_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}" \
                      f"/{row['feature_name']}/surfCO/{row['patient_name']}"
        globbed = sorted(glob.glob(patient_dir + '/*'), reverse=True)
        assert len(globbed) > 0, f"Error: the dataset {row} could not be found"
        data_dir = f"{globbed[0]}"
        return data_dir

    datasets_df['data_dir'] = datasets_df.apply(get_data_dir, axis=1)


    return datasets_df

class baseDataset:
    """
    The dataset base class
    """

    def __init__(self):
        pass


class predictionDataset(baseDataset):
    """
    The base class for prediction datasets (allows option to add detectionDataset in the future).
    """

    def __init__(self):
        super().__init__()


class RawDataset(baseDataset):
    """
    A class for loading data in its raw form
    """

    def __init__(self, dataset_dir: str):
        """
        Initialize a raw dataset.
        Lazy loading of data.
        Args:
            dataset_dir: path to the dataset dir (should be the packages)
        """
        self.dataset_dir = dataset_dir
        assert os.path.exists(dataset_dir), "error: the dataset directory does not exist"
        assert os.path.isfile(f"{dataset_dir}/data_index.csv"), "error: data_index.csv not found"
        assert os.path.isfile(f"{dataset_dir}/patients_index.csv"), "error: patients_index.csv not found"
        assert os.path.isfile(f"{dataset_dir}/seizures_index.csv"), "error: seizures_index.csv not found"
        self.data_df = pd.read_csv(f"{dataset_dir}/data_index.csv", index_col=0)
        self.patients_df = pd.read_csv(f"{dataset_dir}/patients_index.csv", index_col=0)
        self.seizures_df = pd.read_csv(f"{dataset_dir}/seizures_index.csv", index_col=0)

    def get_raw(self, data_row: Series, preload=False) -> Raw:
        """
        gets a random raw sample.
        Args:
            seconds:

        Returns:

        """
        raw_fpath = self._build_fpath_for_raw(data_row)
        raw = mne.io.read_raw_nicolet(raw_fpath, ch_type='eeg', preload=preload)
        return raw

    def _build_fpath_for_raw(self, data_row) -> str:
        """
        return the fpath for the data row
        Args:
            data_row:

        Returns:

        """
        return f'{self.dataset_dir}/{data_row.package.item()}/{data_row.patient.item()}/{data_row.admission.item()}/' \
               f'{data_row.recording.item()}/{data_row.fname.item()}'


class PSPDataset(predictionDataset):
    """
    A class for Physiological Signal Processing Course Datasets
    It is fit for a pattern dataset of pickled numpy arrays representing time windows,
    And a structured dataset.csv according to project standards (see ReadME).
    """

    def __init__(self, dataset_dir: str):
        """
        Initialize a dataset for the PSP course project.
        Loads the windows and provides quick access to splits and folds.
        Args:
            dataset_dir:
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        assert os.path.exists(dataset_dir), "error: the dataset directory does not exist"
        assert os.path.isfile(f"{dataset_dir}/dataset.csv"), "error: dataset.csv file not found"
        self.samples_df = pd.read_csv(f"{dataset_dir}/dataset.csv", index_col='window_id')

        def file_loader():
            """returns a generator object which iterates the folder yields tuples like (window_id, x)."""
            for file in os.listdir(dataset_dir):
                if file.endswith('.pkl'):
                    window_id = int(re.search('\d+', file).group(0))
                    x = pickle.load(open(f"{dataset_dir}/{file}", 'rb'))
                    assert isinstance(x, ndarray), "error: the file loaded is not a numpy array"
                    yield window_id, x.reshape(-1)

        windows_dict = {window_id: x for window_id, x in file_loader()}
        self.samples_df["x"] = self.samples_df.apply(lambda sample: windows_dict[sample.name].reshape(-1), axis=1)

        # self.labels = list(self.samples_df.label)

    def get_X(self):
        return np.vstack(self.samples_df.x)

    def get_labels(self, format='num'):
        if format == 'desc':
            return list(self.samples_df.label_desc)
        elif format == 'num':
            return list(self.samples_df.label)
        else:
            raise ValueError("incorrect format")
