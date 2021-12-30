import glob
import os
import pickle
import re

import mne.io
import numpy as np
import pandas as pd
import portion
from mne.io import Raw
from numpy import ndarray
from pandas import Series

from msc.config import get_config


def static_vars(**kwargs):
    """
    A decorator function which allows initializing static vars in functions.
    https://stackoverflow.com/a/279586/11814443
    Args:
        **kwargs:

    Returns:

    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(data_index_df=None)
def get_data_index_df():
    if get_data_index_df.data_index is None:
        # get config
        config = get_config()
        data_index_fpath = f"{config['PATH']['LOCAL']['RAW_DATASET']}/data_index.csv"

        data_index_df = pd.read_csv(data_index_fpath, index_col=0, parse_dates=['meas_date', 'end_date'])
    return data_index_df


@static_vars(seizures_index_df=None)
def get_seizures_index_df():
    if get_seizures_index_df.seizures_index_df is None:
        # get config
        config = get_config()
        seizures_index_fpath = f"{config['PATH']['LOCAL']['RAW_DATASET']}/seizures_index.csv"

        get_seizures_index_df.seizures_index_df = pd.read_csv(seizures_index_fpath, parse_dates=['onset', 'offset'],
                                                              index_col=0)
    return get_seizures_index_df.seizures_index_df


def get_datasets_df(feature_names=('max_cross_corr', 'phase_lock_val', 'spect_corr', 'time_corr', 'nonlin_interdep'),
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
        globbed = sorted(glob.glob(patient_dir + '/*'),
                         reverse=False)  # if reverse is set to true, get most recent datasets
        # assert len(globbed) > 0, f"Error: the dataset {row} could not be found"
        if len(globbed) > 0:
            data_dir = f"{globbed[0]}"
            return data_dir
        else:
            return None

    datasets_df['data_dir'] = datasets_df.apply(get_data_dir, axis=1)

    return datasets_df.dropna()


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

    def __init__(self, dataset_dir: str, add_next_seizure_info=False):
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
        self.samples_df = pd.read_csv(f"{dataset_dir}/dataset.csv", index_col=0)

        def parse_datetime_interval(interval_str):
            pattern = r"datetime\.datetime\(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+\)"

            def converter(val):
                # noinspection PyUnresolvedReferences
                import datetime

                return eval(val)

            interval = portion.from_string(interval_str, conv=converter, bound=pattern)
            return interval

        def file_loader():
            """returns a generator object which iterates the folder. Yields tuples like (window_id, x)."""
            for file in os.listdir(dataset_dir):
                if file.endswith('.pkl'):
                    window_id = int(re.search('\d+', file).group(0))
                    x = pickle.load(open(f"{dataset_dir}/{file}", 'rb'))
                    assert isinstance(x, ndarray), "error: the file loaded is not a numpy array"
                    yield window_id, x.reshape(-1)

        windows_dict = {window_id: x for window_id, x in file_loader()}
        self.samples_df["x"] = self.samples_df.apply(lambda sample: windows_dict[sample.name].reshape(-1), axis=1)

        self.samples_df['interval'] = self.samples_df['interval'].apply(parse_datetime_interval)
        # self.labels = list(self.samples_df.label)
        self.samples_df['lower'] = self.samples_df['interval'].apply(lambda i: i.lower)
        self.samples_df['upper'] = self.samples_df['interval'].apply(lambda i: i.upper)

        if add_next_seizure_info:
            # add time to seizure
            config = get_config()
            raw_dataset_path = config['PATH']['LOCAL']['RAW_DATASET']
            seizures_index_path = f"{raw_dataset_path}/seizures_index.csv"
            seizures_index_df = pd.read_csv(seizures_index_path, parse_dates=['onset', 'offset'], index_col=0)

            # select patient seizures
            seizures_index_df = seizures_index_df.sort_values(by="onset")
            seizures_index_df = seizures_index_df[
                ['package', 'patient', 'seizure_num', 'classif.', 'onset', 'offset', 'pattern', 'vigilance', 'origin',
                 'semiology']]

            # match on nearest key
            # !assumes seizures_index_df is sorted by onset times!
            self.samples_df = pd.merge_asof(self.samples_df.sort_values(by='lower'), seizures_index_df, left_on="lower",
                                            right_on="onset", by=['package', 'patient'], direction='forward')

    def get_X(self):
        return np.vstack(self.samples_df.x)

    def get_labels(self, format='num'):
        if format == 'desc':
            return list(self.samples_df.label_desc)
        elif format == 'num':
            return list(self.samples_df.label)
        else:
            raise ValueError("incorrect format")


class MaskedDataset(PSPDataset):
    def __init__(self, dataset_dir: str, mask: ndarray):
        super(MaskedDataset, self).__init__(dataset_dir)
        self.mask = mask

    def get_masked_X(self):
        return self.get_X() * self.mask
