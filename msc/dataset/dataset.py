import glob
import itertools
import os
import pickle
import re
from typing import Sequence

import numpy as np
import pandas as pd
import portion
import yaml
from numpy import ndarray
from pandas import DataFrame, Series

from msc.config import get_config
from msc.data_utils.load import add_raws_to_intervals_df, PicksOptions, get_time_as_str
from msc.dataset.build_dataset import add_window_intervals


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
    if get_data_index_df.data_index_df is None:
        # get config
        config = get_config()
        # noinspection PyTypeChecker
        data_index_fpath = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}/data_index.csv"

        data_index_df = pd.read_csv(data_index_fpath, index_col=0, parse_dates=['meas_date', 'end_date'])
    return data_index_df


@static_vars(seizures_index_df=None)
def get_seizures_index_df():
    if get_seizures_index_df.seizures_index_df is None:
        # get config
        config = get_config()
        # noinspection PyTypeChecker
        seizures_index_fpath = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}/seizures_index.csv"

        seizures_index_df = pd.read_csv(seizures_index_fpath, parse_dates=['onset', 'offset'],
                                        index_col=0).set_index(['patient', 'seizure_num'])
        seizures_index_df['interval'] = seizures_index_df.apply(lambda row:
                                                                portion.closedopen(
                                                                    row.onset,
                                                                    row.offset),
                                                                axis=1)
        if seizures_index_df.loc[:, ['onset', 'offset']].isna().any().any():
            print("warning: dropping some seizures because onset or offset are NaT")
            seizures_index_df = seizures_index_df.dropna(subset=['onset', 'offset'])

        print("warning: dropping seizures with length < 5 seconds")
        seizures_index_df = seizures_index_df.loc[seizures_index_df.length > 5]
        get_seizures_index_df.seizures_index_df = seizures_index_df
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
        # noinspection PyTypeChecker
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

    def __init__(self, fast_dev_mode: bool = False):
        self.create_time = get_time_as_str()
        self.fast_dev_mode = fast_dev_mode
        if self.fast_dev_mode:
            print(f"WARNING! {self.fast_dev_mode=} !!! Results are incomplete.")


class XDataset(baseDataset):
    """
    The base class for datasets without labels.
    """

    def __init__(self):
        super().__init__()


class SeizuresDataset(XDataset):
    """
    A class for a dataset composed of seizures only.
    Regular instantiation (__init__()) can be used for loading an existing dataset.
    @classmethod generate_dataset() can be used to generate one from scratch.
    """

    def __init__(self, dataset_dir: str):
        """
        Args:
            data_index_df:
            fast_dev_mode:
            picks:
        """
        # baseDataset.__init__(self, fast_dev_mode)
        super().__init__()
        assert os.path.exists(dataset_dir), "error: the dataset directory does not exist"
        assert os.path.isfile(f"{dataset_dir}/samples_df.csv"), "error: samples_df.csv not found in dataset_dir"
        self.samples_df = pd.read_csv(f"{dataset_dir}/samples_df.csv", index_col=0)

    @classmethod
    def generate_dataset(cls, seizures_index_df: DataFrame, fast_dev_mode: bool = False,
                         picks: Sequence[str] = PicksOptions.common_channels, output_dir: str = None,
                         time_minutes_before=0, time_minutes_after=0):
        """
        Gets the seizure intervals,
        Iterates over the data files
        Returns:

        """
        config = get_config()
        create_time = get_time_as_str()
        if output_dir is None:
            # noinspection PyTypeChecker
            output_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}/SEIZURES/{create_time}"

        print(f"Creating Output Directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        metadata = {"creation_time": create_time,
                    "fast_dev_mode": fast_dev_mode,
                    "ictal_label": config["TASK"]["ICTAL_LABEL"],
                    "picks": picks,
                    "sfreq": config['TASK']['RESAMPLE'],
                    "time_minutes_before": time_minutes_before,
                    "time_minutes_after": time_minutes_after}

        with open(f"{output_dir}/dataset.yml", 'w') as metadata_file:
            yaml.dump(metadata, metadata_file, default_flow_style=False)

        # initialize samples_df
        samples_df = pd.DataFrame(
            columns=['patient_name', 'ictal_interval', 'window_interval', 'window_id', 'fname', 'label', 'label_desc'])

        # get seizure intervals
        ictal_intervals: Series = seizures_index_df.interval
        ictal_intervals = ictal_intervals.rename('ictal_interval')
        intervals_df = ictal_intervals.to_frame()

        # convert ictal intervals into window intervals (expand with time before and after)
        window_intervals: DataFrame = add_window_intervals(intervals_df, time_minutes_before,
                                                           time_minutes_after)

        print("starting to load raw files")
        # load Raws

        intervals_and_raws: DataFrame = add_raws_to_intervals_df(window_intervals, picks, fast_dev_mode)

        print("starting to process raw files")
        counter = itertools.count()
        for ictal_idx, sample_row in intervals_and_raws.dropna().iterrows():
            # create samples_df row
            window_id = next(counter)
            fname = f"{output_dir}/window_{window_id}.pkl"
            # create label and label_desc
            y = config['TASK']['ICTAL_LABEL']
            y_desc = "ictal"
            # append row to samples_df
            row = {"patient_name": ictal_idx[0],
                   "seizure_num": ictal_idx[1],
                   "ictal_interval": sample_row.ictal_interval,
                   "window_interval": sample_row.window_interval,
                   "window_id": window_id,
                   "fname": fname,
                   "label": y,
                   "label_desc": y_desc
                   }
            samples_df = samples_df.append(row, ignore_index=True)

            # Get X
            X = sample_row.raw.get_data()

            # Scale X
            # from sklearn.preprocessing import StandardScaler
            # X = StandardScaler().fit_transform(X)

            # Perform Feature Extraction (Optional)
            # X = mne_features.feature_extraction.FeatureExtractor(sfreq=config['TASK']['RESAMPLE'],
            #                                                      selected_funcs=selected_funcs).fit_transform(X)
            # X = extract_feature_from_numpy(X, selected_func, float(config['TASK']['RESAMPLE']))

            # Dump to file
            print(f"dumping {window_id=} to {fname=}")
            pickle.dump(X, open(fname, 'wb'))

        samples_df_path = f"{output_dir}/samples_df.csv"
        print(f"saving samples_df to {samples_df_path=}")
        samples_df.to_csv(samples_df_path)
        return cls(output_dir)

    def _write_metadata(self):
        # The path of the metadata file
        # todo: refactor this.
        # implement with as YAML

        raise NotImplemented("see todo")
        path = os.path.join(data_dir, 'dataset.txt')
        config = get_config()
        with open(path, 'w') as file:
            # Datetime
            file.write(f'Dataset Creation DateTime: {dataset_timestamp}\n\n')

            # Dataset metdata
            file.write('\nDataset Metadata\n')
            file.write('***************\n')
            file.write(f'Fast Dev Mode: {self.fast_dev_mode}\n')
            file.write(f'Patient Id: {pat_id}\n')
            file.write(f'Features Type: {features_desc}\n')
            file.write(f'Channel Selection: {picks}\n')
            file.write(f"Resample Frequency: {config['TASK']['RESAMPLE']}\n")
            file.write(f"Preictal Min. Diff. (hours): {config['TASK']['PREICTAL_MIN_DIFF_HOURS']}\n")
            file.write(f"Interictal Min. Diff. (hours): {config['TASK']['INTERICTAL_MIN_DIFF_HOURS']}\n")
            file.write(f"Preictal Label: {config['TASK']['PREICTAL_LABEL']}\n")
            file.write(f"Interictal Label: {config['TASK']['INTERICTAL_LABEL']}\n")


class predictionDataset(baseDataset):
    """
    The base class for prediction datasets (allows option to add detectionDataset in the future).
    """

    def __init__(self):
        super().__init__()


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

        def file_loader(dataset_dir):
            """returns a generator object which iterates the folder dataset_dir. Yields tuples like (window_id, x)."""
            for file in os.listdir(dataset_dir):
                if file.endswith('.pkl'):
                    window_id = int(re.search(r'\d+', file).group(0))
                    x = pickle.load(open(f"{dataset_dir}/{file}", 'rb'))
                    assert isinstance(x, ndarray), "error: the file loaded is not a numpy array"
                    yield window_id, x.reshape(-1)

        windows_dict = {window_id: x for window_id, x in file_loader(dataset_dir)}
        self.samples_df["x"] = self.samples_df.apply(lambda sample: windows_dict[sample.name].reshape(-1), axis=1)

        self.samples_df['interval'] = self.samples_df['interval'].apply(parse_datetime_interval)
        # self.labels = list(self.samples_df.label)
        self.samples_df['lower'] = self.samples_df['interval'].apply(lambda i: i.lower)
        self.samples_df['upper'] = self.samples_df['interval'].apply(lambda i: i.upper)

        if add_next_seizure_info:
            # add time to seizure
            config = get_config()
            # noinspection PyTypeChecker
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
