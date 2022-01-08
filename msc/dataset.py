import glob
import itertools
import os
import pickle
import re
from datetime import timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import portion
import torch
import yaml
from numpy import ndarray
from pandas import DataFrame, Series
from portion import Interval
from torch import Tensor

import msc.data_utils as data_utils
from msc import config


# from msc.data_utils import get_preictal_intervals, get_interictal_intervals
# from msc.data_utils.features import extract_feature_from_numpy
# from msc.data_utils.load import add_raws_to_intervals_df, PicksOptions, get_time_as_str


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
        # noinspection PyTypeChecker
        data_index_fpath = f"{config['PATH'][config['INDEX_MACHINE']]['RAW_DATASET']}/data_index.csv"
        try:
            get_data_index_df.data_index_df = pd.read_csv(data_index_fpath, index_col=0,
                                                          parse_dates=['meas_date', 'end_date'])
        except FileNotFoundError as e:
            print(f"file {data_index_fpath} not found: check the config file")
            raise e
    return get_data_index_df.data_index_df


@static_vars(seizures_index_df=None)
def get_seizures_index_df():
    if get_seizures_index_df.seizures_index_df is None:
        # noinspection PyTypeChecker
        seizures_index_fpath = f"{config['PATH'][config['INDEX_MACHINE']]['RAW_DATASET']}/seizures_index.csv"

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


def add_window_intervals(intervals_df: DataFrame, time_minutes_before=0, time_minutes_after=0) -> DataFrame:
    """
    Adds window_interval to dataframe with ictal_intervals.
    Args:
        intervals_df:
        time_minutes_before:
        time_minutes_after:
    Returns:
    """

    assert 'ictal_interval' in intervals_df

    def expand(interval):
        return portion.closedopen(interval.lower - timedelta(minutes=time_minutes_before),
                                  interval.upper + timedelta(minutes=time_minutes_after))

    intervals_df["window_interval"] = intervals_df['ictal_interval'].apply(expand)

    return intervals_df


def get_random_intervals(N=1000, L=1000) -> DataFrame:
    """
    Samples N random time intervals from data_df
    Args:
        N: Number of intervals to sample
        L: number of samples per interval

    Returns:

    """

    data_index_df = get_data_index_df()

    samples = data_index_df.sample(N, replace=True)

    sfreq = int(config['TASK']['RESAMPLE'])

    def get_interval(data_file_row) -> Interval:
        """gets a random timestamped Interval"""
        interval_length = L / sfreq  # length of segment in seconds

        recording_start = data_file_row['meas_date']
        recording_end = data_file_row['end_date']
        recording_length = (recording_end - recording_start).total_seconds()

        interval_start = recording_start + timedelta(seconds=recording_length * np.random.uniform(0, 0.5))
        interval_end = interval_start + timedelta(seconds=interval_length)
        return portion.closedopen(interval_start, interval_end)

    samples['window_interval'] = samples.apply(get_interval, axis=1)
    return samples


class baseDataset:
    """
    The dataset base class
    """

    def __init__(self, dataset_dir, fast_dev_mode: bool = False):
        self.dataset_dir = dataset
        self.create_time = data_utils.get_time_as_str()
        self.fast_dev_mode = fast_dev_mode
        if self.fast_dev_mode:
            print(f"WARNING! {self.fast_dev_mode=} !!! Results are incomplete.")

    @staticmethod
    def parse_datetime_interval(interval_str: str, pattern=None) -> Interval:
        """
        parses an interval time stamp
        Args:
            interval_str:
            pattern:

        Returns: Interval with the resolved end points

        """
        if pattern is None:
            # pattern = r"datetime\.datetime\(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+\)"
            pattern = r"Timestamp\('.{3,30}'\)"

        def converter(val):
            # noinspection PyUnresolvedReferences
            from pandas import Timestamp

            return eval(val)

        interval = portion.from_string(interval_str, conv=converter, bound=pattern)
        return interval

    @staticmethod
    def file_loader(dataset_dir):
        """returns a generator object which iterates the folder dataset_dir. Yields tuples like (window_id, x)."""
        for file in os.listdir(dataset_dir):
            if file.endswith('.pkl'):
                window_id = int(re.search(r'\d+', file).group(0))
                x = pickle.load(open(f"{dataset_dir}/{file}", 'rb'))
                assert isinstance(x, ndarray), "error: the file loaded is not a numpy array"
                yield window_id, x


class RawDataset(baseDataset):
    """
    The base class for datasets without labels.
    """

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        assert os.path.isfile(f"{dataset_dir}/dataset.yml"), "error: dataset.yml not found in dataset_dir"
        with open(f"{dataset_dir}/dataset.yml", "r") as metadata_stream:
            try:
                self.metadata = yaml.load(metadata_stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print("problem loading dataset.yml")
                raise exc

    def _load_data(self):
        try:
            print("loading")
            self.samples_df = self.samples_df.reset_index()
            windows_dict = {window_id: x for window_id, x in self.file_loader(self.dataset_dir)}
            self.samples_df["x"] = self.samples_df.apply(lambda sample: windows_dict[sample.name], axis=1)
            self.samples_df['interval'] = self.samples_df['window_interval'].apply(self.parse_datetime_interval)
            self.samples_df['lower'] = self.samples_df['interval'].apply(lambda i: i.lower)
            self.samples_df['upper'] = self.samples_df['interval'].apply(lambda i: i.upper)
            print("done loading")
            self.data_loaded = True
            # todo: fix this for SeizuresDataset
            # self.samples_df = self.samples_df.set_index(['patient_name', 'seizure_num'])
        # self.labels = list(self.samples_df.label)
        except Exception as e:
            raise e

    def get_X(self):
        return np.vstack(self.samples_df.x)

    def get_labels(self, format='num'):
        if format == 'desc':
            return list(self.samples_df.label_desc)
        elif format == 'num':
            return list(self.samples_df.label)
        else:
            raise ValueError("incorrect format")

    @property
    def T_max(self):
        def get_interval_length(interval: Interval):
            assert isinstance(interval, Interval), "Error: interval is not of type Interval"
            return (interval.upper - interval.lower).total_seconds()

        return self.samples_df.interval.apply(get_interval_length).max()

    def get_train_x(self, crop: float = 400) -> Tensor:
        """
        return the time axis
        Args:

        Returns:

        """
        if not self.data_loaded:
            print("loading data")
            self._load_data()

        sfreq = self.metadata.get('sfreq')
        T = self.T_max
        N = sfreq * T
        crop_idx = int(sfreq * crop)
        return torch.linspace(0, T, int(N))[:crop_idx]

    @torch.no_grad()
    def get_train_y(self, num_channels: int = 2, crop: float = 400, normalize=True) -> Tensor:
        if not self.data_loaded:
            print("loading data")
            self._load_data()

        def pad(tensor, length):
            """right zero-padding on last dimension to length"""
            delta = int(length - tensor.shape[-1])
            return torch.nn.functional.pad(tensor, (0, delta), mode='constant', value=0)

        sfreq = self.metadata.get('sfreq')
        N = sfreq * self.T_max
        crop_idx = int(sfreq * crop)
        padded = torch.stack([pad(torch.tensor(x[:num_channels]), N)[..., :crop_idx] for x in self.samples_df.x])
        train_y = padded

        if normalize:
            train_y = (train_y - train_y.mean()) / train_y.std()
            # m = nn.BatchNorm1d(num_features=num_channels, dtype=torch.double)
            # train_y = m(train_y.double())

        return train_y


class SeizuresDataset(RawDataset):
    """
    A class for a dataset composed of seizures only.
    Regular instantiation (__init__()) can be used for loading an existing dataset.
    @classmethod generate_dataset() can be used to generate one from scratch.
    """

    def __init__(self, dataset_dir: str, num_channels=2, preload_data=False):
        """
        Args:
            dataset_dir:
            num_channels:
            preload_data:
        """
        super().__init__(dataset_dir)
        assert os.path.exists(dataset_dir), "error: the dataset directory does not exist"
        assert os.path.isfile(f"{dataset_dir}/samples_df.csv"), "error: samples_df.csv not found in dataset_dir"
        self.samples_df = pd.read_csv(f"{dataset_dir}/samples_df.csv", index_col=0)
        self.samples_df = self.samples_df.set_index(['patient_name', 'seizure_num'])
        self.dataset_dir = dataset_dir
        self.data_loaded = False
        self.num_channels = num_channels

        if preload_data:
            self._load_data()

    @classmethod
    def generate_dataset(cls, seizures_index_df: DataFrame, fast_dev_mode: bool = False,
                         picks: Sequence[str] = data_utils.PicksOptions.common_channels, output_dir: str = None,
                         time_minutes_before=0, time_minutes_after=0):
        """
        Gets the seizure intervals,
        Iterates over the data files
        Returns:

        """
        create_time = data_utils.get_time_as_str()
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

        intervals_and_raws: DataFrame = data_utils.add_raws_to_intervals_df(window_intervals, picks, fast_dev_mode)

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


class UniformDataset(RawDataset):
    """
    A class for a dataset composed of seizures only.
    Regular instantiation (__init__()) can be used for loading an existing dataset.
    @classmethod generate_dataset() can be used to generate one from scratch.
    """

    def __init__(self, dataset_dir: str, num_channels=2, preload_data=True, add_check_isseizure=True):
        """
        Args:
            dataset_dir:
            num_channels:
            preload_data:
            add_check_isseizure:
        """
        super().__init__(dataset_dir)
        assert os.path.exists(dataset_dir), "error: the dataset directory does not exist"
        assert os.path.isfile(f"{dataset_dir}/samples_df.csv"), "error: samples_df.csv not found in dataset_dir"

        self.samples_df = pd.read_csv(f"{dataset_dir}/samples_df.csv", index_col=0)

        # self.samples_df = self.samples_df.set_index(['window_id'])
        self.dataset_dir = dataset_dir
        self.data_loaded = False
        self.num_channels = num_channels

        if preload_data:
            self._load_data()

    @classmethod
    def generate_dataset(cls, N=1000, L=1000, fast_dev_mode: bool = False,
                         picks: Sequence[str] = data_utils.PicksOptions.common_channels, output_dir: str = None):
        """
        Gets the seizure intervals,
        Iterates over the data files
        Returns:

        """
        create_time = data_utils.get_time_as_str()
        if output_dir is None:
            # noinspection PyTypeChecker
            output_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}/UNIFORM/{create_time}"

        print(f"Creating Output Directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        metadata = {"creation_time": create_time,
                    "fast_dev_mode": fast_dev_mode,
                    "picks": picks,
                    "sfreq": config['TASK']['RESAMPLE']}

        with open(f"{output_dir}/dataset.yml", 'w') as metadata_file:
            yaml.dump(metadata, metadata_file, default_flow_style=False)

        # convert ictal intervals into window intervals (expand with time before and after)
        window_intervals: DataFrame = get_random_intervals(N=N, L=L)

        # load Raws
        print("starting to load raw files")
        intervals_and_raws: DataFrame = data_utils.add_raws_to_intervals_df(window_intervals, picks, fast_dev_mode)

        print("starting to process raw files")
        # initialize samples_df
        samples_df = pd.DataFrame(
            columns=['patient_name', 'window_interval', 'window_id', 'fname', 'label', 'label_desc'])
        counter = itertools.count()
        for window_idx, sample_row in intervals_and_raws.dropna().iterrows():
            # create samples_df row
            window_id = next(counter)
            fname = f"{output_dir}/window_{window_id}.pkl"
            # append row to samples_df
            row = {"package": sample_row.package,
                   "patient": sample_row.patient,
                   "window_interval": sample_row.window_interval,
                   "window_id": window_id,
                   "fname": fname,
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

    def add_check_isseizure(self):
        seizures_index_df = get_seizures_index_df()

        def check_isseizure(samples_row) -> bool:
            patient = samples_row['patient']
            interval = samples_row['interval']
            patient_seizures = seizures_index_df.loc[patient]
            patient_seizures['overlaps_interval'] = patient_seizures.interval.apply(
                lambda p_interval: p_interval.overlaps(interval))
            return patient_seizures['overlaps_interval'].any()

        self.samples_df['isseizure'] = self.samples_df.apply(check_isseizure, axis=1)


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

        windows_dict = {window_id: x for window_id, x in self.file_loader(dataset_dir)}
        self.samples_df["x"] = self.samples_df.apply(lambda sample: windows_dict[sample.name].reshape(-1), axis=1)

        self.samples_df['window_interval'] = self.samples_df['window_interval'].apply(self.parse_datetime_interval)
        # self.labels = list(self.samples_df.label)
        self.samples_df['lower'] = self.samples_df['window_interval'].apply(lambda i: i.lower)
        self.samples_df['upper'] = self.samples_df['window_interval'].apply(lambda i: i.upper)

        if add_next_seizure_info:
            # add time to seizure
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

    @classmethod
    def generate_dataset(cls, patient_name, selected_func, fast_dev_mode: bool = False,
                         picks: Sequence[str] = data_utils.PicksOptions.common_channels, output_dir: str = None):
        """
        Gets the seizure intervals,
        Iterates over the data files
        Returns:

        """
        if fast_dev_mode:
            print(f"WARNING! {fast_dev_mode=} !!! Results are incomplete.")

        create_time = data_utils.get_time_as_str()
        if output_dir is None:
            # noinspection PyTypeChecker
            output_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}/{selected_func}/{patient_name}/{create_time}"

        print(f"Creating Output Directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        metadata = {"creation_time": create_time,
                    "patient_name": patient_name,
                    "fast_dev_mode": fast_dev_mode,
                    "picks": picks,
                    "selected_func": selected_func,
                    "sfreq": config['TASK']['RESAMPLE']}

        with open(f"{output_dir}/dataset.yml", 'w') as metadata_file:
            yaml.dump(metadata, metadata_file, default_flow_style=False)

        # get intervals
        preictal_intervals = data_utils.get_preictal_intervals(patient_name)
        interictal_intervals = data_utils.get_interictal_intervals(patient_name)

        intervals = pd.concat([preictal_intervals, interictal_intervals])

        window_intervals = cls.intervals_to_windows(intervals)
        window_intervals['patient_name'] = patient_name
        print("starting to load raw files")
        # load Raws
        intervals_and_raws: DataFrame = data_utils.add_raws_to_intervals_df(window_intervals, picks, fast_dev_mode)

        print("starting to process raw files")
        # initialize samples_df
        samples_df = pd.DataFrame(
            columns=['patient_name', 'window_interval', 'window_id', 'fname', 'label', 'label_desc'])
        counter = itertools.count()
        for window_idx, sample_row in intervals_and_raws.dropna().iterrows():
            # create samples_df row
            window_id = next(counter)
            fname = f"{output_dir}/window_{window_id}.pkl"

            y = config['TASK'][f"{sample_row['label_desc'].upper()}_LABEL"]
            # append row to samples_df

            row = {"patient_name": sample_row.patient_name,
                   "window_interval": sample_row.window_interval,
                   "window_id": window_id,
                   "fname": fname,
                   "label": y,
                   "label_desc": sample_row['label_desc']
                   }
            samples_df = samples_df.append(row, ignore_index=True)

            # Get X
            X = sample_row.raw.get_data()

            # Scale X
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)

            # Perform Feature Extraction (Optional)
            X = data_utils.extract_feature_from_numpy(X, selected_func, float(config['TASK']['RESAMPLE']))

            # Dump to file
            print(f"dumping {window_id=} to {fname=}")
            pickle.dump(X, open(fname, 'wb'))

        samples_df_path = f"{output_dir}/dataset.csv"
        print(f"saving samples_df to {samples_df_path=}")
        samples_df.to_csv(samples_df_path)
        return cls(output_dir)

    def get_X(self):
        return np.vstack(self.samples_df.x)

    def get_labels(self, format='num'):
        if format == 'desc':
            return list(self.samples_df.label_desc)
        elif format == 'num':
            return list(self.samples_df.label)
        else:
            raise ValueError("incorrect format")

    @staticmethod
    def intervals_to_windows(intervals: DataFrame, time_minutes=5):
        def split_intervals_to_windows(group: DataFrame):
            start_times = [list(portion.iterate(interval, step=timedelta(minutes=time_minutes))) for interval in
                           group.interval]
            windows_in_lists = [
                [portion.closedopen(times[i], times[i + 1]) for i in range(len(times) - 1)] for times in
                start_times]
            windows = list(itertools.chain.from_iterable(windows_in_lists))  # chain together sublists
            windows_df = DataFrame({"window_interval": windows, "label_desc": group.name})
            return windows_df

        windows = intervals.groupby('label_desc').apply(func=split_intervals_to_windows)
        return windows


class MaskedDataset(PSPDataset):
    def __init__(self, dataset_dir: str, mask: ndarray):
        super(MaskedDataset, self).__init__(dataset_dir)
        self.mask = mask

    def get_masked_X(self):
        return self.get_X() * self.mask


if __name__ == '__main__':
    dataset_dir = r"/results/epilepsiae/UNIFORM/20220106T165558"
    dataset = UniformDataset(dataset_dir)
    samples_df = dataset.samples_df