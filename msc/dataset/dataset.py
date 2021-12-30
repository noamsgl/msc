import glob
import itertools
import os
import pickle
import re
from typing import Sequence

import numpy as np
import pandas as pd
import portion
from numpy import ndarray
from pandas import DataFrame

from msc.config import get_config
from msc.data_utils.load import get_raws_from_data_and_intervals, get_interval_from_raw
from msc.data_utils.windower import get_ictal_intervals
from msc.dataset.build_dataset import intervals_to_windows


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
        data_index_fpath = f"{config['PATH']['LOCAL']['RAW_DATASET']}/data_index.csv"

        data_index_df = pd.read_csv(data_index_fpath, index_col=0, parse_dates=['meas_date', 'end_date'])
    return data_index_df


@static_vars(seizures_index_df=None)
def get_seizures_index_df():
    if get_seizures_index_df.seizures_index_df is None:
        # get config
        config = get_config()
        # noinspection PyTypeChecker
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
    A class for generating a dataset composed of seizures only
    """

    def __init__(self, data_index_df: DataFrame, fast_dev_mode: bool, picks: Sequence[str]):
        """
         todo: implement self._write_metadata()
        Args:
            data_index_df:
            fast_dev_mode:
            picks:
        """
        baseDataset.__init__(self, fast_dev_mode)
        self.data_index_df = data_index_df
        self.picks = picks

    def _generate_dataset(self, output_dir: str = None):
        """
        Gets the seizure intervals,
        Iterates over the data files
        Returns:

        """
        config = get_config()

        if output_dir is None:
            # noinspection PyTypeChecker
            output_dir = f"{config['PATH'][config['RESULTS_MACHINE']]['RESULTS']}/{config['DATASET']}/SEIZURES"
        samples_df_path = f"{output_dir}/dataset.csv"

        print(f"Creating Output Directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # initialize samples_df
        samples_df = pd.DataFrame(columns=['package', 'patient', 'interval',
                                           'window_id', 'fname', 'label', 'label_desc'])

        # getting seizure intervals
        ictal_intervals = get_ictal_intervals(self.data_index_df)

        # splitting into windowed intervals
        ictal_window_intervals = intervals_to_windows(ictal_intervals)

        # load Raws
        ictal_raws = get_raws_from_data_and_intervals(self.data_index_df, self.picks, ictal_window_intervals,
                                                      self.fast_dev_mode)

        print("starting to process raw files")
        counter = itertools.count()
        for raw in ictal_raws:
            # create samples_df row
            interval = get_interval_from_raw(raw)
            window_id = next(counter)
            fname = f"{output_dir}/window_{window_id}.pkl"
            y = config['TASK']['PREICTAL_LABEL']
            row = {"patient_name": raw.info['patient_name'],
                   "interval": interval,
                   "window_id": window_id,
                   "fname": fname,
                   "label": y,
                   "label_desc": "preictal"}
            samples_df = samples_df.append(row, ignore_index=True)

            # Get X
            X = raw.get_data()

            # Scale X
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)

            # Perform Feature Extraction (Optional)
            # X = mne_features.feature_extraction.FeatureExtractor(sfreq=config['TASK']['RESAMPLE'],
            #                                                      selected_funcs=selected_funcs).fit_transform(X)
            # X = extract_feature_from_numpy(X, selected_func, float(config['TASK']['RESAMPLE']))

            # Dump to file
            print(f"dumping {window_id=} to {fname=}")
            pickle.dump(X, open(fname, 'wb'))

        samples_df.to_csv(samples_df_path)
        print(f"saving samples_df to {samples_df_path=}")
        return

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
