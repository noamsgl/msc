"""Build the dataset for the Advanced Topics in Physiological Signal Processing Course
* Clean and standardize the data (artifact removal, zero mean & unit variance per data file)
* Resample to sfreq (256 in original paper)
* Split the dataset to train/test
* Split the dataset into 5 min windows
* Transform each 5s epoch into a feature vector
* Save each 60 feature vectors as a pattern

optional features: ['max_cross_corr', 'phase_lock_val', 'nonlin_interdep', 'time_corr', 'spect_corr']

References
[1] Mirowski, Piotr, et al. "Classification of patterns of EEG synchronization for seizure prediction." Clinical neurophysiology 120.11 (2009): 1927-1940.
[2] Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT,
“An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings” Proc. IEEE ICASSP Conf. 2018
"""
import argparse
import os.path

from msc.config import get_config
from msc.data_utils import get_time_as_str, PicksOptions
from msc.dataset import PSPDataset


def main(raw_args=None):
    config = get_config()

    parser = argparse.ArgumentParser(description='build a eeg prediction dataset')
    parser.add_argument('-d', '--dev',
                        action='store_true',
                        help='whether to run in fast_dev_mode')
    parser.add_argument('-p', '--patient',
                        action='append',
                        help='patient id',
                        required=True,
                        default=[])
    parser.add_argument('-f', '--feature',
                        action='append',
                        help='which features to extract',
                        required=True,
                        default=[])
    parser.add_argument('-m', '--machine',
                        help="which machine to get paths from (options 'LOCAL', 'BGUCLUSTER', 'MIRIAM')",
                        required=False,
                        default='MIRIAM')
    parser.add_argument('-i', '--input',
                        help="path to input raw data dir",
                        required=False,
                        default=config['PATH'][config['RAW_MACHINE']]['RAW_DATASET'])
    parser.add_argument('-o', '--output',
                        help="path to output results dir",
                        required=False,
                        default=config['PATH'][config['RESULTS_MACHINE']]['RESULTS'])
    args = parser.parse_args(raw_args)
    fast_dev_mode = args.dev
    patients = args.patient
    features = args.feature

    dataset_timestamp = get_time_as_str()

    print(f"Starting {os.path.basename(__file__)} with {fast_dev_mode=} at time {dataset_timestamp}")

    for selected_func in features:
        for patient in patients:
            # get package
            picks = PicksOptions.common_channels
            PSPDataset.generate_dataset(patient, selected_func, picks=picks, fast_dev_mode=False)


if __name__ == '__main__':
    main()
