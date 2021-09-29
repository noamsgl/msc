import argparse
from time import gmtime, strftime

import mne
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import scipy.io

import msc

# get command line arguments
parser = argparse.ArgumentParser(description="Run BOCD with EEG Data")
parser.add_argument('-i', '--input', help="input filepath pointing to a Nicolete EEG .data", required=False, type=str,
                    default=r'input/raw_eeg/8010200_0006.data')
parser.add_argument('-on', '--onset', help="seizure onset time (relative to file) in seconds", required=False, type=int,
                    default=387)
parser.add_argument('-off', '--offset', help="seizure offset time (relative to file) in seconds", required=False,
                    type=int, default=439)
parser.add_argument('-o', '--output', help="output base path", required=False, type=str, default='output/exp2')
parser.add_argument('-m', '--margin', help="margin data to take around seizure in %percent", required=False, type=int,
                    default=30)
parser.add_argument('-p', '--picks', help="channel picks", nargs='+', required=False, type=str,
                    default=['all'])

args = vars(parser.parse_args())

# extract arguments
output_path = args['output']
raw_path = args['input']
onset = args['onset']
offset = args['offset']
margin = args['margin']
picks = args['picks']
argv0 = sys.argv[0]
print(f"running {argv0} with the following arguments:\n {args}")

# constants
# https://physionet.org/files/eegmmidb/1.0.0/64_channel_sharbrough.png
CHANNELS = ['FP1', 'AF7', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3',
            'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'AF8', 'O2', 'F9',
            'F10', 'T9', 'T10', 'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8', 'MT1+',
            'MT2+', 'EOG+', 'ECG']

if picks != ["all"]:
    CHANNELS = picks

# read data to DataFrame
raw_path = r'input/raw_eeg/8010200_0006.data'
extension = os.path.splitext(raw_path)[1]
if extension == '.data':
    # file is stored in raw nicolet format
    seizure_length = (offset - onset)
    absolute_margin = (margin/100) * seizure_length
    raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True).crop(onset - absolute_margin,
                                                                              offset + absolute_margin)
    df = raw.to_data_frame()

else:
    raise ValueError("the data extension is incompatible")

# apply vanilla BOCD to each channel separately
for channel in CHANNELS:
    print(f"beginning bocd on channel {channel}, margin {margin}% around seizure")
    chn_output_path = os.path.join(output_path, str(margin), channel)
    os.makedirs(chn_output_path, exist_ok=True)
    data = raw.get_data(channel)

    changepoints = msc.bocd.get_BOCD_changepoints(data)

    # print results to file
    with open(os.path.join(chn_output_path, f"{channel}_bocd.txt"), 'w') as f:
        f.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        f.write('\n')
        f.write(str(changepoints))

    # plot changepoints and save in chn_output_path as png
    msc.bocd.plot_BOCD_changepoints_and_seizure(data, changepoints, raw.info["sfreq"], onset, offset, chn_output_path, channel)
