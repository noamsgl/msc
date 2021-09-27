import argparse

import mne
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

import msc

# get command line arguments
# todo: add channel pick
parser = argparse.ArgumentParser(description="Run BOCD with EEG Data")
parser.add_argument('-i', '--input', help="input filepath pointing to a Nicolete EEG .data", required=False, type=str,
                    default=r'input/raw_eeg/8010200_0006.data')
parser.add_argument('-o', '--output', help="output base path", required=False, type=str, default='output')
parser.add_argument('-l', '--limit', help="limit data length", required=False, type=int, default=10000)
parser.add_argument('-t', '--test', help="toggle test mode", required=False, type=bool, default=True)

args = vars(parser.parse_args())

# extract arguments
output_path = args['output']
raw_path = args['input']
test = args['test']
limit = args['limit']

argv0 = sys.argv[0]
print(f"running {argv0} with test={test}, limit={limit}")


# constants
CHANNELS = ['FP1', 'AF7', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3',
            'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'AF8', 'O2', 'F9',
            'F10', 'T9', 'T10', 'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8', 'MT1+',
            'MT2+', 'EOG+', 'ECG']
            


# read data as DataFrame
raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True)
df = raw.to_data_frame()

# for TEST
if test:
  CHANNELS = ['FP1']
  


# extract BOCD on channels
for channel in CHANNELS:
    print(f"beginning bocd on channel {channel}")
    output_channel_path = os.path.join(output_path, channel)
    os.makedirs(output_channel_path, exist_ok=True)
    data = df[channel]
    
    # trim data
    if test:
      data = data[:limit]
    changepoints = msc.bocd.get_BOCD_changepoints(data)
    msc.bocd.plot_BOCD_changepoints(data, changepoints, output_channel_path + "/" + f"{channel}_bocd.png", channel)

