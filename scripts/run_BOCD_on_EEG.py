import argparse
from time import gmtime, strftime

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
parser.add_argument('-l', '--limit', help="limit data length in seconds", required=False, type=int, default=30)
parser.add_argument('-p', '--picks', help="channel picks", nargs='+', required=False, type=str, default=['FP1', 'AF7', 'T8', 'PZ', 'O1', 'ECG'])

args = vars(parser.parse_args())

# extract arguments
output_path = args['output']
raw_path = args['input']
limit = args['limit']
picks = args['picks']
argv0 = sys.argv[0]
print(f"running {argv0} with:\n {args}")


# constants
CHANNELS = ['FP1', 'AF7', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3',
            'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'AF8', 'O2', 'F9',
            'F10', 'T9', 'T10', 'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8', 'MT1+',
            'MT2+', 'EOG+', 'ECG']
            
if picks != ["all"]:
  CHANNELS = picks

for limit in [10, 30, 120, 240]:
  # read limited data to DataFrame
  raw_path = r'input/raw_eeg/8010200_0006.data'
  raw = mne.io.read_raw_nicolet(raw_path, ch_type='eeg', preload=True).crop(0, limit)
  df = raw.to_data_frame()
  
  
  # extract BOCD on channels
  for channel in CHANNELS:
      print(f"beginning bocd on channel {channel}, limit {limit}")
      chn_output_path = os.path.join(output_path, str(limit), channel)
      os.makedirs(chn_output_path, exist_ok=True)
      data = df[channel]
      
      changepoints = msc.bocd.get_BOCD_changepoints(data)
      # print to file
      with open(os.path.join(chn_output_path, f"{channel}_bocd.txt"), 'w') as f:
        f.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        f.write('\n')
        f.write(str(changepoints))
    
      # plot changepoints
      msc.bocd.plot_BOCD_changepoints(data, changepoints, raw.info["sfreq"], chn_output_path, channel)
      
      