import mne
import os


def load_data(fpath):
    extension = os.path.splitext(fpath)[-1]
    if extension == ".data":
        raw = mne.io.read_raw_nicolet(fpath, ch_type='eeg')
    elif extension == ".edf":
        raw = mne.io.read_raw_edf(fpath)
    else:
        raise ValueError("unknown extension. currently supports: .data, .edf")
    return raw