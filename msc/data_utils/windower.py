from datetime import datetime, timedelta
from typing import List

import pandas as pd
import portion as P

# read local file `config.ini`
from msc.config import get_config

config = get_config()


def get_recording_start(package: str, patient: str) -> datetime:
    data_index_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}/data_index.csv"
    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    return min(patient_data_df.meas_date)


def get_recording_end(package: str, patient: str) -> datetime:
    data_index_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}/data_index.csv"
    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    return max(patient_data_df.end_date)


def get_interictal_times(package: str, patient: str) -> List[datetime]:
    min_diff = timedelta(hours=float(config.get('DATA', 'INTERICTAL_MIN_DIFF_HOURS')))
    recording_start = get_recording_start(package, patient)
    recording_end = get_recording_end(package, patient)

    onsets = get_seiz_onsets(package, patient)

    first_interictal = P.open(recording_start, onsets[0] - min_diff)
    middle_interictals = [P.open(onsets[i] + min_diff, onsets[i + 1] - min_diff) for i in range(0, len(onsets) - 1)]
    last_interictal = P.open(onsets[-1] + min_diff, recording_end)

    return [first_interictal] + middle_interictals + [last_interictal]


def get_preictal_times(package: str, patient: str) -> List[datetime]:
    onsets = get_seiz_onsets(package, patient)
    preictals = [(onset - timedelta(hours=1), onset) for onset in onsets]
    return preictals


def get_seiz_onsets(package: str, patient: str) -> List[datetime]:
    seizures_index_path = f"{config.get('DATA', 'DATASETS_PATH_LOCAL')}/{config.get('DATA', 'DATASET')}/seizures_index.csv"

    seizures_index_df = pd.read_csv(seizures_index_path, parse_dates=['onset', 'offset'])

    patient_seizures_df = seizures_index_df.loc[
        (seizures_index_df['package'] == package) & (seizures_index_df['patient'] == patient)]

    return list(patient_seizures_df.onset)
