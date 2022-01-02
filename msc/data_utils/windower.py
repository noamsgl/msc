from datetime import datetime, timedelta
from typing import List

import pandas as pd
import portion as P
from pandas import DataFrame
from portion import Interval

from msc.config import get_config

config = get_config()


def get_recording_start(package: str, patient: str) -> datetime:
    """
    Get first measurement timestamp for patient from data_index.csv
    Args:
        package:
        patient:

    Returns:

    """
    data_index_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}/data_index.csv"
    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    return min(patient_data_df.meas_date)


def get_recording_end(package: str, patient: str) -> datetime:
    """
    Get last measurement timestamp for the patient from data_index.csv
    Args:
        package:
        patient:

    Returns:

    """
    data_index_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}/data_index.csv"
    data_index_df = pd.read_csv(data_index_path, parse_dates=['meas_date', 'end_date'])

    patient_data_df = data_index_df.loc[
        (data_index_df['package'] == package) & (data_index_df['patient'] == patient)]

    return max(patient_data_df.end_date)


def get_interictal_intervals(package: str, patient: str) -> List[Interval]:
    """
    return interictal time intervals

    example: get_preictal_intervals("surfCO", "pat_4000") -> interictals=[(Timestamp('2010-03-01 10:25:24'),Timestamp('2010-03-01 19:34:12')), (Timestamp('2010-03-01 23:34:12'),Timestamp('2010-03-02 04:13:38')), (Timestamp('2010-03-02 08:13:38'),Timestamp('2010-03-02 10:18:45')), (Timestamp('2010-03-02 14:18:45'),Timestamp('2010-03-02 15:27:23')), (Timestamp('2010-03-02 19:27:23'),Timestamp('2010-03-02 23:07:14')), (), (Timestamp('2010-03-03 04:10:30'),Timestamp('2010-03-04 09:07:01'))]

    Args:
        package: | example "surfCO"
        patient: | example "pat_4000"

    Returns:

    """
    min_diff = timedelta(hours=float(config['TASK']['INTERICTAL_MIN_DIFF_HOURS']))
    recording_start = get_recording_start(package, patient)
    recording_end = get_recording_end(package, patient)

    onsets = get_seiz_onsets(package, patient)

    first_interictal = P.open(recording_start, onsets[0] - min_diff)
    middle_interictals = [P.open(onsets[i] + min_diff, onsets[i + 1] - min_diff) for i in range(0, len(onsets) - 1)]
    last_interictal = P.open(onsets[-1] + min_diff, recording_end)
    interictals = [first_interictal] + middle_interictals + [last_interictal]
    return interictals


def get_preictal_intervals(package: str, patient: str) -> List[Interval]:
    """
    return preictal time intervals

    example: get_preictal_intervals("surfCO", "pat_4000") -> preictals=[(Timestamp('2010-03-01 20:34:12'), Timestamp('2010-03-01 21:34:12')), (Timestamp('2010-03-02 05:13:38'), Timestamp('2010-03-02 06:13:38')), (Timestamp('2010-03-02 11:18:45'), Timestamp('2010-03-02 12:18:45')), (Timestamp('2010-03-02 16:27:23'), Timestamp('2010-03-02 17:27:23')), (Timestamp('2010-03-03 00:07:14'), Timestamp('2010-03-03 01:07:14')), (Timestamp('2010-03-03 01:10:30'), Timestamp('2010-03-03 02:10:30'))]
    Args:
        package: | example "surfCO"
        patient: | example "pat_4000"

    Returns:

    """
    onsets = get_seiz_onsets(package, patient)
    preictals = [P.open(onset - timedelta(hours=float(config['TASK']['PREICTAL_MIN_DIFF_HOURS'])), onset) for onset in
                 onsets]
    return preictals


def get_ictal_intervals(seizures_index_df: DataFrame):
    """Get the ictal (seizure) intervals from the seizures_index_df
    # todo: implement this
    """



    raise NotImplementedError()


def get_seiz_onsets(package: str, patient: str) -> List[datetime]:
    """
    returns seizure onset times.
    example: get_seiz_onsets("surfCO", "pat_4000") -> onsets=[Timestamp('2010-03-01 21:34:12'), Timestamp('2010-03-02 06:13:38'), Timestamp('2010-03-02 12:18:45'), Timestamp('2010-03-02 17:27:23'), Timestamp('2010-03-03 01:07:14'), Timestamp('2010-03-03 02:10:30')]
    Args:
        package: | example "surfCO"
        patient: | example "pat_4000"

    Returns:

    """
    seizures_index_path = f"{config['PATH'][config['RAW_MACHINE']]['RAW_DATASET']}/seizures_index.csv"

    seizures_index_df = pd.read_csv(seizures_index_path, parse_dates=['onset', 'offset'])

    patient_seizures_df = seizures_index_df.loc[
        (seizures_index_df['package'] == package) & (seizures_index_df['patient'] == patient)]

    return list(patient_seizures_df.onset)
