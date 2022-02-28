import datetime
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import List, OrderedDict

import torch
import xmltodict


def get_first_create_time(data_dict: dict):
    annotations: List[OrderedDict] = data_dict['annotations']['annotation']
    create_times = [annotations[i].get('@createTime') for i in range(len(annotations))]
    create_time = create_times[0]
    # parse string to datetime
    create_time = datetime.datetime.strptime(create_time, "%Y-%m-%dT%H:%M:%SZ")
    return create_time


def get_record_start(dog_num: int):
    xml_path = fr"C:\raw_data\canine_db\packages\Dog {dog_num}\annotations.xml"
    tree = ET.parse(xml_path)

    xml_data = tree.getroot()

    xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')

    data_dict = dict(xmltodict.parse(xmlstr))

    # we assume the first create time is the recording start time (in fact all create times should be equal)
    record_start = get_first_create_time(data_dict)
    return record_start


def get_onsets(dog_num: int):
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 1\annotations.xml"
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 2\annotations.xml"
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 3\annotations.xml"
    xml_path = fr"C:\raw_data\canine_db\packages\Dog {dog_num}\annotations.xml"
    tree = ET.parse(xml_path)

    xml_data = tree.getroot()

    xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')

    data_dict = dict(xmltodict.parse(xmlstr))

    # we assume the first create time is the recording start time (in fact all create times should be equal)
    record_start = get_record_start(dog_num=dog_num)
    # record_start = datetime.datetime(year=2014, month=1, day=1)

    annotations: List[OrderedDict] = data_dict['annotations']['annotation']
    seizure_onsets = [record_start + timedelta(microseconds=int(annotations[i].get('@startOffsetUsecs'))) for i in
                      range(len(annotations))]

    return seizure_onsets


def get_label_desc_from_fname(fname: str):
    if 'interictal' in fname:
        label_desc = 'interictal'
    elif 'ictal' in fname:
        label_desc = 'ictal'
    elif 'test' in fname:
        label_desc = 'test'
    else:
        raise ValueError(f"unknown label description, {fname=}")
    return label_desc


def get_ipp_training_data(dog_num, hparams):
    record_start = get_record_start(dog_num)

    onset_datetimes = get_onsets(dog_num)

    onsets_real = torch.Tensor(
        [(onset - record_start).total_seconds() / hparams['real_time_step'] for onset in onset_datetimes])

    train_x = torch.arange(0, max(onsets_real))
    train_y = torch.zeros_like(train_x).index_fill_(0, onsets_real.long(), 1)

    # instantiate inducing points
    onsets_inducing = torch.Tensor(
        [(onset - record_start).total_seconds() / hparams['inducing_time_step'] for onset in onset_datetimes])
    inducing_points = torch.arange(0, max(onsets_inducing))

    return train_x, train_y, inducing_points
