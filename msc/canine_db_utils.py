import datetime
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import List, OrderedDict

import xmltodict


def get_onsets(dog_num: int):
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 1\annotations.xml"
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 2\annotations.xml"
    # xml_path = r"C:\raw_data\canine_db\packages\Dog 3\annotations.xml"
    xml_path = fr"C:\raw_data\canine_db\packages\Dog {dog_num}\annotations.xml"
    tree = ET.parse(xml_path)

    xml_data = tree.getroot()

    xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')

    data_dict = dict(xmltodict.parse(xmlstr))

    # todo: get real record_start
    record_start = datetime.datetime(year=2014, month=1, day=1)

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
        raise ValueError("unknown label desc")
    return label_desc
