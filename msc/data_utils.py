import numpy as np


def count_nans(data):
    """return count of nan entries"""
    return np.count_nonzero(np.isnan(data))

def prop_nans(data):
    """"return proportion of nan entries"""
    return count_nans(data) / data.size