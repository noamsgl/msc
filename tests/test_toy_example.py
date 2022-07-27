import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import seaborn as sns
import unittest

from msc.plot_utils import set_size
from msc.config_utils import config


class TestBSLEToyExample(unittest.TestCase):
    """Test BSLE algorithm on a one dimensional case"""

    def test_mh_mcmc(self) -> None:
        """ensure the mh_mcmc alg returns good samples from posterior"""
        pass
    TODO: write BSLE unit tests