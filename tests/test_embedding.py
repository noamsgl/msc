"""Estimate class tests
"""
from black import assert_equivalent
import numpy as np
from numpy.testing import assert_array_equal
import unittest
from unittest import assertEqual
from sklearn.utils import assert_all_finite

from msc.embedding import GPEmbeddor
from msc.data import IEEGDataFactory

TIME: float = 5
DURATION = 10
DATASET_ID: str = 'I004_A0003_D001'
NUM_CHANNELS = 2
SFREQ = 400


class TestEmbeddorComputesEmbeddings(unittest.TestCase):
    """Test that the Embeddor computes embeddings"""
    
    def test_embeddor_computes_embeddings(self):
        """Ensures the embeddor recieves data as expected"""
        dataset = IEEGDataFactory.get_dataset(DATASET_ID)
        data = dataset.get_data(TIME, DURATION, np.arange(NUM_CHANNELS))
        
        self.assertEqual(data.shape, [SFREQ * DURATION, NUM_CHANNELS])
        
        embeddor = GPEmbeddor()
        embedding = GPEmbeddor.embed(data)
        
        assert_all_finite(embedding)


if __name__ == '__main__':
    unittest.main()