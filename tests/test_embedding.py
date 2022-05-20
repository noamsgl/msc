"""Estimate class tests
"""
import subprocess
import sys

# TODO: move to archive
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package]
# install('debugpy')

import numpy as np
import unittest
from sklearn.utils import assert_all_finite

from msc.embedding import GPEmbeddor
from msc.data import IEEGDataFactory

import debugpy
print("listening to client on localhost:5678")
debugpy.listen(5678)
print("waiting for client to attach")
debugpy.wait_for_client()

TIME: float = 5
DURATION = 10
DATASET_ID: str = 'I004_A0003_D001'
NUM_CHANNELS = 2


class CustomAssertions:
    def assertShapesSimilar(self, first, second, tolerance=0):
        if len(first) != len(second):
            raise AssertionError(f"Shape dimension mismatch: {len(first)=} != {len(second)=}")
        for i in range(len(first)):
            if abs(first[i] - second[i]) > tolerance:
                raise AssertionError(f"Shapes mismatch at dimension {i}")

class TestEmbeddorComputesEmbeddings(unittest.TestCase):
    """Test that the Embeddor computes embeddings"""
    
    def test_embeddor_computes_embeddings(self):
        """Ensures the embeddor recieves data as expected"""
        dataset = IEEGDataFactory.get_dataset(DATASET_ID)
        data = dataset.get_data(TIME, DURATION, np.arange(NUM_CHANNELS))
        assert_all_finite(data)
        
        embeddor = GPEmbeddor()
        embedding = GPEmbeddor.embed(data)
        

        dataset.close()
        
if __name__ == '__main__':
    unittest.main()