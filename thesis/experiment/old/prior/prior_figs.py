"""Run the scripts for prior figures
"""


import matplotlib.pyplot as plt
import numpy as np


class Novelty:
    def __init__(self) -> None:
        pass
    



class EEGIter:
    def __init__(self, t, skipNaN=True) -> None:
        """_summary_
        Iterate  iEEG.org dataset in order
        Args:
            t (number): current time
            skipNan (bool): whether to skip missing values
        """
        self.t = t
        
    def def __iter__(self):
      self.t = 1
      return self
        
return iter


class EEG:
    def __init__(self) -> None:
        pass
    
    def get_segment_iterator(self):

        
eeg = EEG()

data_iter = eeg.get_segment_iterator()

for t, data_t in data_iter:
    novelty = Novelty(data_t)
