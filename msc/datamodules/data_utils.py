from torch import Tensor

from ieegpy.ieeg.auth import Session
from ieegpy.ieeg.dataset import Dataset
from msc.config import get_authentication


class SingleSampleDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.samples = ((x, y),)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.samples[idx]



class IEEGDataFactory:
    def __init__(self, dataset_id) -> None:
        self.dataset_id = dataset_id
        
    
    @classmethod 
    def get_dataset(cls, dataset_id) -> Dataset:
        username, password = get_authentication()
        with Session(username, password) as s:# start streaming session
            ds = s.open_dataset(dataset_id)  # open dataset stream
        return ds
    