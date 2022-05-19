
from ieegpy.ieeg.auth import Session
from ieegpy.ieeg.dataset import Dataset
from msc.config import get_authentication


class IEEGDataFactory:
    def __init__(self, dataset_id) -> None:
        self.dataset_id = dataset_id
        
    
    @classmethod 
    def get_dataset(cls, dataset_id) -> Dataset:
        username, password = get_authentication()
        s = Session(username, password)  # start streaming session
        ds = s.open_dataset(dataset_id)  # open dataset stream
        return ds