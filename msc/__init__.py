__version__ = "0.1.0a1"

from .config import get_config
from msc.dataset import get_data_index_df, get_seizures_index_df

config = get_config()
data_index_df = get_data_index_df()
seizures_index_df = get_seizures_index_df()