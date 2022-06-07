import datetime
import matplotlib.pyplot as plt
import pandas as pd

from msc import config
from msc.datamodules.data_utils import IEEGDataFactory
from msc.plot_utils import set_size

plt.style.use(['science', 'no-latex'])

SEC = 1e6
MIN = 60 * SEC
HOUR = 60 * MIN

def get_dataset(dataset_id):
    # get dataset from iEEG.org
    ds = IEEGDataFactory.get_dataset(dataset_id)
    return ds
    
def karoly_prior(t):
    ds = get_dataset(config['dataset_id'])

def get_events_df(events) -> pd.DataFrame:
    events_df = pd.DataFrame(events, columns=['datetime'])
    events_df['year'] = events_df['datetime'].dt.year
    events_df['month'] = events_df['datetime'].dt.month
    events_df['day'] = events_df['datetime'].dt.day
    events_df['hour'] = events_df['datetime'].dt.hour
    events_df['minute'] = events_df['datetime'].dt.minute
    events_df['second'] = events_df['datetime'].dt.second
    return events_df

if __name__ == "__main__":
    ds = get_dataset(config['dataset_id'])
    start_time = datetime.datetime.fromtimestamp(ds.start_time / SEC, datetime.timezone.utc)

    seizures = ds.get_annotations('seizures')
    events =  [start_time + datetime.timedelta(microseconds=seizure.start_time_offset_usec) for seizure in seizures]
    
    events_df = get_events_df(events)

    plt.clf()
    width = 478  #pt
    events_df.hist('hour', bins=24, figsize=set_size(width))
    plt.title("Circadian Seizure Distribution")
    plt.savefig(f"{config['path']['figures']}/prior/hist.pdf", bbox_inches='tight')


