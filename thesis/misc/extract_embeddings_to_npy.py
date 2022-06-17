import numpy as np

from msc import config
from msc.cache_handler import get_samples_df


dataset_id = str(config['dataset_id'])
t_max = config['t_max']

# load samples_df
samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)



embeddings = np.stack(samples_df['embedding'])  # type: ignore

times = np.stack(samples_df['time'])  # type: ignore

np.save(f"{config['path']['data']}/{config['dataset_id']}/embeddings.npy", embeddings)
np.save(f"{config['path']['data']}/{config['dataset_id']}/times.npy", times)