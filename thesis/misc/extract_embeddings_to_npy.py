import numpy as np

from msc import config
from msc.cache_handler import get_samples_df


dataset_id = str(config['dataset_id'])
t_max = config['t_max']

# load samples_df
samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)

embeddings = np.stack(samples_df['embedding'])  # type: ignore

times = np.stack(samples_df['time'])  # type: ignore
event_times = samples_df.loc[samples_df['is_event'], "onset"].to_numpy()

to_save = False
if to_save:
    np.save(f"{config['path']['data']}/{config['dataset_id']}/embeddings.npy", embeddings)
    np.save(f"{config['path']['data']}/{config['dataset_id']}/times.npy", times)
    np.save(f"{config['path']['data']}/{config['dataset_id']}/event_times.npy", event_times)
