import pandas as pd
import zarr

from .data_utils import get_dataset

from .data_utils import get_event_sample_times

def get_samples_df(dataset_id:str, with_events=False, with_time_to_event=True) -> pd.DataFrame:
    cache_path = r"data/cache.zarr"

    cache_zarr = zarr.open(cache_path, 'r')

    ds_zarr = cache_zarr[f"{dataset_id}"]

    # collect embeddings from ds_zarr
    embeddings = []
    exclude_groups = ['std', 'mu', 'events']
    samples_df = []
    for key in sorted([int(k) for k in ds_zarr.keys() if k not in exclude_groups]):
        time_zarr = ds_zarr[f'{key}']
        if 'embedding' in time_zarr:
            embedding = time_zarr['embedding'][1:9]
            embeddings.append(embedding)
            data = {"time": key,
                    "embedding": embedding}
            samples_df.append(data)

    if with_events:
        # collect embeddings from events_zarr
        events_zarr = ds_zarr['events']
        for key in sorted([int(k) for k in events_zarr.keys() if k not in exclude_groups]):
            time_zarr = events_zarr[f'{key}']
            if 'embedding' in time_zarr:
                embedding = time_zarr['embedding'][1:9]
                embeddings.append(embedding)
                data = {"time": key,
                        "embedding": embedding}
                samples_df.append(data)
    
    if with_time_to_event:
        # collect events from ieeg.org  #TODO: get from cache
        ds = get_dataset(dataset_id)
        events = get_event_sample_times(ds, augment=False)
        # events_zarr = ds_zarr['events']
        # events = list(sorted(events_zarr.key()))        

        # compute time to event and add to samples_df
        events_df = pd.DataFrame(events, columns=['onset'])
        events_df = events_df.sort_values(by='onset', ignore_index=True)

        # create samples_df DataFrame
        samples_df = pd.DataFrame(samples_df)
        samples_df = samples_df.sort_values(by='time', ignore_index=True)
        samples_df = pd.merge_asof(samples_df, events_df, left_on='time', right_on='onset', direction='forward')
        samples_df['time_to_event'] = samples_df['onset'] - samples_df['time']

    # create samples_df DataFrame
    samples_df = pd.DataFrame(samples_df)
    return samples_df