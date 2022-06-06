import pandas as pd
import zarr

def get_samples_df(dataset_id:str, with_events=False) -> pd.DataFrame:
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

    # create samples_df DataFrame
    samples_df = pd.DataFrame(samples_df)
    return samples_df