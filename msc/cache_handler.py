import datetime
import pandas as pd
import zarr

from .data_utils import get_dataset
from .data_utils import get_event_sample_times
from .config_utils import config
from .time_utils import uSEC


def get_samples_df(
    dataset_id: str, with_wall_time=True, with_events=False, with_time_to_event=False
) -> pd.DataFrame:
    cache_path = f"{config['path']['data']}/cache.zarr"
    cache_zarr = zarr.open(cache_path, "r")

    ds_zarr = cache_zarr[f"{dataset_id}"]

    # collect embeddings from ds_zarr
    embeddings = []
    exclude_groups = ["std", "mu", "events"]
    samples_df = []

    if with_wall_time:
        # get dataset's start time
        ds = get_dataset(dataset_id)
        start_time = datetime.datetime.fromtimestamp(
            ds.start_time * uSEC, datetime.timezone.utc
        )
    else:
        start_time = datetime.datetime(year=2000, month=1, day=1)

    for key in sorted([int(k) for k in ds_zarr.keys() if k not in exclude_groups]):
        time_zarr = ds_zarr[f"{key}"]
        if "embedding" in time_zarr:
            embedding = time_zarr["embedding"][1:-2]
            embeddings.append(embedding)

            data = time_zarr["data"][:]
            wall_time = start_time + datetime.timedelta(seconds=key)
            row_dict = {"time": key, "wall_time": wall_time, "data": data, "embedding": embedding}
            samples_df.append(row_dict)

    if with_events:
        # collect embeddings from events_zarr
        events_zarr = ds_zarr["events"]
        for key in sorted(
            [int(k) for k in events_zarr.keys() if k not in exclude_groups]
        ):
            time_zarr = events_zarr[f"{key}"]
            if "embedding" in time_zarr:
                embedding = time_zarr["embedding"][1:9]
                embeddings.append(embedding)
                row_dict = {"time": key, "embedding": embedding}
                samples_df.append(row_dict)

    if with_time_to_event:
        # collect events from ieeg.org  #TODO: get from cache
        ds = get_dataset(dataset_id)
        events = get_event_sample_times(ds, augment=False)
        # events_zarr = ds_zarr['events']
        # events = list(sorted(events_zarr.key()))

        # compute time to event and add to samples_df
        events_df = pd.DataFrame(events, columns=["onset"])
        events_df = events_df.sort_values(by="onset", ignore_index=True)

        # create samples_df DataFrame
        samples_df = pd.DataFrame(samples_df)
        samples_df = samples_df.sort_values(by="time", ignore_index=True)
        samples_df = pd.merge_asof(
            samples_df, events_df, left_on="time", right_on="onset", direction="forward"
        )
        samples_df["time_to_event"] = samples_df["onset"] - samples_df["time"]
        samples_df["is_event"] = samples_df["time_to_event"].apply(
            lambda x: True if x == 0 else False
        )

    # create samples_df DataFrame
    samples_df = pd.DataFrame(samples_df)
    return samples_df
