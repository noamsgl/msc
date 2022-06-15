import argparse
import hydra
from hydra import initialize, compose
import numpy as np

import zarr

from msc import config
from msc.data_utils import get_dataset
from msc.models.embedding import GPEmbeddor
from msc import get_logger
from msc.logs import nan_report


def embed(job_code, dataset_id, duration, num_channels, into_events) -> None:
    logger.info(f"beginning embedding with {job_code=} {dataset_id=} {duration=} {num_channels=}, {into_events=}")
    # get dataset
    ds = get_dataset(dataset_id)

    # parallel access synchronization
    synchronizer = zarr.ProcessSynchronizer(f"{config['path']['data']}/embed.sync")
    cache_zarr = zarr.open(f"{config['path']['data']}/cache.zarr", 'a', synchronizer=synchronizer)
    assert f"{config['dataset_id']}" in cache_zarr
    ds_zarr = cache_zarr[f"{config['dataset_id']}"]
    assert ('mu' in ds_zarr)
    assert ('std' in ds_zarr)

    mu = ds_zarr['mu'][:]
    logger.info(f"{mu=}")
    std = ds_zarr['std'][:]
    logger.info(f"{std=}")
    
    
    # get times from times zarr
    job_inputs_zarr = zarr.open(f"{config['path']['data']}/job_inputs.zarr", mode='r')
    times_zarr = job_inputs_zarr[f"{job_code}/times"]
    
    # get data & results zarr
    for t in times_zarr:
        logger.info(f"beginning embedding of time {t=}")
        # initialize t_zarr (a zarray for time t)
        if into_events:
            events_zarr = ds_zarr["events"]
            t_zarr = events_zarr.require_group(f"{t}")
        else:
            t_zarr = ds_zarr.require_group(f"{t}")
        
        # skip if embedding is in cache
        if 'embedding' in t_zarr:
            logger.info(f"'embedding' in t_zarr at time {t}, skipping")
            continue
        
        # dont redownload if data is in cache
        if 'data' in t_zarr:
            logger.info(f"'data' in t_zarr at time {t}, loading from cache")
            data = t_zarr['data'][:]

        else:
            # get data from iEEG.org
            data = ds.get_data(t, duration, np.arange(num_channels))

            # normalize data
            data = (data - mu)/std

            # save data to {t}/data
            data_zarr = t_zarr.zeros('data', shape=data.shape, dtype=data.dtype)
            data_zarr[:] = data

        # skip if data contains nans
        logger.info(f"{data.shape=}")
        if not np.all(np.isfinite(data)):
            logger.warning(f"Warning: {nan_report(data)}. skipping.")
            continue

        # initialize gp
        with initialize(config_path="../../config/embeddor/"):
            cfg = compose(config_name="gp", overrides=[])
            gp : GPEmbeddor = hydra.utils.instantiate(cfg.embeddor)
            # initialize pytorch-lightning logging directory
            pl_logger_dirpath = f"{config['path']['lightning_logs']}/{dataset_id}/{t}"
            # begin embedding process
            embedding = gp.embed(data, pl_logger_dirpath)
            # save embedding to cache[t]/embedding
            embedding_zarr = t_zarr.zeros('embedding', shape=embedding.shape, dtype=embedding.dtype)
            embedding_zarr[:] = embedding


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("job_code", type=int)
    CLI.add_argument("dataset_id")
    CLI.add_argument("duration", type=int)
    CLI.add_argument("num_channels", type=int)
    CLI.add_argument("--into-events", default=False, action="store_true")
    args = CLI.parse_args()
    job_code = args.job_code
    dataset_id = args.dataset_id
    duration = args.duration
    num_channels = args.num_channels
    into_events = args.into_events
    logger = get_logger()
    embed(job_code, dataset_id, duration, num_channels, into_events)
