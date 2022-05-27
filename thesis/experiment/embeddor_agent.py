import hydra
from hydra import initialize, compose
import logging
from omegaconf import DictConfig, OmegaConf
import numpy as np
import socket

import sys
import zarr

from msc import config
from msc.models.embedding import GPEmbeddor
from msc.datamodules.data_utils import IEEGDataFactory
from msc import get_logger

def get_dataset(dataset_id):
    # get dataset from iEEG.org
    ds = IEEGDataFactory.get_dataset(dataset_id)
    return ds


def embed(job_code, dataset_id, duration, num_channels) -> None:
    logger = get_logger()
    logger.info(f"beginning embedding with {job_code=} {dataset_id=} {duration=} {num_channels=}")
    # get dataset
    ds = get_dataset(dataset_id)

    # parallel access synchronization
    synchronizer = zarr.ProcessSynchronizer(f"{config['path']['data']}/embed.sync")
    cache_zarr = zarr.open_group(f"{config['path']['data']}/cache.zarr", 'a', synchronizer=synchronizer)
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
        t_zarr = ds_zarr.create_group(f"{t}")

        # get data from iEEG.org
        data = ds.get_data(t, duration, np.arange(num_channels))

        # normalize data
        data = (data - mu)/std

        # save data to {t}/data
        data_zarr = t_zarr.zeros('data', shape=data.shape, dtype=data.dtype)
        data_zarr[:] = data

        logger.info(f"{data=}")
        if not np.all(np.isfinite(data)):
            logger.warn(f"{data=} contains nan entries. skipping.")

        # initialize gp
        with initialize(config_path="../../config/embeddor/"):
            cfg = compose(config_name="gp", overrides=[])
            gp : GPEmbeddor = hydra.utils.instantiate(cfg.embeddor)
            # initialize pytorch-lightning logging directory
            pl_logger_dirpath = f"{config['path']['lightning_logs']}/{dataset_id}/{t}"
            # begin embedding process
            embedding = gp.embed(data, pl_logger_dirpath)
            # save embedding to {t}/embedding
            embedding_zarr = t_zarr.zeros('embedding', shape=embedding.shape, dtype=embedding.dtype)
            embedding_zarr[:] = embedding


if __name__ == "__main__":
    job_code = int(sys.argv[1])
    dataset_id = str(sys.argv[2])
    duration = int(sys.argv[3])
    num_channels = int(sys.argv[4])
    logger = get_logger()
    embed(job_code, dataset_id, duration, num_channels)
