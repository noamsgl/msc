import os
import sys
import yaml
import numpy as np
import zarr

from msc import get_logger
from msc.data_utils import count_nans, get_dataset, get_event_sample_times, get_sample_times
from msc.logs import nan_report
from msc.slurm_handler import SlurmHandler


class parallel_embeddor:
    """ This class orchestrates multimachine (slurm) parallel embedding.
    """

    def __init__(self, config, ds, mode:str):
      assert mode in ('online', 'offline'), f"error: {mode=} should be 'online' or 'offline'"
      self.config = config
      self.ds = ds
      self.mode = mode
      self.logger = get_logger()
    
    def compute_ds_stats(self, N):
        """A function to estimate a dataset's mean and standard deviation, channel-wise"""
        # initialize sample times
        sample_times = get_sample_times(N, 'offline')
        sample_times.sort()
        samples = []
        
        # initialize nan counter
        segments_with_nan = 0

        # download samplesl
        for idx, t in enumerate(sample_times):  # type: ignore
            data_t = self.ds.get_data(t, self.config['duration'], np.arange(self.config['num_channels']))
            samples.append(data_t)
            nan_count = count_nans(data_t)
            if nan_count != 0:
                self.logger.info(f"At time {t} ({idx}/{len(sample_times)}) {nan_report(data_t)}")
                segments_with_nan += 1
        self.logger.info(f"{segments_with_nan=}, total {N=}")
        data = np.vstack(samples)
        # calculate mean and std
        mu = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        # save mean and std to cache
        cache_zarr = zarr.open(f"{self.config['path']['data']}/cache.zarr", 'a')
        ds_zarr = cache_zarr.create_group(f"{self.config['dataset_id']}", 'w')
        mu_zarr = ds_zarr.zeros('mu', shape=mu.shape)
        mu_zarr[:] = mu
        std_zarr = ds_zarr.zeros('std', shape=std.shape)
        std_zarr[:] = std
        return mu, std

    def run(self, into_events=False) -> None:
        # begin experiment        
        self.logger.info(f"{self.config=}")
        
        # get dataset's mean and std from cache or create if nonexistent
        mu = None
        std = None
        ds_stats_are_computed = False
        if os.path.exists(f"{config['path']['data']}/cache.zarr"):
            cache_zarr = zarr.open(f"{config['path']['data']}/cache.zarr", 'r')
            if self.config['dataset_id'] in cache_zarr:
                ds_zarr = cache_zarr[f"{self.config['dataset_id']}"]
                if ('mu' in ds_zarr) and ('std' in ds_zarr):
                    ds_stats_are_computed = True
                    mu = ds_zarr['mu'][:]
                    std = ds_zarr['std'][:]
                    self.logger.info("found mu and std in cache. reusing.")

        if not ds_stats_are_computed:
            mu, std = self.compute_ds_stats(self.config['n_ds_stats'])
        
        self.logger.info(f"{mu=}")
        self.logger.info(f"{std=}")

        # create times
        if into_events:  # event samples
            times = get_event_sample_times(self.ds, augment=True)
        else:  # random subsampling
            t_end = int((self.ds.end_time - self.ds.start_time) / 1e6)
            times = get_sample_times(self.config['n_embeddings'], self.mode, t_end)

        # split times into groups (processes)
        groups = np.array_split(times, self.config['n_jobs'])
        
        # initialize times array
        root_zarr = zarr.open(f"{config['path']['data']}/job_inputs.zarr", mode='w')

        # initialize jobs array
        jobs = []
        job_code = 0
        # for each group of times, submit a Slurm job
        for job_times in groups:
            # persist job times in cache
            job_zarr = root_zarr.require_group(str(job_code))
            times_zarr = job_zarr.zeros('times', shape=job_times.shape, dtype='i8')
            times_zarr[:] = job_times

            # create job configuration
            job_config = {
                "job_name": "embed",
                "job_code": job_code,
                "job_times": job_times,
                "dataset_id": self.config['dataset_id'],
                "duration": self.config['duration'],
                "num_channels": self.config['num_channels'],
                "into_events": into_events,
                "gpus": 1
            }
            # add job to jobs
            jobs.append(job_config)
            # increase job_code
            job_code += 1
        
        # submit Slurm Jobs
        slurm = SlurmHandler()
        self.logger.info(f"submitting {len(jobs)} embed jobs to SlurmHandler.")
        jobIDs = []  # for keeping track of dependencies
        for job_config in jobs:
            jobID = slurm.submitJob(job_config)
            jobIDs.append(jobID)

        self.logger.info(f"all OfflineExperiment jobs submitted.")
        return


if __name__ == "__main__":
    print("Beginning embedding")
    config_fpath = sys.argv[1]
    mode = sys.argv[2]
    assert mode in ['offline', 'online']
    
    config = yaml.safe_load(open(f'{config_fpath}', 'r'))
    ds = get_dataset(config['dataset_id'])
    embeddor = parallel_embeddor(config, ds, mode=mode)

    embeddor.run()
    print(f"finished gracefully")
    
    