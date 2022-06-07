import logging
import os
import sys
import yaml

import numpy as np
import zarr
from msc.config import get_authentication

from msc.datamodules.data_utils import IEEGDataFactory
from msc.data_utils import count_nans

import debugpy

from msc.slurm_handler import SlurmHandler
from msc import get_logger
from msc.logs import nan_report

# print("listening to client on localhost:5678")
# debugpy.listen(5678)
# print("waiting for client to attach")
# debugpy.wait_for_client()

class OfflineExperiment:
    """ This class orchestrates the offline experiment from start to finish.
    The offline eperiment consists of:
    * get config in init
    * initialize sample times array
    * partition into 100 workers working iteratively
    * Each agents script
    * connect to dataset
    * 
    * save & load data to disk               # zarr
    * transform dataset z(x) for all x       # GP embedding transformation
    * estimate density p(z)                  # GMM to joblib
    * extract novelty score n(x)             # p-value
    * init prior p(S)                        # perhaps PyroModule
    * dump state
    """

    def __init__(self, config):
      self.config = config
      self.logger = get_logger()
      self.results = None
    

    def get_dataset(self):
        # get dataset from iEEG.org
        ds_id = self.config['dataset_id']
        ds = IEEGDataFactory.get_dataset(ds_id)
        return ds
    
    def analyze_results(self, results):
        assert self.results is not None, "error: self.results is None"
        raise NotImplementedError()
    
    def compute_ds_stats(self, N=400):
        """A function to estimate a dataset's mean and standard deviation per channel"""
        ds = self.get_dataset()
        # initialize sample times
        sample_times = self.get_sample_times(N)
        sample_times.sort()
        samples = []
        
        # initialize nan counter
        segments_with_nan = 0

        # download samples
        for idx, t in enumerate(sample_times):
            data_t = ds.get_data(t, self.config['duration'], np.arange(self.config['num_channels']))
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

    def get_sample_times(self, N=200):
        """subsample uniformly N time points between 0 and config['t_max']"""
        np.random.seed(self.config['random_seed'])
        times = np.random.randint(0, self.config['t_max'], size=N)
        return times
    
    def get_event_sample_times(self, augment=True):
        ds = self.get_dataset()
        seizures = ds.get_annotations('seizures')
        seizure_onsets_usec = np.array([seizure.start_time_offset_usec for seizure in seizures])
        # convert usec to sec
        times = seizure_onsets_usec / 1e6
        # append pre-ictal segments
        if augment:
            times = np.concatenate([times, times-5, times-10, times-15, times-20, times-25, times-30])
        return times.astype(int)

    def run(self, into_events=False):
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
            mu, std = self.compute_ds_stats()
        
        self.logger.info(f"{mu=}")
        self.logger.info(f"{std=}")

        # create times
        if into_events:
            times = self.get_event_sample_times()
        else:
            times = self.get_sample_times(N=self.config['n_embeddings'])

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

        # submit analysis job
        job_config = {
            "job_name": "calc_likelihood",
            "job_code": job_code,
            "dataset_id": self.config['dataset_id'],
            "dependencies": jobIDs,
            "gpus": 1  # in order to be on same cluster as prev jobs
        }

        self.logger.info(f"submitting calc_likelihood job with {job_code=} to SlurmHandler.")
        slurm.submitJob(job_config)

        self.logger.info(f"all OfflineExperiment jobs submitted.")
        return results



class OnlineExperiment:
    """ This class orchestrates the online experiment.
    The online experiment consists of:
    * load state
    * init times
    * for t in times:
    * init x_t = ds[t]
    * estimate p(S_t)
    * estimate n(x_t)
    * multiply p(S|X) = n(x_t) * p(S_t)
    * save to results.csv 
    """
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    print("Beginning experiment")
    print(f"{sys.argv}")
    config_fpath = sys.argv[1]
    config = yaml.safe_load(open(f'{config_fpath}', 'r'))
    print(f"{config=}")
    np.random.seed(config['random_seed'])
    experiment = OfflineExperiment(config)

    results = experiment.run()
    print(f"Recieved {results=}")
    
    