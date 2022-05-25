import logging
import os
import sys
import yaml

import numpy as np
import zarr

from msc.datamodules.data_utils import IEEGDataFactory

import debugpy

from msc.slurm_handler import SlurmHandler

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
    * save & load data to disk               # perhaps hdf5
    * transform dataset z(x) for all x       # GP embedding transformation
    * estimate density p(z)                  # GMM
    * extract novelty score n(x)             # p-value
    * init prior p(S)                        # PyroModule
    * dump state
    """

    def __init__(self, config):
      self.config = config
      self.logger = self.get_logger()
      self.results = None
    
    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
     
        # create a formatter that creates a single line of json with a comma at the end
        formatter = logging.Formatter(
            (
                '{"unix_time":%(created)s, "time":"%(asctime)s", "module":"%(name)s",'
                ' "line_no":%(lineno)s, "level":"%(levelname)s", "msg":"%(message)s"},'
            )
        )
        
        # create a channel for handling the logger and set its format
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)

        # connect the logger to the channel
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        # send an example message
        logger.info('logging is working')
        return logger

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
        sample_times = np.random.randint(0, self.config['t_max'], size=N)
        sample_times.sort()
        samples = []
        
        # initialize nan counter
        segments_with_nan = 0

        # download samples
        for idx, t in enumerate(sample_times):
            data_t = ds.get_data(t, self.config['duration'], np.arange(self.config['num_channels']))
            samples.append(data_t)
            nan_count = np.count_nonzero(np.isnan(data_t))
            if nan_count != 0:
                self.logger.info(f"At time {t} ({idx}/{len(sample_times)}) there are {nan_count}/{data_t.size} ({nan_count/data_t.size:.0f}%) nan entries")
                segments_with_nan += 1
        self.logger.info(f"{segments_with_nan=}, total {N=}")
        data = np.vstack(samples)
        # calculate mean and std
        mu = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        # save mean and std to cache
        cache_zarr = zarr.open(f"{self.config['path']['data']}/cache.zarr")
        ds_zarr = cache_zarr.create_group(f"{self.config['dataset_id']}")
        mu_zarr = ds_zarr.zeros('mu', shape=mu.shape)
        mu_zarr[:] = mu
        std_zarr = ds_zarr.zeros('std', shape=std.shape)
        std_zarr[:] = std
        return mu, std

    def run(self):
        # begin experiment        
        self.logger.info(f"{self.config=}")
        
        # get dataset's mean and std from cache or create if nonexistent
        ds_stats_are_computed = False
        if os.path.exists(f"{config['path']['data']}/cache.zarr"):
            cache_zarr = zarr.open(f"{config['path']['data']}/cache.zarr", 'r')
            if self.config['dataset_id'] in cache_zarr:
                ds_zarr = cache_zarr[f"{self.config['dataset_id']}"]
                if ('mu' in ds_zarr) and ('std' in ds_zarr):
                    ds_stats_are_computed = True
                    mu = ds_zarr['mu'][:]
                    std = ds_zarr['std'][:]

        if not ds_stats_are_computed:
            mu, std = self.compute_ds_stats()

        # create times
        times = np.array([5, 10, 15, 20])

        # split times into groups (processes)
        groups = np.split(times, 2)
        
        # initialize times array
        root_zarr = zarr.open(f"{config['path']['data']}/job_inputs.zarr", mode='w')

        # initialize jobs array
        jobs = []

        # for each group of times, submit a Slurm job
        for job_code, job_times in enumerate(groups):
            # input times data
            job_zarr = root_zarr.create_group(str(job_code))
            times_zarr = job_zarr.zeros('times', shape=job_times.shape, dtype='i8')
            times_zarr[:] = job_times

            # create job configuration
            job_config = {
                "job_code": job_code,
                "job_times": job_times,
                "dataset_id": self.config['dataset_id'],
                "duration": self.config['duration'],
                "num_channels": self.config['num_channels']
            }
            # add job to jobs
            jobs.append(job_config)
        
        # submit Slurm Jobs
        slurm = SlurmHandler(jobname='embed')
        map(slurm.submitJob, jobs)

        # TODO: verify all jobs finished
        # TODO: collect results
        # TODO: analyze results
        results = None
        if results is not None:
            self.analyze_results(results)
        return results



class OnlineExperiment:
    """ This class orchestrates the online experiment from start to finish.
    The online eperiment consists of:
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

    experiment = OfflineExperiment(config)

    results = experiment.run()
    print(f"Recieved {results=}")
    
    