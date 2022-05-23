import logging
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
    
    def run(self):
        # begin experiment        
        self.logger.info(f"{self.config=}")

        # create times
        times = np.array([5, 10, 15, 20])

        groups = np.split(times, 2)
        
        root = zarr.open('data/embed.zarr', 'w')
        raw = root.create_group('data')
        embeddings = root.create_group('embeddings')

        # initialize times array
        root_zarr = zarr.open('data/job_inputs.zarr', mode='w')


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

            # submit Slurm Job(group)
            slurm = SlurmHandler(jobname='embed')
            slurm.submitJob(job_config)

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
    
    