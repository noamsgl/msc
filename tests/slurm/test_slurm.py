from msc.slurm_handler import SlurmHandler
from msc import get_logger
from msc import config

if __name__ == "__main__":
    logger = get_logger()
    
    # begin experiment        
    logger.info(f"{config=}")
    
    # submit Slurm Jobs
    slurm = SlurmHandler()
    
    job_code = 42
    dependencies = ['3742373']
    dependencies = None
    # submit test job
    job_config = {
        "job_name": "test",
        "job_code": job_code,
        "dataset_id": config['dataset_id'],
        "dependencies": dependencies,
        "gpus": 0
    }

    logger.info(f"submitting test job with {job_code=} to SlurmHandler.")
    slurm.submitJob(job_config)
