import os
import time
import subprocess
import random
import re
import textwrap
from typing import Tuple
import zarr
from msc.logs import get_logger

logger = get_logger()

class SlurmHandler(object):
    """Class to handle submission of jobs to a Slurm cluster"""
    def __init__(self, tmpdir="/home/noamsi/msc/scratch", logdir="/home/noamsi/msc/output", usermail="noamsi@post.bgu.ac.il"):
        self.name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.logdir = os.path.abspath(logdir)
        self.usermail = usermail
        self.tmpdir = tmpdir
        self.mail_type = "ALL"
        # self.mail_type = "NONE"


    def _slurmHeader(self, jobname, usermail, output_logs, error_logs, gpus) -> str:
        command = f"""#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:30:00
#SBATCH --job-name {jobname}
#SBATCH -e {error_logs}
#SBATCH -o {output_logs}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user={usermail}
#SBATCH --mail-type={self.mail_type}
#SBATCH --gpus={gpus}				### number of GPUs, allocating more than 1 requires IT team's permission

echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n\\n"

module load anaconda
source activate msc				### activate a conda environment, replace my_env with your conda environment
"""
        return command
    
    def _slurmFooter(self) -> str:
        command = """
# deactivate virtual environment
conda deactivate

# job end
date
"""
        return command

    def _slurmJobCommand(self, job_config) -> str:
        """unpack job_config, then dispatch command"""
        if job_config['job_name'] == "embed":
            # unpack job_config
            job_code = job_config['job_code']
            dataset_id = job_config['dataset_id']
            duration = job_config['duration']
            num_channels = job_config['num_channels']
            into_events = job_config['into_events']
            # assemble command
            command = f"python ~/msc/thesis/experiment/embedding/embeddor_agent.py {job_code} {dataset_id} {duration} {num_channels} {'--into-events' if into_events else ''}"

        elif job_config['job_name'] == "calc_likelihood":
            # unpack job_config
            dataset_id = job_config['dataset_id']
            # assemble command
            command = f"python ~/msc/thesis/experiment/calc_likelihood_agent.py {dataset_id}"

        elif job_config['job_name'] == 'test':
            command = f"python ~/msc/tests/test_agent.py"

        else:
            raise ValueError(f"{job_config['job_name']} is not supported.")
        return command

    def _slurmSubmitJob(self, jobFile, dependencies=('-1',)) -> Tuple[bytes, bytes]:
        """Submit command to shell."""
        if dependencies is None:
            dependencies=('-1',)
        command = f"sbatch --dependency=afterok:{':'.join(dependencies)} {jobFile}"
        logger.info(f"executing {command=}")
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        return p.communicate()

    def submitJob(self, job_config) -> str:
        """submit job to Slurm job scheduler.
        returns jobID as string.
        """
        # unpack job_config
        job_name = job_config['job_name']
        job_code = job_config['job_code']
        dependencies = job_config.get('dependencies', ('-1',))
        gpus = job_config.get('gpus', 0)

        # define logging paths
        output_logs = f"output/{job_name}_{job_code}.out"
        error_logs = f"output/{job_name}_{job_code}.err"

        # define job script path
        job_file = f"{self.tmpdir}/{job_name}_{job_code}.sh"  # e.g. .../embed_0.sh
    
        # assemble job commands
        job_commands = self._slurmJobCommand(job_config)

        # assemble job file
        job = self._slurmHeader(f"{job_name}_{job_code}", self.usermail, output_logs, error_logs, gpus) + job_commands + self._slurmFooter()
        
        # write job file to disk
        with open(job_file, 'w') as handle:
            handle.write(textwrap.dedent(job))
        
        # submit job
        output, err = self._slurmSubmitJob(job_file, dependencies)
        output_text = output.decode('utf-8')
        jobID = re.sub("\D", "", output_text)
        logger.info(f"submitted {jobID=}")
        return jobID

