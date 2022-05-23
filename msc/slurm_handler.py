import os
import time
import subprocess
import random
import re
import textwrap
from typing import Tuple
import zarr

class SlurmHandler(object):
    """Class to hand submission of jobs to a Slurm cluster"""
    def __init__(self, jobname="embed", tmpdir="/home/noamsi/msc/scratch", logdir="/home/noamsi/msc/output", usermail="noamsi@post.bgu.ac.il"):
        self.name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.logdir = os.path.abspath(logdir)
        self.usermail = usermail
        self.jobname = jobname
        self.tmpdir = tmpdir


    def _slurmHeader(self, jobname, usermail, output_logs, error_logs) -> str:
        command = f"""#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:30:00
#SBATCH --job-name {jobname}
#SBATCH -e {error_logs}
#SBATCH -o {output_logs}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user={usermail}
#SBATCH --mail-type=ALL
#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission

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

    def _slurmSubmitJob(self, jobFile) -> Tuple[bytes, bytes]:
        """Submit command to shell."""
        command = f"sbatch {jobFile}"
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        return p.communicate()

    def submitJob(self, job_config) -> None:
        """Add task to be performed with data."""
        job_code = job_config['job_code']
        job_times = job_config['job_times']
        dataset_id = job_config['dataset_id']
        duration = job_config['duration']
        num_channels = job_config['num_channels']
        
        # define logs paths
        output_logs = f"output/{self.jobname}_{job_code}.out"
        error_logs = f"output/{self.jobname}_{job_code}.err"

        # define job file path
        job_file = f"{self.tmpdir}/embed_{job_code}.sh"
    
        # assemble job commands
        job_commands = f"python ~/msc/thesis/experiment/embeddor_agent.py {job_code} {dataset_id} {duration} {num_channels}"

        # assemble job file
        job = self._slurmHeader(f"{self.jobname}_{job_code}", self.usermail, output_logs, error_logs) + job_commands + self._slurmFooter()
        
        # write job file to disk
        with open(job_file, 'w') as handle:
            handle.write(textwrap.dedent(job))
        
        output, err = self._slurmSubmitJob(job_file)
        output_text = output.decode('utf-8')
        jobID = re.sub("\D", "", output_text)
        print(f"{jobID=}")
        return None

