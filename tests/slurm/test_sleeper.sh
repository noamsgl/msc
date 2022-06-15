#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:30:00
#SBATCH --job-name test
#SBATCH -e output/test_%J.err
#SBATCH -o output/test_%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=noamsi@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=0				### number of GPUs, allocating more than 1 requires IT team's permission

echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

module load anaconda
source activate msc				### activate a conda environment, replace my_env with your conda environment
python ~/msc/thesis/experiment/test_agent_sleeper.py
# deactivate virtual environment
conda deactivate

# job end
date
