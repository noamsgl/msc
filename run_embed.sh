#!/bin/bash


CONFIG="config/config.yaml"
sbatch thesis/experiment/scripts/embed.batch $CONFIG online

watch -n 0.5 'squeue --me | tail -n 20'
