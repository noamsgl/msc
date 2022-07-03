#!/bin/bash


CONFIG="config/config.yaml"
sbatch thesis/experiment/scripts/embed.batch $CONFIG offline

watch -n 0.5 'squeue --me | tail -n 20'
