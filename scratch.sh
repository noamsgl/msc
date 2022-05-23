#!/bin/bash


CONFIG="config/config.yaml"
sbatch thesis/experiment/scripts/embed.batch $CONFIG

watch -n 0.5 squeue --me
