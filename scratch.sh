#!/bin/bash

CONFIG="config/config.yaml"
sbatch thesis/experiment/scripts/embed.batch $CONFIG
