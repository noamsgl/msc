#!/bin/bash

CONFIG="config/embeddor/gp.yaml"
sbatch thesis/experiment/scripts/embed.batch $CONFIG
