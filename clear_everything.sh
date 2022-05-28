#!/bin/bash

DATASET_ID=I004_A0003_D001

# clear logs
rm -r output/*

# clear temporary shell scripts
rm -r scratch/*

# clear temporary job inputs
rm -r data/job_inputs*

# clear time points cache
find data/cache.zarr/$DATASET_ID -depth | egrep "$DATASET_ID/[0-9]" | xargs -r rm -rf

# clear synchronization 
rm -r data/embed.sync*

# clear results
rm -r results/*
