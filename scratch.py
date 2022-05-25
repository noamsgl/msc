import zarr

z = zarr.open('data/storage.zarr')

z.info

# TODO: add %jobid print to log
# TODO: add early stopping