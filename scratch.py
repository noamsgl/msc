import zarr

z = zarr.open('data/cache.zarr')

z.info

# TODO: move std,mu to zarr attrs
print("Hello world")
