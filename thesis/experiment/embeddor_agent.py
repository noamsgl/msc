import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
import sys
import zarr

from msc.models.embedding import GPEmbeddor

from msc.datamodules.data_utils import IEEGDataFactory

def get_dataset(dataset_id):
    # get dataset from iEEG.org
    ds = IEEGDataFactory.get_dataset(dataset_id)
    return ds

def embed(job_code, dataset_id, duration, num_channels) -> None:
    print(f"beginning embedding with {job_code=} {dataset_id=} {duration=}")
    with initialize(config_path="../config/embeddor/"):
        cfg = compose(config_name="gp", overrides=[])
        gp : GPEmbeddor = hydra.utils.instantiate(cfg.embeddor)

        # get dataset
        ds = get_dataset(dataset_id)

        # get times from times zarr
        root_zarr = zarr.open('data/job_inputs.zarr', mode='r')
        times_zarr = root_zarr[f"{job_code}/times"]
        print(f"{times_zarr=}")

        # get data from iEEG.org and save to data zarr
        for t in times_zarr:
            data = ds.get_data(t, duration, np.arange(num_channels))
            
        embedding = gp.embed(data)


if __name__ == "__main__":
    job_code = sys.argv[1]
    dataset_id = sys.argv[2]
    duration = sys.argv[3]
    num_channels = sys.argv[4]
    embed(job_code, dataset_id, duration, num_channels)