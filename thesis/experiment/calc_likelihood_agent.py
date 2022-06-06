import argparse
import sys
from joblib import dump
import numpy as np

from sklearn import mixture
from msc.cache_handler import get_samples_df

from msc.logs import get_logger
from msc import config

def calc_likelihood(dataset_id) -> None:
    logger.info(f"beginning calc_likelihood with {dataset_id=}")
    
    # load embeddings
    samples_df = get_samples_df(dataset_id)
    
    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(np.stack(samples_df['embedding']))  # type: ignore

    gmm_path = f"{config['path']['data']}/dataset_id/GMM.joblib"
    logger.info(f"persisting GMM to disk at {gmm_path}")
    dump(clf, gmm_path)

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("dataset_id", type=str)
    args = CLI.parse_args() 
    dataset_id = args.dataset_id
    logger = get_logger()
    calc_likelihood(dataset_id)